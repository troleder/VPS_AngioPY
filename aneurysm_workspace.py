import os
import io
import time
import math
import re
import json
import zipfile
import shutil
import base64
import numpy as np
import pydicom
from PIL import Image
import streamlit as st
import cv2
import scipy.ndimage
import scipy.interpolate
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import angioPyFunctions
import predict

# Storage base for Sylwia's uploaded patient cases
ANEURYSM_STORAGE_DIR = "/var/www/analiza-dicom/aneurysm_storage/sylwia"
if not os.path.exists(os.path.dirname(ANEURYSM_STORAGE_DIR)):
    # Fallback for local testing
    ANEURYSM_STORAGE_DIR = os.path.abspath("./aneurysm_storage/sylwia")

def ensure_storage_dir():
    try:
        os.makedirs(ANEURYSM_STORAGE_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error creating storage dir: {e}")

def get_aneurysm_patients():
    ensure_storage_dir()
    if not os.path.exists(ANEURYSM_STORAGE_DIR):
        return []
    pats = []
    try:
        for fn in sorted(os.listdir(ANEURYSM_STORAGE_DIR)):
            p_path = os.path.join(ANEURYSM_STORAGE_DIR, fn)
            if os.path.isdir(p_path) and not fn.startswith('.'):
                pats.append(fn)
    except Exception as e:
        print(f"Error reading aneurysm patients: {e}")
    return pats

def sanitize_folder_name(name):
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name.strip())
    return clean if clean else "Patient_Aneurysm"

def load_dicom_file(filepath):
    try:
        dcm = pydicom.dcmread(filepath, force=True)
        pixel_array = dcm.pixel_array
        if len(pixel_array.shape) == 4:
            pixel_array = pixel_array[:, :, :, 0]
        elif len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:
            pixel_array = pixel_array[np.newaxis, :, :, 0]
        elif len(pixel_array.shape) == 2:
            pixel_array = pixel_array[np.newaxis, ...]
        
        # Gantry angles
        primary_angle = getattr(dcm, "PositionerPrimaryAngle", None)
        secondary_angle = getattr(dcm, "PositionerSecondaryAngle", None)
        try: primary_angle = float(primary_angle) if primary_angle is not None else 0.0
        except: primary_angle = 0.0
        try: secondary_angle = float(secondary_angle) if secondary_angle is not None else 0.0
        except: secondary_angle = 0.0
        
        # Pixel Spacing
        spacing = getattr(dcm, "ImagerPixelSpacing", None)
        if spacing is not None:
            try: spacing = float(spacing[0])
            except: spacing = 0.20
        else:
            spacing = 0.20

        # Distance magnification factor
        try:
            if hasattr(dcm, "DistanceSourceToPatient") and hasattr(dcm, "DistanceSourceToDetector"):
                ds_pat = float(dcm.DistanceSourceToPatient)
                ds_det = float(dcm.DistanceSourceToDetector)
                if ds_det > 0 and ds_pat > 0:
                    spacing = spacing * (ds_pat / ds_det)
        except Exception:
            pass
            
        series_desc = getattr(dcm, "SeriesDescription", "") or os.path.basename(filepath)
        cine_rate = getattr(dcm, "CineRate", 15) or 15
        
        return {
            "pixels": pixel_array,
            "primary_angle": primary_angle,
            "secondary_angle": secondary_angle,
            "spacing": spacing,
            "series_desc": series_desc,
            "cine_rate": int(cine_rate),
            "total_frames": pixel_array.shape[0],
            "dcm_obj": dcm
        }
    except Exception as e:
        print(f"Error loading DICOM {filepath}: {e}")
        return None

def compute_3d_angle_diff(alpha1, beta1, alpha2, beta2):
    """
    Computes true 3D spatial angle between two projection vectors in degrees.
    alpha = PositionerPrimaryAngle (LAO/RAO)
    beta  = PositionerSecondaryAngle (CRA/CAU)
    """
    a1, b1 = math.radians(alpha1), math.radians(beta1)
    a2, b2 = math.radians(alpha2), math.radians(beta2)
    
    v1 = np.array([math.sin(a1) * math.cos(b1), -math.sin(b1), math.cos(a1) * math.cos(b1)])
    v2 = np.array([math.sin(a2) * math.cos(b2), -math.sin(b2), math.cos(a2) * math.cos(b2)])
    
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = math.acos(dot)
    return math.degrees(angle_rad)

def simpsons_volume_biplane(d1_max, d2_max, ref1, ref2, length_mm, n_slices=20):
    """
    Computes aneurysm 3D lumen volume using Simpson's rule of elliptical discs.
    """
    if length_mm <= 0 or d1_max <= 0 or d2_max <= 0:
        return 0.0, 0.0
        
    s = np.linspace(0, length_mm, n_slices + 1)
    ds = length_mm / n_slices
    
    sin_profile = np.sin(np.pi * s / length_mm)
    d1_profile = ref1 + (d1_max - ref1) * sin_profile
    d2_profile = ref2 + (d2_max - ref2) * sin_profile
    
    areas = (np.pi / 4.0) * d1_profile * d2_profile
    
    weights = np.ones(len(areas))
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0
    
    total_volume_mm3 = (ds / 3.0) * np.sum(weights * areas)
    ref_areas = (np.pi / 4.0) * ref1 * ref2
    healthy_volume_mm3 = ref_areas * length_mm
    excess_volume_mm3 = max(0.0, total_volume_mm3 - healthy_volume_mm3)
    
    return total_volume_mm3, excess_volume_mm3

def extract_aneurysm_profile(mask_2d, mm_per_pixel=0.20):
    """
    Analyzes binary mask (512x512) of artery/aneurysm:
    - Finds main contour
    - Prunes and orders skeleton path
    - Fits B-spline to centerline
    - Raycasts orthogonal lumen thicknesses along centerline
    - Auto-detects landmarks: prox_idx, dist_idx, max_idx, and length
    """
    try:
        mask_u8 = (mask_2d > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
            
        main_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(main_cnt) < 40:
            return None
            
        clean_mask = np.zeros_like(mask_u8)
        cv2.drawContours(clean_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
        
        mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        skel = angioPyFunctions.skeletonise(mask_bgr)
        start_pt, end_pt = angioPyFunctions.skelEndpoints(skel)
        ordered_pts = angioPyFunctions.skelPointsInOrder(skel, start_pt)
        
        if len(ordered_pts) < 10:
            return None
            
        EDT = scipy.ndimage.distance_transform_edt(clean_mask)
        tck = angioPyFunctions.skelSplinerWithThickness(skel=skel, EDT=EDT, smoothing=30)
        
        num_eval = 200
        sp_y, sp_x, sp_t = scipy.interpolate.splev(np.linspace(0.0, 1.0, num_eval), tck)
        
        # Clip ends to avoid boundary artifacts
        clip = 8
        sp_x = sp_x[clip:-clip]
        sp_y = sp_y[clip:-clip]
        sp_dx = np.gradient(sp_x)
        sp_dy = np.gradient(sp_y)
        
        _diffs = np.hypot(np.diff(sp_x), np.diff(sp_y))
        cum_dist_px = np.concatenate([[0], np.cumsum(_diffs)])
        cum_dist_mm = cum_dist_px * mm_per_pixel
        
        # Orthogonal raycasting for true vessel diameter
        _h, _w = clean_mask.shape
        _mask_bool = clean_mask > 0
        def _raycast(cx, cy, nx, ny, max_r=160):
            for r in range(1, max_r + 1):
                rx = int(round(cx + nx * r))
                ry = int(round(cy + ny * r))
                if rx < 0 or ry < 0 or rx >= _w or ry >= _h or not _mask_bool[ry, rx]:
                    return float(r - 1)
            return float(max_r)
            
        thickness_px = np.empty(len(sp_x), dtype=np.float32)
        for i in range(len(sp_x)):
            tx, ty = sp_dx[i], sp_dy[i]
            L = np.hypot(tx, ty)
            if L > 0:
                nx, ny = -ty / L, tx / L
                rp = _raycast(sp_x[i], sp_y[i], nx, ny)
                rn = _raycast(sp_x[i], sp_y[i], -nx, -ny)
                thickness_px[i] = rp + rn
            else:
                thickness_px[i] = sp_t[clip + i] * 2.0
                
        # Smooth thicknesses slightly
        kernel = np.ones(5) / 5.0
        thickness_px = np.convolve(thickness_px, kernel, mode='same')
        thickness_px[:2] = thickness_px[2]
        thickness_px[-2:] = thickness_px[-3]
        thickness_mm = thickness_px * mm_per_pixel
        
        N = len(sp_x)
        prox_idx = max(2, int(N * 0.15))
        dist_idx = min(N - 3, int(N * 0.85))
        if prox_idx >= dist_idx:
            prox_idx, dist_idx = 2, N - 3
            
        # Max diameter in range [prox_idx, dist_idx]
        max_idx = prox_idx + int(np.argmax(thickness_px[prox_idx:dist_idx+1]))
        
        ref_prox_mm = round(float(thickness_mm[prox_idx]), 2)
        ref_dist_mm = round(float(thickness_mm[dist_idx]), 2)
        max_diam_mm = round(float(thickness_mm[max_idx]), 2)
        interp_ref_mm = round((ref_prox_mm + ref_dist_mm) / 2.0, 2)
        
        # Aneurysm length between landmarks
        aneurysm_len_mm = round(float(cum_dist_mm[dist_idx] - cum_dist_mm[prox_idx]), 2)
        if aneurysm_len_mm < 1.0:
            aneurysm_len_mm = 5.0
            
        return {
            "sp_x": sp_x,
            "sp_y": sp_y,
            "sp_dx": sp_dx,
            "sp_dy": sp_dy,
            "cum_dist_mm": cum_dist_mm,
            "thickness_px": thickness_px,
            "thickness_mm": thickness_mm,
            "prox_idx": prox_idx,
            "dist_idx": dist_idx,
            "max_idx": max_idx,
            "ref_prox_mm": max(0.5, ref_prox_mm),
            "ref_dist_mm": max(0.5, ref_dist_mm),
            "interp_ref_mm": max(0.5, interp_ref_mm),
            "max_diam_mm": max(0.5, max_diam_mm),
            "aneurysm_len_mm": max(1.0, aneurysm_len_mm),
            "mm_per_pixel": mm_per_pixel,
            "clean_mask": clean_mask
        }
    except Exception as e:
        print(f"Error in extract_aneurysm_profile: {e}")
        return None

def render_aneurysm_overlay(base_img_512, mask_2d, profile, landmarks=None, zoom=1.0, focus_x=256, focus_y=256, mld_idx=None):
    """
    Renders high-contrast clinical overlay:
    - Base angiographic frame
    - Green contours for vessel
    - Amber highlight over aneurysm dilation
    - Cyan centerline
    - Calipers with measurement tags (Prox, Dist, Dmax, optional MLD)
    """
    if len(base_img_512.shape) == 2:
        overlay = cv2.cvtColor(base_img_512, cv2.COLOR_GRAY2RGB)
    else:
        overlay = base_img_512.copy()
        
    if mask_2d is not None and np.sum(mask_2d) > 0:
        mask_u8 = (mask_2d > 0).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Highlight dilated aneurysm body in translucent amber
        amber_layer = overlay.copy()
        amber_layer[mask_u8 > 0] = [245, 158, 11]
        cv2.addWeighted(amber_layer, 0.35, overlay, 0.65, 0, overlay)
        
        # Outer vessel boundary in bright green
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)
        
    if profile is not None:
        sp_x = profile["sp_x"]
        sp_y = profile["sp_y"]
        sp_dx = profile["sp_dx"]
        sp_dy = profile["sp_dy"]
        thick_px = profile["thickness_px"]
        mm_pp = profile["mm_per_pixel"]
        
        p_idx = landmarks.get("prox", profile["prox_idx"]) if landmarks else profile["prox_idx"]
        d_idx = landmarks.get("dist", profile["dist_idx"]) if landmarks else profile["dist_idx"]
        m_idx = landmarks.get("max", profile["max_idx"]) if landmarks else profile["max_idx"]
        
        p_idx = int(np.clip(p_idx, 0, len(sp_x) - 1))
        d_idx = int(np.clip(d_idx, 0, len(sp_x) - 1))
        m_idx = int(np.clip(m_idx, 0, len(sp_x) - 1))
        
        # Draw Centerline (cyan)
        pts_centerline = np.vstack((sp_x, sp_y)).T.astype(np.int32)
        cv2.polylines(overlay, [pts_centerline], isClosed=False, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        
        def draw_caliper(idx, color, label_text, is_bold=False):
            cx, cy = sp_x[idx], sp_y[idx]
            tx, ty = sp_dx[idx], sp_dy[idx]
            L = np.hypot(tx, ty)
            if L > 0:
                nx, ny = -ty / L, tx / L
                radius = thick_px[idx] / 2.0
                p1 = (int(round(cx + nx * radius)), int(round(cy + ny * radius)))
                p2 = (int(round(cx - nx * radius)), int(round(cy - ny * radius)))
                thick = 3 if is_bold else 2
                cv2.line(overlay, p1, p2, color, thick, lineType=cv2.LINE_AA)
                
                # Caliper end ticks
                tick_len = 5 if is_bold else 3
                t1a = (int(round(p1[0] + tx/L * tick_len)), int(round(p1[1] + ty/L * tick_len)))
                t1b = (int(round(p1[0] - tx/L * tick_len)), int(round(p1[1] - ty/L * tick_len)))
                t2a = (int(round(p2[0] + tx/L * tick_len)), int(round(p2[1] + ty/L * tick_len)))
                t2b = (int(round(p2[0] - tx/L * tick_len)), int(round(p2[1] - ty/L * tick_len)))
                cv2.line(overlay, t1a, t1b, color, 1, lineType=cv2.LINE_AA)
                cv2.line(overlay, t2a, t2b, color, 1, lineType=cv2.LINE_AA)
                
                # Label
                lx = int(max(p1[0], p2[0]) + 6)
                ly = int((p1[1] + p2[1]) / 2)
                lx = min(420, max(10, lx))
                ly = min(500, max(20, ly))
                cv2.putText(overlay, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Proximal caliper (Green)
        prox_mm = thick_px[p_idx] * mm_pp
        draw_caliper(p_idx, (50, 255, 50), f"Ref Prox: {prox_mm:.1f} mm")
        
        # Distal caliper (Green)
        dist_mm = thick_px[d_idx] * mm_pp
        draw_caliper(d_idx, (50, 255, 50), f"Ref Dist: {dist_mm:.1f} mm")
        
        # Max diameter caliper (Bright Red)
        max_mm = thick_px[m_idx] * mm_pp
        draw_caliper(m_idx, (255, 50, 50), f"Dmax: {max_mm:.1f} mm", is_bold=True)
        
        # MLD caliper (if present)
        if mld_idx is not None and 0 <= mld_idx < len(sp_x):
            mld_mm = thick_px[mld_idx] * mm_pp
            draw_caliper(mld_idx, (50, 150, 255), f"MLD: {mld_mm:.1f} mm")
            
    # Apply zoom and pan if requested
    if zoom > 1.0:
        h, w = overlay.shape[:2]
        hz, wz = int(h / zoom), int(w / zoom)
        top = int(np.clip(focus_y - hz // 2, 0, h - hz))
        left = int(np.clip(focus_x - wz // 2, 0, w - wz))
        crop = overlay[top:top+hz, left:left+wz]
        overlay = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        
    return overlay

def render_coronary_aneurysm_workspace():
    st.markdown("<h1 style='color: #38bdf8; font-family: Outfit, sans-serif;'>🩺 Coronary Artery Aneurysm (CAA) Workspace</h1>", unsafe_allow_html=True)
    st.markdown("Dedicated platform for quantitative assessment, morphological classification, eccentricity, and 3D Simpson volume analysis of coronary aneurysms.")
    
    ensure_storage_dir()
    
    # ── 1. PATIENT SELECTION & UPLOAD LIBRARY ─────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 My Patient Cases (Private Library)")
    
    patients = get_aneurysm_patients()
    
    if "aneurysm_patient_id" not in st.session_state:
        st.session_state.aneurysm_patient_id = patients[0] if patients else ""
        
    selected_pat = st.sidebar.selectbox(
        "Select Patient Case:",
        options=["-- Select Patient --"] + patients if patients else ["(No cases uploaded yet)"],
        index=(patients.index(st.session_state.aneurysm_patient_id) + 1) if (patients and st.session_state.aneurysm_patient_id in patients) else 0,
        key="aneurysm_patient_select"
    )
    if selected_pat and selected_pat not in ["-- Select Patient --", "(No cases uploaded yet)"]:
        st.session_state.aneurysm_patient_id = selected_pat
        
    # Uploader Expander in sidebar
    with st.sidebar.expander("📤 Upload New Patient Case (ZIP / Folder)", expanded=(len(patients) == 0)):
        st.markdown("Upload DICOM files or a ZIP archive from your computer. The archive will be saved permanently in your private server library.")
        uploaded_zip = st.file_uploader("Choose DICOM ZIP Archive:", type=["zip"], key="aneurysm_zip_uploader")
        
        default_name = ""
        if uploaded_zip:
            default_name = os.path.splitext(uploaded_zip.name)[0]
            
        custom_patient_name = st.text_input("Patient ID / Study Name:", value=default_name, key="aneurysm_custom_pid_input")
        
        if uploaded_zip and st.button("🚀 Upload & Save to Library", key="btn_save_aneurysm_zip", use_container_width=True):
            pid_clean = sanitize_folder_name(custom_patient_name if custom_patient_name.strip() else uploaded_zip.name)
            target_patient_dir = os.path.join(ANEURYSM_STORAGE_DIR, pid_clean)
            os.makedirs(target_patient_dir, exist_ok=True)
            
            with st.spinner(f"Extracting and saving case '{pid_clean}' to private storage..."):
                try:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zf:
                        valid_members = [m for m in zf.infolist() if not m.filename.startswith('__MACOSX') and not os.path.basename(m.filename).startswith('.')]
                        for m in valid_members:
                            zf.extract(m, target_patient_dir)
                    st.success(f"✅ Successfully uploaded and saved case: **{pid_clean}**")
                    st.session_state.aneurysm_patient_id = pid_clean
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error extracting ZIP archive: {e}")
                    
    # Active patient check
    active_pid = st.session_state.get("aneurysm_patient_id")
    if not active_pid or active_pid in ["-- Select Patient --", "(No cases uploaded yet)"]:
        st.info("👋 Welcome! Please upload your patient DICOM archive or select an existing case from the left sidebar to begin analysis.")
        return

    patient_dir = os.path.join(ANEURYSM_STORAGE_DIR, active_pid)
    if not os.path.exists(patient_dir):
        st.warning(f"Patient directory for '{active_pid}' was not found.")
        return

    # Find all DICOM files in patient directory recursively
    dicom_files = []
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if f.startswith('.'): continue
            fp = os.path.join(root, f)
            if f.lower().endswith('.dcm') or os.path.getsize(fp) > 132:
                dicom_files.append(fp)

    if not dicom_files:
        st.warning(f"No DICOM series found in folder '{active_pid}'. Please verify the uploaded archive contains valid DICOM files.")
        return

    # ── 2. DICOM SERIES SELECTION ─────────────────────────────────────────
    st.markdown(f"### 🗂️ Active Patient: **{active_pid}** *(Found {len(dicom_files)} DICOM files)*")
    
    series_map = {}
    for idx, dfp in enumerate(dicom_files):
        d_meta = load_dicom_file(dfp)
        if d_meta is not None:
            lbl = f"Series #{idx+1}: {d_meta['series_desc']} (LAO/RAO: {d_meta['primary_angle']}°, CRA/CAU: {d_meta['secondary_angle']}°, {d_meta['total_frames']} frames)"
            series_map[lbl] = (dfp, d_meta)

    if not series_map:
        st.error("Could not parse DICOM headers for this patient.")
        return

    selected_series_lbl = st.selectbox("Select Angiographic Sequence:", options=list(series_map.keys()), key="aneurysm_series_select")
    selected_dfp, d_meta = series_map[selected_series_lbl]

    # ── 3. CINE PLAYER & FRAME SELECTION ──────────────────────────────────
    n_frames = d_meta["total_frames"]
    frame_slider = st.slider("Frame Index:", min_value=0, max_value=max(0, n_frames - 1), value=min(int(n_frames / 2), n_frames - 1), key="aneurysm_frame_slider")
    
    # Extract and normalize frame to 512x512
    frame_pixels = d_meta["pixels"][frame_slider]
    norm_pixels = frame_pixels.astype(float)
    p_min, p_max = np.min(norm_pixels), np.max(norm_pixels)
    if p_max > p_min:
        norm_pixels = ((norm_pixels - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    else:
        norm_pixels = norm_pixels.astype(np.uint8)
        
    orig_h, orig_w = norm_pixels.shape
    if orig_h != 512 or orig_w != 512:
        norm_512 = cv2.resize(norm_pixels, (512, 512), interpolation=cv2.INTER_AREA)
        mm_per_pixel = d_meta["spacing"] * (orig_w / 512.0)
    else:
        norm_512 = norm_pixels
        mm_per_pixel = d_meta["spacing"]

    case_key = f"{active_pid}_{os.path.basename(selected_dfp)}_{frame_slider}"
    mask_key = f"caa_mask_{case_key}"
    prof_key = f"caa_prof_{case_key}"
    lm_key = f"caa_lm_{case_key}"
    points_key = f"caa_pts_{case_key}"
    
    active_mask = st.session_state.get(mask_key, None)
    active_profile = st.session_state.get(prof_key, None)
    active_landmarks = st.session_state.get(lm_key, None)

    # ── 4. TWO-COLUMN WORKSPACE: LEFT = VISUALIZER / SEGMENTATION, RIGHT = PARAMETERS
    col_vis, col_params = st.columns([1.3, 1])
    
    with col_vis:
        st.markdown("#### 🔬 Aneurysm Visualization & Segmentation")
        
        vis_tab1, vis_tab2, vis_tab3 = st.tabs([
            "👁️ Podgląd obrysu i kalipery",
            "🎯 Segmentuj naczynie (AI Guide)",
            "✏️ Korekta obrysu worka (Zoom & Nudge)"
        ])
        
        # ── TAB 1: VISUAL OVERLAY & LANDMARK SLIDERS ──────────────────────
        with vis_tab1:
            if active_mask is not None and active_profile is not None:
                overlay_img = render_aneurysm_overlay(
                    norm_512, active_mask, active_profile, active_landmarks
                )
                pil_overlay = Image.fromarray(overlay_img)
                try:
                    st.image(pil_overlay, caption=f"Obrys tętniaka | Klatka {frame_slider + 1}/{n_frames} | Kąty: {d_meta['primary_angle']}° / {d_meta['secondary_angle']}°", use_column_width=True)
                except Exception:
                    st.image(pil_overlay, caption=f"Obrys tętniaka | Klatka {frame_slider + 1}/{n_frames}")
                    
                with st.expander("📍 Skoryguj pozycje znaczników (Landmarks)", expanded=True):
                    st.caption("Przesuwaj suwaki, aby dokładnie skorygować punkty pomiaru referencji i najszerszego miejsca tętniaka wzdłuż osi naczynia:")
                    N_pts = len(active_profile["sp_x"])
                    
                    p_def = active_landmarks["prox"] if active_landmarks else active_profile["prox_idx"]
                    d_def = active_landmarks["dist"] if active_landmarks else active_profile["dist_idx"]
                    m_def = active_landmarks["max"] if active_landmarks else active_profile["max_idx"]
                    
                    c_sl1, c_sl2, c_sl3 = st.columns(3)
                    prox_override = c_sl1.slider("Ref Proksymalna", 0, N_pts - 1, int(p_def), key=f"sl_prox_{case_key}")
                    dist_override = c_sl2.slider("Ref Dystalna", 0, N_pts - 1, int(d_def), key=f"sl_dist_{case_key}")
                    max_override = c_sl3.slider("Max Dilation (Dmax)", 0, N_pts - 1, int(m_def), key=f"sl_max_{case_key}")
                    
                    c_lm_btn1, c_lm_btn2 = st.columns(2)
                    if c_lm_btn1.button("✅ Zastosuj pozycje znaczników", key=f"btn_apply_lm_{case_key}", use_container_width=True):
                        st.session_state[lm_key] = {
                            "prox": prox_override,
                            "dist": dist_override,
                            "max": max_override
                        }
                        st.rerun()
                        
                    if c_lm_btn2.button("↩ Przywróć wykryte automatycznie", key=f"btn_rev_lm_{case_key}", use_container_width=True):
                        st.session_state[lm_key] = {
                            "prox": active_profile["prox_idx"],
                            "dist": active_profile["dist_idx"],
                            "max": active_profile["max_idx"]
                        }
                        st.rerun()
            else:
                pil_raw = Image.fromarray(norm_512)
                try:
                    st.image(pil_raw, caption=f"Klatka {frame_slider + 1}/{n_frames} | Oczekiwanie na segmentację", use_column_width=True)
                except Exception:
                    st.image(pil_raw, caption=f"Klatka {frame_slider + 1}/{n_frames}")
                st.info("💡 Przejdź do zakładki **'🎯 Segmentuj naczynie (AI Guide)'**, aby kliknąć punkty naczynia i automatycznie wyznaczyć obrys tętniaka!")

        # ── TAB 2: AI SEGMENTATION (POINT GUIDANCE) ───────────────────────
        with vis_tab2:
            st.markdown(r"""
            **Instrukcja segmentacji:**
            1. Kliknij **2–4 punkty** wzdłuż tętnicy na obrazie poniżej (np. początek naczynia, worek tętniaka, koniec naczynia).
            2. Kliknij **'🚀 Segmentuj tętniak (AI)'**.
            """)
            
            canvas_key = f"seg_canvas_{case_key}_{st.session_state.get('caa_canvas_suffix', 0)}"
            pil_for_canvas = Image.fromarray(norm_512)
            
            annotation_canvas = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#22c55e",
                background_color="black",
                background_image=pil_for_canvas,
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode="point",
                point_display_radius=5,
                key=canvas_key
            )
            
            c_seg1, c_seg2 = st.columns(2)
            with c_seg1:
                if st.button("🚀 Segmentuj tętniak (AI)", type="primary", key=f"btn_run_seg_{case_key}", use_container_width=True):
                    if annotation_canvas.json_data is not None and "objects" in annotation_canvas.json_data:
                        objs = annotation_canvas.json_data["objects"]
                        if len(objs) >= 2:
                            with st.spinner(f"Segmentacja naczynia na podstawie {len(objs)} punktów..."):
                                try:
                                    pts = []
                                    for o in objs:
                                        pts.append([float(o.get('top', 0)), float(o.get('left', 0))])
                                    pts = np.array(pts, dtype=np.float32)
                                    
                                    mask = angioPyFunctions.arterySegmentation(norm_512, pts)
                                    mask_bin = (mask > 0).astype(np.uint8)
                                    
                                    prof = extract_aneurysm_profile(mask_bin, mm_per_pixel)
                                    if prof is not None:
                                        st.session_state[mask_key] = mask_bin
                                        st.session_state[prof_key] = prof
                                        st.session_state[lm_key] = {
                                            "prox": prof["prox_idx"],
                                            "dist": prof["dist_idx"],
                                            "max": prof["max_idx"]
                                        }
                                        st.session_state[points_key] = pts
                                        st.success("✅ Sukces: Tętniak został pomyślnie obrysowany!")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error("Nie udało się wyznaczyć osi naczynia. Upewnij się, że punkty leżą wzdłuż światła tętnicy.")
                                except Exception as e:
                                    st.error(f"Błąd podczas segmentacji: {e}")
                        else:
                            st.warning("Kliknij co najmniej 2 punkty na naczyniu przed uruchomieniem.")
                            
            with c_seg2:
                if st.button("🗑️ Wyczyść punkty", key=f"btn_clr_pts_{case_key}", use_container_width=True):
                    st.session_state["caa_canvas_suffix"] = st.session_state.get("caa_canvas_suffix", 0) + 1
                    st.rerun()

        # ── TAB 3: SMART CONTOUR CORRECTION & NUDGE ───────────────────────
        with vis_tab3:
            st.markdown("**Interaktywna korekta maski i obrysu worka tętniaka:**")
            st.caption("Powiększ obraz i dorysuj lub skoryguj obrys worka tętniaka (np. pominiętą skrzeplinę).")
            
            c_z1, c_z2, c_z3 = st.columns(3)
            zoom = c_z1.slider("🔍 Zoom", 1.0, 3.0, 1.0, 0.5, key=f"caa_zoom_{case_key}")
            if zoom > 1.0:
                h_z = int(512 / zoom)
                w_z = int(512 / zoom)
                focus_x = c_z2.slider("🧭 Centrum X", w_z//2, 512 - w_z//2, 256, key=f"caa_focx_{case_key}")
                focus_y = c_z3.slider("🧭 Centrum Y", h_z//2, 512 - h_z//2, 256, key=f"caa_focy_{case_key}")
            else:
                focus_x, focus_y = 256, 256
                
            stroke_width = st.slider("🖌 Grubość pędzla", 2, 30, 8, key=f"caa_stroke_{case_key}")
            draw_mode = st.radio("Narzędzie rysowania:", ["freedraw", "polygon"], horizontal=True, key=f"caa_mode_{case_key}")
            
            curr_bg = cv2.cvtColor(norm_512, cv2.COLOR_GRAY2RGB)
            if active_mask is not None and np.sum(active_mask) > 0:
                cnts_c, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(curr_bg, cnts_c, -1, (0, 255, 0), 2)
                
            if zoom > 1.0:
                hz, wz = int(512 / zoom), int(512 / zoom)
                top = int(np.clip(focus_y - hz // 2, 0, 512 - hz))
                left = int(np.clip(focus_x - wz // 2, 0, 512 - wz))
                viewport_bg = cv2.resize(curr_bg[top:top+hz, left:left+wz], (512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                viewport_bg = curr_bg
                top, left, wz, hz = 0, 0, 512, 512
                
            nudge_canvas_key = f"nudge_canvas_{case_key}_{st.session_state.get('caa_nudge_suffix', 0)}"
            nudge_canvas = st_canvas(
                fill_color="rgba(255, 140, 0, 0.4)",
                stroke_width=stroke_width,
                stroke_color="rgba(255, 140, 0, 0.9)",
                background_color="black",
                background_image=Image.fromarray(viewport_bg),
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode=draw_mode,
                key=nudge_canvas_key
            )
            
            c_nud1, c_nud2 = st.columns(2)
            with c_nud1:
                if st.button("✅ Zastosuj dorysowany obrys do maski", type="primary", key=f"btn_apply_nudge_{case_key}", use_container_width=True):
                    if nudge_canvas.json_data is not None and "objects" in nudge_canvas.json_data:
                        objs = nudge_canvas.json_data["objects"]
                        if len(objs) > 0:
                            base_mask = active_mask.copy() if active_mask is not None else np.zeros((512, 512), dtype=np.uint8)
                            
                            for obj in objs:
                                if obj.get("type") == "path":
                                    path_cmds = obj.get("path", [])
                                    stroke_pts = []
                                    for cmd in path_cmds:
                                        if len(cmd) >= 3:
                                            stroke_pts.append([cmd[-2], cmd[-1]])
                                    if len(stroke_pts) > 1:
                                        stroke_pts = np.array(stroke_pts, dtype=np.float32)
                                        scaled_pts = np.zeros_like(stroke_pts)
                                        scaled_pts[:, 0] = left + stroke_pts[:, 0] * (wz / 512.0)
                                        scaled_pts[:, 1] = top + stroke_pts[:, 1] * (hz / 512.0)
                                        scaled_pts = scaled_pts.astype(np.int32)
                                        cv2.polylines(base_mask, [scaled_pts], isClosed=False, color=255, thickness=int(stroke_width * (wz / 512.0)))
                                elif obj.get("type") == "polygon":
                                    poly_pts = obj.get("points", [])
                                    if len(poly_pts) > 2:
                                        p_arr = np.array([[p["x"], p["y"]] for p in poly_pts], dtype=np.float32)
                                        scaled_pts = np.zeros_like(p_arr)
                                        scaled_pts[:, 0] = left + p_arr[:, 0] * (wz / 512.0)
                                        scaled_pts[:, 1] = top + p_arr[:, 1] * (hz / 512.0)
                                        cv2.fillPoly(base_mask, [scaled_pts.astype(np.int32)], 255)
                                        
                            prof = extract_aneurysm_profile(base_mask, mm_per_pixel)
                            if prof is not None:
                                st.session_state[mask_key] = base_mask
                                st.session_state[prof_key] = prof
                                st.session_state[lm_key] = {
                                    "prox": prof["prox_idx"],
                                    "dist": prof["dist_idx"],
                                    "max": prof["max_idx"]
                                }
                                st.session_state["caa_nudge_suffix"] = st.session_state.get("caa_nudge_suffix", 0) + 1
                                st.success("✅ Zaktualizowano obrys i profil tętniaka!")
                                st.rerun()
                            else:
                                st.warning("Zaktualizowano maskę, lecz oś naczynia nie mogła zostać zaktualizowana.")
                        else:
                            st.info("Narysuj najpierw kształt na powiększeniu powyżej.")
            with c_nud2:
                if st.button("↩ Wyczyść szkic", key=f"btn_clr_nudge_{case_key}", use_container_width=True):
                    st.session_state["caa_nudge_suffix"] = st.session_state.get("caa_nudge_suffix", 0) + 1
                    st.rerun()

    # ── 5. RIGHT COLUMN: CLINICAL CLASSIFICATION & QUANTITATIVE MEASUREMENTS
    with col_params:
        st.markdown("#### Aneurysm Morphology & Measurements")
        
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            vessel = st.selectbox("Coronary Vessel:", ["LAD", "LCx", "RCA", "LM (Left Main)"], key="caa_vessel")
            morphology = st.selectbox("Aneurysm Morphology:", ["Workowaty (Saccular)", "Wrzecionowaty (Fusiform)", "Ektazja naczynia (Coronary Ectasia)"], key="caa_morph")
        with c_v2:
            aha_segment = st.selectbox("AHA Segment:", [f"Segment {i}" for i in range(1, 17)], index=5 if vessel=="LAD" else (0 if vessel=="RCA" else 10), key="caa_aha")
            thrombus = st.selectbox("Thrombus Presence:", ["Brak (None)", "Obecna (Present)", "Podejrzenie (Suspected)"], key="caa_thrombus")
            
        calcification = st.selectbox("Wall Calcification:", ["Brak (None)", "Łagodne (Mild)", "Masywne (Severe)"], key="caa_calc")
        
        st.markdown("---")
        st.markdown("##### 📏 Quantitative Calibration & Dimensions")
        
        # Calculate live defaults from profile if available
        if active_profile is not None:
            p_idx = active_landmarks["prox"] if active_landmarks else active_profile["prox_idx"]
            d_idx = active_landmarks["dist"] if active_landmarks else active_profile["dist_idx"]
            m_idx = active_landmarks["max"] if active_landmarks else active_profile["max_idx"]
            
            p_idx = int(np.clip(p_idx, 0, len(active_profile["thickness_mm"]) - 1))
            d_idx = int(np.clip(d_idx, 0, len(active_profile["thickness_mm"]) - 1))
            m_idx = int(np.clip(m_idx, 0, len(active_profile["thickness_mm"]) - 1))
            
            calc_ref_prox = round(float(active_profile["thickness_mm"][p_idx]), 2)
            calc_ref_dist = round(float(active_profile["thickness_mm"][d_idx]), 2)
            calc_max_diam = round(float(active_profile["thickness_mm"][m_idx]), 2)
            calc_len = round(float(abs(active_profile["cum_dist_mm"][d_idx] - active_profile["cum_dist_mm"][p_idx])), 2)
            if calc_len < 1.0: calc_len = 5.0
        else:
            calc_ref_prox = 3.0
            calc_ref_dist = 2.6
            calc_max_diam = 6.2
            calc_len = 12.5
            
        col_dim1, col_dim2 = st.columns(2)
        with col_dim1:
            ref_prox = st.number_input("Proximal Reference [mm]:", min_value=0.5, max_value=15.0, value=float(calc_ref_prox), step=0.1, key=f"caa_ref_prox_{case_key}")
            ref_dist = st.number_input("Distal Reference [mm]:", min_value=0.5, max_value=15.0, value=float(calc_ref_dist), step=0.1, key=f"caa_ref_dist_{case_key}")
            aneurysm_len = st.number_input("Aneurysm Length [mm]:", min_value=1.0, max_value=100.0, value=float(calc_len), step=0.5, key=f"caa_len_{case_key}")
            
        with col_dim2:
            max_diam = st.number_input("Max Aneurysm Diameter [mm]:", min_value=0.5, max_value=40.0, value=float(calc_max_diam), step=0.1, key=f"caa_max_diam_{case_key}")
            ref_mode = st.radio("Reference Baseline:", ["Interpolated (Midpoint)", "Proximal Only", "Distal Only"], horizontal=True, key=f"caa_ref_mode_{case_key}")
            
        # Compute Reference Baseline
        if ref_mode == "Interpolated (Midpoint)":
            interp_ref = round((ref_prox + ref_dist) / 2.0, 2)
        elif ref_mode == "Proximal Only":
            interp_ref = ref_prox
        else:
            interp_ref = ref_dist
            
        # Compute Eccentricity / Expansion Ratio
        expansion_ratio = round(max_diam / interp_ref, 2) if interp_ref > 0 else 1.0
        pct_dilation = round((expansion_ratio - 1.0) * 100.0, 1)
        
        # Clinical classification
        if expansion_ratio < 1.2:
            cls_text = "⚪ Normal / Mild Normal Variation (< 1.2x)"
            cls_badge = "color: #94a3b8;"
        elif expansion_ratio < 1.5:
            cls_text = "🟡 Borderline / Coronary Ectasia (1.2 - 1.5x)"
            cls_badge = "color: #facc15;"
        elif max_diam >= 8.0 or expansion_ratio >= 4.0:
            cls_text = "🔴 GIANT Coronary Aneurysm (≥ 8 mm or ≥ 4.0x)"
            cls_badge = "color: #ef4444; font-weight: bold;"
        else:
            cls_text = "🟠 Coronary Artery Aneurysm (> 1.5x)"
            cls_badge = "color: #fb923c; font-weight: bold;"
            
        st.markdown(f"""
        <div style='background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px;'>
            <div style='font-size: 13px; color: #94a3b8;'>Wskaźnik rozszerzenia / Ekscentryczność:</div>
            <div style='font-size: 20px; font-weight: 700; color: #38bdf8;'>{expansion_ratio}x <span style='font-size: 14px; font-weight: normal; color: #cbd5e1;'>({pct_dilation:+.1f}% względem ref {interp_ref} mm)</span></div>
            <div style='font-size: 13px; margin-top: 4px; {cls_badge}'>{cls_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Optional Concomitant Stenosis
        st.markdown("---")
        has_stenosis = st.checkbox("Współistniejące zwężenie w obrębie tętniaka (Concomitant Stenosis)", key=f"caa_has_stenosis_{case_key}")
        mld_val = None
        pct_stenosis = 0.0
        if has_stenosis:
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                mld_val = st.number_input("Minimal Lumen Diameter (MLD) [mm]:", min_value=0.2, max_value=float(interp_ref), value=min(1.5, float(interp_ref)), step=0.1, key=f"caa_mld_input_{case_key}")
            with c_s2:
                pct_stenosis = round((1.0 - (mld_val / interp_ref)) * 100.0, 1)
                st.metric("% Stenosis:", f"{pct_stenosis}%", delta=f"-{round(interp_ref - mld_val, 2)} mm")

    # ── 6. BIPLANE DUAL-PROJECTION 3D SIMPSON VOLUME MODULE ──────────────
    st.markdown("---")
    st.markdown("### 🌐 Biplane Dual-Projection Volumetry (Reguła Simpsona 3D)")
    
    with st.expander("Oblicz objętość 3D tętniaka z 2 różnych projekcji pod kątem ≥ 30°", expanded=True):
        st.markdown(r"""
        Wybierz drugą projekcję angiograficzną tego samego segmentu naczynia. 
        System sprawdzi kąt trójwymiarowy $\Delta \theta$ z nagłówków DICOM i obliczy objętość tętniaka metodą eliptycznych dysków Simpsona.
        """)
        
        other_series_opts = [lbl for lbl in series_map.keys() if lbl != selected_series_lbl]
        if not other_series_opts:
            st.info("Dla tego pacjenta dostępna jest tylko 1 sekwencja DICOM. Do wyliczenia objętości Simpsona 3D wymagane są minimum 2 projekcje.")
        else:
            c_bp1, c_bp2 = st.columns([1.5, 1])
            with c_bp1:
                proj2_lbl = st.selectbox("Wybierz drugą projekcję (Projection 2):", options=other_series_opts, key="caa_proj2_select")
                _, d_meta2 = series_map[proj2_lbl]
                
            angle_diff = compute_3d_angle_diff(
                d_meta["primary_angle"], d_meta["secondary_angle"],
                d_meta2["primary_angle"], d_meta2["secondary_angle"]
            )
            
            with c_bp2:
                st.markdown(rf"**Różnica kątów w przestrzeni ($\Delta \theta$):**")
                if angle_diff >= 30.0:
                    st.success(rf"✅ **{angle_diff:.1f}°** (Spełnia warunek $\ge 30^\circ$)")
                else:
                    st.warning(rf"⚠️ **{angle_diff:.1f}°** (< 30° – kąt może być zbyt mały)")
                    
            c_bp_dim1, c_bp_dim2 = st.columns(2)
            with c_bp_dim1:
                d2_max_diam = st.number_input("Max Diameter w projekcji 2 [mm]:", min_value=0.5, max_value=40.0, value=float(max_diam), step=0.1, key="caa_d2_max")
                d2_ref = st.number_input("Referencja w projekcji 2 [mm]:", min_value=0.5, max_value=15.0, value=float(interp_ref), step=0.1, key="caa_d2_ref")
                
            with c_bp_dim2:
                total_vol, excess_vol = simpsons_volume_biplane(
                    d1_max=max_diam, d2_max=d2_max_diam,
                    ref1=interp_ref, ref2=d2_ref,
                    length_mm=aneurysm_len
                )
                
                st.markdown(f"""
                <div style='background-color: #042f2e; padding: 14px; border-radius: 8px; border: 1px solid #0f766e;'>
                    <div style='color: #2dd4bf; font-size: 13px; font-weight: 600;'>📐 Wyniki objętości metodą Simpsona (Simpson's Discs):</div>
                    <div style='font-size: 22px; font-weight: 700; color: #5eead4; margin-top: 4px;'>
                        {total_vol:.1f} mm³ <span style='font-size: 14px; font-weight: normal; color: #ccfbf1;'>({total_vol:.1f} μl)</span>
                    </div>
                    <div style='font-size: 13px; color: #a7f3d0; margin-top: 4px;'>
                        Objętość nadmiarowa rozstrzeni: <b>{excess_vol:.1f} mm³</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.session_state["caa_calculated_volume"] = {
                "total_volume_mm3": round(total_vol, 2),
                "excess_volume_mm3": round(excess_vol, 2),
                "angle_diff_deg": round(angle_diff, 1),
                "proj2_name": d_meta2["series_desc"]
            }

    # ── 7. SAVE ANNOTATION TO FIRESTORE ───────────────────────────────────
    st.markdown("---")
    c_save1, c_save2 = st.columns([1, 2])
    with c_save1:
        if st.button("💾 Zapisz oznaczenie tętniaka", type="primary", use_container_width=True, key="btn_save_caa"):
            vol_data = st.session_state.get("caa_calculated_volume", {})
            
            # Generate thumbnail base64
            img_b64 = None
            try:
                ov = render_aneurysm_overlay(norm_512, active_mask, active_profile, active_landmarks)
                pil_thumb = Image.fromarray(ov)
                pil_thumb.thumbnail((450, 450))
                buf = io.BytesIO()
                pil_thumb.save(buf, format="JPEG", quality=80)
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Error encoding thumbnail: {e}")
                
            record = {
                "patient_id": active_pid,
                "dicom_name": os.path.basename(selected_dfp),
                "frame_idx": int(frame_slider),
                "vessel": vessel,
                "aha_segment": aha_segment,
                "morphology": morphology,
                "thrombus": thrombus,
                "calcification": calcification,
                "primary_angle": d_meta["primary_angle"],
                "secondary_angle": d_meta["secondary_angle"],
                "ref_prox_mm": float(ref_prox),
                "ref_dist_mm": float(ref_dist),
                "ref_interp_mm": float(interp_ref),
                "max_aneurysm_diam_mm": float(max_diam),
                "aneurysm_length_mm": float(aneurysm_len),
                "expansion_ratio": float(expansion_ratio),
                "pct_dilation": float(pct_dilation),
                "classification": cls_text,
                "has_concomitant_stenosis": bool(has_stenosis),
                "mld_mm": float(mld_val) if mld_val is not None else None,
                "pct_stenosis": float(pct_stenosis) if has_stenosis else 0.0,
                "simpson_total_vol_mm3": vol_data.get("total_volume_mm3"),
                "simpson_excess_vol_mm3": vol_data.get("excess_volume_mm3"),
                "simpson_angle_diff_deg": vol_data.get("angle_diff_deg"),
                "has_contour": bool(active_mask is not None),
                "thumbnail_b64": img_b64,
                "annotator": st.session_state.user.get("email", "syl.iwanczyk@gmail.com"),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                import firebase_admin
                from firebase_admin import firestore
                db = firestore.client()
                
                doc_id = f"{active_pid}_{vessel}_{aha_segment}_{int(time.time())}"
                db.collection("aneurysm_results").document(doc_id).set(record)
                
                st.success(f"✅ Zapisano pomyślnie oznaczenie tętniaka dla pacjenta **{active_pid}** ({vessel} {aha_segment}) w kolekcji 'aneurysm_results'!")
            except Exception as e:
                st.error(f"Błąd zapisu do bazy: {e}")

    # ── 8. TABLE & SAVED CONTOUR PREVIEW ─────────────────────────────────
    st.markdown("#### 📋 Zapisane tętniaki dla pacjenta:")
    try:
        from firebase_admin import firestore
        db = firestore.client()
        docs = list(db.collection("aneurysm_results").where("patient_id", "==", active_pid).stream())
        if docs:
            rows = []
            doc_dict = {}
            for d in docs:
                dt = d.to_dict()
                doc_dict[d.id] = dt
                vol_str = f"{dt.get('simpson_total_vol_mm3')} mm³" if dt.get('simpson_total_vol_mm3') else "—"
                rows.append({
                    "ID": d.id,
                    "Naczynie": dt.get("vessel"),
                    "Segment": dt.get("aha_segment"),
                    "Morfologia": dt.get("morphology"),
                    "Max Diam [mm]": dt.get("max_aneurysm_diam_mm"),
                    "Ref [mm]": dt.get("ref_interp_mm"),
                    "Ratio (Ekscentryczność)": f"{dt.get('expansion_ratio')}x",
                    "Objętość 3D Simpsona": vol_str,
                    "Data": dt.get("created_at")
                })
            st.dataframe(pd.DataFrame(rows).drop(columns=["ID"]), use_container_width=True)
            
            with st.expander("🔍 Podgląd zapisanego obrysu tętniaka z bazy", expanded=False):
                selected_doc_id = st.selectbox("Wybierz zapisane oznaczenie:", options=list(doc_dict.keys()), key="caa_saved_preview_select")
                if selected_doc_id:
                    saved_dt = doc_dict[selected_doc_id]
                    st.markdown(f"**Pacjent:** `{saved_dt.get('patient_id')}` | **Naczynie:** `{saved_dt.get('vessel')}` `{saved_dt.get('aha_segment')}` | **Data:** `{saved_dt.get('created_at')}`")
                    if saved_dt.get("thumbnail_b64"):
                        img_bytes = base64.b64decode(saved_dt["thumbnail_b64"])
                        st.image(img_bytes, caption=f"Zapisany obrys tętniaka: {saved_dt.get('vessel')} {saved_dt.get('aha_segment')} (Dmax={saved_dt.get('max_aneurysm_diam_mm')} mm, Ratio={saved_dt.get('expansion_ratio')}x)", use_column_width=True)
                    else:
                        st.info("Dla tego rekordu brak zapisanego zrzutu obrysu.")
        else:
            st.info("Brak zapisanych oznaczeń dla tego pacjenta. Wypełnij parametry powyżej i kliknij 'Zapisz'.")
    except Exception as e:
        print(f"Error fetching saved aneurysms: {e}")
