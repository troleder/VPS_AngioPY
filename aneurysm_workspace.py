import os
import io
import time
import math
import re
import json
import zipfile
import shutil
import base64
import hashlib
import struct
import numpy as np
import pydicom
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import cv2
import scipy.ndimage
import scipy.interpolate
import scipy.signal
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import angioPyFunctions
import predict

# ── AHA Vessel Segment Hierarchy ──────────────────────────────────────────────
AHA_VESSEL_SEGMENTS = {
    "LM & LAD – Left Main & Left Anterior Descending": {
        "key": "LAD",
        "segments": [
            ("5", "Seg 5 – Left Main (LMCA)"),
            ("6", "Seg 6 – Proximal LAD"),
            ("7", "Seg 7 – Mid LAD"),
            ("8", "Seg 8 – Distal LAD"),
            ("9", "Seg 9 – D1 (First Diagonal)"),
            ("10", "Seg 10 – D2 (Second Diagonal)"),
            ("16", "Seg 16 – IM (Intermedius)"),
        ]
    },
    "LCx – Left Circumflex": {
        "key": "CX",
        "segments": [
            ("11", "Seg 11 – Proximal LCx"),
            ("12", "Seg 12 – OM1 (First Obtuse Marginal)"),
            ("12a", "Seg 12a – OM2 (Second Obtuse Marginal)"),
            ("13", "Seg 13 – Distal LCx"),
            ("14L", "Seg 14L – PL (Posterolateral Branch)"),
            ("15", "Seg 15 – PDA (Left Dominant)"),
        ]
    },
    "RCA – Right Coronary Artery": {
        "key": "RCA",
        "segments": [
            ("1", "Seg 1 – Proximal RCA"),
            ("2", "Seg 2 – Mid RCA"),
            ("3", "Seg 3 – Distal RCA"),
            ("4", "Seg 4 – PDA (Posterior Descending)"),
            ("14R", "Seg 14R – PLV (Posterior Left Ventricular)"),
        ]
    },
}
ALL_SYSTEM_NAMES = list(AHA_VESSEL_SEGMENTS.keys())

def _seg_labels(system_name):
    if system_name in AHA_VESSEL_SEGMENTS:
        return [s[1] for s in AHA_VESSEL_SEGMENTS[system_name]["segments"]]
    return []

def _seg_codes(system_name):
    if system_name in AHA_VESSEL_SEGMENTS:
        return [s[0] for s in AHA_VESSEL_SEGMENTS[system_name]["segments"]]
    return []

def safe_display_image(img, caption=None):
    try:
        st.image(img, caption=caption, use_column_width=True)
    except TypeError:
        try:
            st.image(img, caption=caption, use_container_width=True)
        except TypeError:
            st.image(img, caption=caption)

# Storage base for Sylwia's uploaded patient cases
ANEURYSM_STORAGE_DIR = "/var/www/analiza-dicom/aneurysm_storage/sylwia"
if not os.path.exists(os.path.dirname(ANEURYSM_STORAGE_DIR)):
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
        if not hasattr(dcm, "pixel_array"):
            return None
            
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
        return None

def compute_3d_angle_diff(alpha1, beta1, alpha2, beta2):
    """
    Computes true 3D spatial angle between two projection vectors in degrees.
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

def compute_biplane_simpsons_volumetry(thick1, cum_dist1, prox1, dist1, max1, thick2, cum_dist2, prox2, dist2, max2, n_slices=30):
    """
    Computes true biplane 3D lumen volume directly from two segmented projection profiles:
    - Normalizes centerline path length between proximal and distal landmarks
    - Evaluates cross-sectional orthogonal diameters D1(s) and D2(s) along the length
    - Evaluates reference vessel diameters Dref1(s) and Dref2(s)
    - Integrates elliptical slice areas: A(s) = (pi / 4) * D1(s) * D2(s)
    - Computes total volume and excess aneurysm pouch volume
    - Computes cross-sectional eccentricity and morphological ratios
    """
    try:
        # Segment 1
        i_s1, i_e1 = min(prox1, dist1), max(prox1, dist1)
        i_s1 = max(0, min(len(thick1) - 1, i_s1))
        i_e1 = max(0, min(len(thick1) - 1, i_e1))
        if i_s1 == i_e1:
            i_s1, i_e1 = 0, len(thick1) - 1
            
        L1 = float(abs(cum_dist1[i_e1] - cum_dist1[i_s1]))
        if L1 < 0.5: L1 = 5.0
        
        seg_thick1 = thick1[i_s1:i_e1+1]
        u1 = np.linspace(0.0, 1.0, len(seg_thick1))
        u_eval = np.linspace(0.0, 1.0, n_slices + 1)
        D1 = np.interp(u_eval, u1, seg_thick1)
        ref1_prox, ref1_dist = float(D1[0]), float(D1[-1])
        ref1_mean = (ref1_prox + ref1_dist) / 2.0
        Dref1 = ref1_prox + u_eval * (ref1_dist - ref1_prox)

        # Segment 2
        i_s2, i_e2 = min(prox2, dist2), max(prox2, dist2)
        i_s2 = max(0, min(len(thick2) - 1, i_s2))
        i_e2 = max(0, min(len(thick2) - 1, i_e2))
        if i_s2 == i_e2:
            i_s2, i_e2 = 0, len(thick2) - 1
            
        L2 = float(abs(cum_dist2[i_e2] - cum_dist2[i_s2]))
        if L2 < 0.5: L2 = 5.0
        
        seg_thick2 = thick2[i_s2:i_e2+1]
        u2 = np.linspace(0.0, 1.0, len(seg_thick2))
        D2 = np.interp(u_eval, u2, seg_thick2)
        ref2_prox, ref2_dist = float(D2[0]), float(D2[-1])
        ref2_mean = (ref2_prox + ref2_dist) / 2.0
        Dref2 = ref2_prox + u_eval * (ref2_dist - ref2_prox)

        L = (L1 + L2) / 2.0
        ds = L / n_slices

        areas = (np.pi / 4.0) * D1 * D2
        ref_areas = (np.pi / 4.0) * Dref1 * Dref2

        # Kliniczna reguła odcięcia: do 1.2x to fizjologiczna norma (zdrowe naczynie).
        # Właściwy tętniak / rozstrzeń (szyja i worek) to strefa >= 1.2x.
        # Tylko dla plastrów >= 1.2x liczymy nadmiarową objętość tętniaka:
        mean_D_slice = (D1 + D2) / 2.0
        mean_ref_slice = np.maximum(0.1, (Dref1 + Dref2) / 2.0)
        dilation_ratio_slice = mean_D_slice / mean_ref_slice
        excess_areas = np.where(dilation_ratio_slice >= 1.2, np.maximum(0.0, areas - ref_areas), 0.0)

        weights = np.ones(len(areas))
        weights[1:-1:2] = 4.0
        weights[2:-2:2] = 2.0

        total_vol = (ds / 3.0) * float(np.sum(weights * areas))
        ref_vol = (ds / 3.0) * float(np.sum(weights * ref_areas))
        excess_vol = (ds / 3.0) * float(np.sum(weights * excess_areas))

        dilated_mask = (dilation_ratio_slice >= 1.2)
        if np.any(dilated_mask):
            aneurysm_indices = np.where(dilated_mask)[0]
            aneurysm_len = float((aneurysm_indices[-1] - aneurysm_indices[0]) / n_slices * L)
            aneurysm_len = max(0.5, round(aneurysm_len, 1))
        else:
            aneurysm_len = 0.0

        d1_max = float(np.max(D1))
        d2_max = float(np.max(D2))
        a = max(d1_max, d2_max)
        b = min(d1_max, d2_max)
        ellipticity = float(a / b) if b > 0 else 1.0
        eccentricity = float(np.sqrt(max(0.0, 1.0 - (b / a)**2))) if a > 0 else 0.0

        dil_ratio_1 = round(d1_max / ref1_mean, 2) if ref1_mean > 0 else 1.0
        dil_ratio_2 = round(d2_max / ref2_mean, 2) if ref2_mean > 0 else 1.0

        effective_len = aneurysm_len if aneurysm_len > 0 else L
        morphology = "Wrzecionowaty (Fusiform)" if (effective_len / a >= 2.0) else "Workowaty (Saccular)"

        s_slices = (u_eval * L).tolist()
        max_idx_slice = int(np.argmax((D1 + D2) / 2.0))

        return {
            "total_vol": round(total_vol, 2),
            "excess_vol": round(excess_vol, 2),
            "ref_vol": round(ref_vol, 2),
            "d1_max": round(d1_max, 2),
            "d2_max": round(d2_max, 2),
            "ref1_mean": round(ref1_mean, 2),
            "ref2_mean": round(ref2_mean, 2),
            "L1": round(L1, 2),
            "L2": round(L2, 2),
            "L": round(L, 2),
            "aneurysm_len": aneurysm_len,
            "ellipticity": round(ellipticity, 2),
            "eccentricity": round(eccentricity, 2),
            "dilation_ratio_1": dil_ratio_1,
            "dilation_ratio_2": dil_ratio_2,
            "morphology": morphology,
            "D1_slices": [round(float(v), 3) for v in D1],
            "D2_slices": [round(float(v), 3) for v in D2],
            "Dref1_slices": [round(float(v), 3) for v in Dref1],
            "Dref2_slices": [round(float(v), 3) for v in Dref2],
            "s_slices": [round(float(v), 3) for v in s_slices],
            "max_idx_slice": max_idx_slice,
            "n_slices": n_slices
        }
    except Exception as e:
        print(f"Error in compute_biplane_simpsons_volumetry: {e}")
        return None

def generate_aneurysm_3d_figure(simp, pair, color_mode="diameter", orientation="vertical", show_ref=True, show_rings=True, n_phi=36):
    """
    Constructs an interactive 3D mesh reconstruction of the coronary artery aneurysm
    derived from biplane Simpson cross-sectional slice profiles.
    Supports vertical (pionowa - default, matching anatomical vessel catheterization flow)
    and horizontal (pozioma) orientations.
    """
    try:
        D1 = np.array(simp["D1_slices"])
        D2 = np.array(simp["D2_slices"])
        Dref1 = np.array(simp["Dref1_slices"])
        Dref2 = np.array(simp["Dref2_slices"])
        s = np.array(simp["s_slices"])
    except (KeyError, TypeError):
        return None

    n_s = len(s)
    if n_s < 2:
        return None

    L_tot = float(s[-1]) if len(s) > 0 else 10.0
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)

    # Semi-axes along the length
    R1 = (D1 / 2.0)[:, None]
    R2 = (D2 / 2.0)[:, None]
    mean_D = (D1 + D2) / 2.0
    mean_Dref = (Dref1 + Dref2) / 2.0

    is_vertical = (orientation == "vertical")

    if is_vertical:
        # Longitudinal axis mapped to vertical Z:
        # Proximal (s=0) at TOP (+Z), Distal (s=L_tot) at BOTTOM (-Z)
        z_long = (L_tot / 2.0) - s
        Z, Phi = np.meshgrid(z_long, phi, indexing="ij")
        X = R1 * np.cos(Phi)
        Y = R2 * np.sin(Phi)
    else:
        # Horizontal layout along X axis:
        # Proximal at left (-X), Distal at right (+X)
        x_long = s - (L_tot / 2.0)
        X, Phi = np.meshgrid(x_long, phi, indexing="ij")
        Y = R1 * np.cos(Phi)
        Z = R2 * np.sin(Phi)

    # Determine surface color values
    dil_ratio = mean_D / np.maximum(0.1, mean_Dref)

    if color_mode == "zones":
        # Wyraźne, dyskretne strefy kliniczne (bez gradientu!):
        # 0: Zdrowe naczynie (< 1.2x Ref) -> Zielony
        # 1: Szyja tętniaka / Ektazja (1.2x - 1.4x Ref) -> Bursztynowy/Żółty
        # 2: Właściwy worek tętniaka (>= 1.4x Ref) -> Czerwony
        C_1d = np.zeros(n_s, dtype=float)
        for i in range(n_s):
            if dil_ratio[i] < 1.2:
                C_1d[i] = 0.0
            elif dil_ratio[i] < 1.4:
                C_1d[i] = 1.0
            else:
                C_1d[i] = 2.0

        color_title = "<b>Strefy kliniczne</b>"
        cmin, cmax = 0.0, 2.0
        colorscale = [
            [0.0, "#10b981"],   # Zielony: Zdrowe naczynie (< 1.2x)
            [0.25, "#10b981"],
            [0.25, "#f59e0b"],  # Bursztynowy/Żółty: Szyja tętniaka (1.2x - 1.4x)
            [0.75, "#f59e0b"],
            [0.75, "#ef4444"],  # Czerwony: Worek tętniaka (>= 1.4x)
            [1.0, "#ef4444"]
        ]
        colorbar_cfg = dict(
            title=dict(text=color_title, font=dict(color="#e2e8f0", size=12)),
            tickmode="array",
            tickvals=[0.0, 1.0, 2.0],
            ticktext=["🟢 Zdrowe (<1.2x)", "🟡 Szyja (1.2–1.4x)", "🔴 Worek (≥1.4x)"],
            tickfont=dict(color="#e2e8f0", size=10),
            len=0.70,
            x=1.02,
            thickness=16
        )
    elif color_mode == "dilation":
        C_1d = dil_ratio
        color_title = "Rozstrzeń (x Ref)"
        cmin, cmax = 1.0, max(2.5, float(np.max(C_1d)))
        colorscale = "Turbo"
        colorbar_cfg = dict(
            title=dict(text=color_title, font=dict(color="#e2e8f0", size=12)),
            tickfont=dict(color="#e2e8f0", size=10),
            len=0.75,
            x=1.02,
            thickness=14
        )
    else:  # diameter
        C_1d = mean_D
        color_title = "Średnica [mm]"
        cmin, cmax = float(np.min(C_1d)), float(np.max(C_1d))
        colorscale = "Turbo"
        colorbar_cfg = dict(
            title=dict(text=color_title, font=dict(color="#e2e8f0", size=12)),
            tickfont=dict(color="#e2e8f0", size=10),
            len=0.75,
            x=1.02,
            thickness=14
        )

    C = np.tile(C_1d[:, None], (1, n_phi))

    # Custom hover text for vertices
    hover_text = []
    for i in range(n_s):
        row_txt = []
        seg_pos_name = "Proksymalny (Wlot)" if i == 0 else ("Dystalny (Wylot)" if i == n_s - 1 else f"{s[i]:.1f} mm od wlotu")
        r_val = dil_ratio[i]
        if r_val < 1.2:
            zone_badge = "🟢 Zdrowe naczynie (< 1.2x Ref)"
        elif r_val < 1.4:
            zone_badge = "🟡 Szyja tętniaka / Rozstrzeń (1.2x – 1.4x Ref)"
        else:
            zone_badge = "🔴 Worek tętniaka (≥ 1.4x Ref)"

        for j in range(n_phi):
            txt = (
                f"<b>Strefa:</b> {zone_badge}<br>"
                f"<b>Wskaźnik rozstrzeni:</b> {r_val:.2f}x Ref<br>"
                f"<b>Pozycja:</b> {seg_pos_name}<br>"
                f"<b>Odległość od wlotu:</b> {s[i]:.1f} mm<br>"
                f"<b>Średnica Proj 1 (D1):</b> {D1[i]:.2f} mm<br>"
                f"<b>Średnica Proj 2 (D2):</b> {D2[i]:.2f} mm<br>"
                f"<b>Średnia średnica:</b> {mean_D[i]:.2f} mm<br>"
                f"<b>Pole przekroju:</b> {(np.pi/4 * D1[i] * D2[i]):.1f} mm²"
            )
            row_txt.append(txt)
        hover_text.append(row_txt)

    fig = go.Figure()

    # 1. Main Aneurysm Lumen Surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=C,
        colorscale=colorscale,
        cmin=cmin, cmax=cmax,
        colorbar=colorbar_cfg,
        lighting=dict(
            ambient=0.68,
            diffuse=0.82,
            specular=0.55,
            roughness=0.35,
            fresnel=0.25
        ),
        lightposition=dict(x=100, y=200, z=150),
        hoverinfo="text",
        text=hover_text,
        opacity=0.94,
        name="Światło tętniaka"
    ))

    # 2. Optional: Ghost healthy reference lumen
    if show_ref:
        Rref1 = (Dref1 / 2.0)[:, None]
        Rref2 = (Dref2 / 2.0)[:, None]
        if is_vertical:
            X_ref = Rref1 * np.cos(Phi)
            Y_ref = Rref2 * np.sin(Phi)
            Z_ref = Z
        else:
            X_ref = X
            Y_ref = Rref1 * np.cos(Phi)
            Z_ref = Rref2 * np.sin(Phi)

        fig.add_trace(go.Surface(
            x=X_ref, y=Y_ref, z=Z_ref,
            surfacecolor=np.ones_like(C),
            colorscale=[[0, "rgba(56, 189, 248, 0.20)"], [1, "rgba(56, 189, 248, 0.20)"]],
            showscale=False,
            lighting=dict(ambient=0.8, diffuse=0.3),
            hoverinfo="skip",
            opacity=0.32,
            name="Zdrowe naczynie (Ref)"
        ))

    # 3. Centerline trace
    if is_vertical:
        cx = np.zeros_like(s)
        cy = np.zeros_like(s)
        cz = z_long
    else:
        cx = x_long
        cy = np.zeros_like(s)
        cz = np.zeros_like(s)

    fig.add_trace(go.Scatter3d(
        x=cx,
        y=cy,
        z=cz,
        mode="lines",
        line=dict(color="#38bdf8", width=3, dash="dash"),
        hoverinfo="skip",
        name="Oś centralna"
    ))

    # 4. Optional: Landmark caliper rings (Prox, Dist, Dmax)
    if show_rings:
        def add_caliper_ring(idx, ring_color, label):
            theta_r = np.linspace(0.0, 2.0 * np.pi, 60)
            if is_vertical:
                rx = (D1[idx] / 2.0) * np.cos(theta_r)
                ry = (D2[idx] / 2.0) * np.sin(theta_r)
                rz = np.full_like(theta_r, z_long[idx])
            else:
                rx = np.full_like(theta_r, x_long[idx])
                ry = (D1[idx] / 2.0) * np.cos(theta_r)
                rz = (D2[idx] / 2.0) * np.sin(theta_r)

            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="lines",
                line=dict(color=ring_color, width=5),
                hoverinfo="text",
                text=f"{label}: {mean_D[idx]:.1f} mm (od wlotu {s[idx]:.1f} mm)",
                name=label
            ))

        add_caliper_ring(0, "#22c55e", f"Ref Prox (Góra) ({mean_D[0]:.1f} mm)")
        add_caliper_ring(n_s - 1, "#22c55e", f"Ref Dist (Dół) ({mean_D[-1]:.1f} mm)")
        m_idx = simp.get("max_idx_slice", int(np.argmax(mean_D)))
        add_caliper_ring(m_idx, "#ef4444", f"Dmax ({mean_D[m_idx]:.1f} mm)")

    max_r = max(float(np.max(D1 / 2.0)), float(np.max(D2 / 2.0)), 3.0)

    if is_vertical:
        z_aspect = max(2.0, min(4.2, float(L_tot / (2.0 * max_r)) * 1.5))
        scene_aspect = dict(x=1.0, y=1.0, z=z_aspect)
        scene_camera = dict(
            eye=dict(x=0.25, y=2.2, z=0.0),
            up=dict(x=0, y=0, z=1)
        )
        xaxis_cfg = dict(
            title=dict(text="X: Proj 1 (mm)", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#64748b", size=9),
            backgroundcolor="#0f172a",
            gridcolor="#334155",
            showbackground=True,
            range=[-max_r * 1.5, max_r * 1.5]
        )
        yaxis_cfg = dict(
            title=dict(text="Y: Proj 2 (mm)", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#64748b", size=9),
            backgroundcolor="#0f172a",
            gridcolor="#334155",
            showbackground=True,
            range=[-max_r * 1.5, max_r * 1.5]
        )
        zaxis_cfg = dict(
            title=dict(text="Z: Oś podłużna (Góra: Prox ↓ Dół: Dist) [mm]", font=dict(color="#38bdf8", size=11)),
            tickfont=dict(color="#64748b", size=9),
            backgroundcolor="#0b1120",
            gridcolor="#334155",
            showbackground=True,
            range=[-L_tot / 2.0 - 1.5, L_tot / 2.0 + 1.5]
        )
        camera_buttons = [
            dict(
                label="↕️ Front (Wertykalnie)",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=0.25, y=2.2, z=0.0), up=dict(x=0, y=0, z=1))}]
            ),
            dict(
                label="📐 Izometria",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=1.4, y=1.4, z=0.8), up=dict(x=0, y=0, z=1))}]
            ),
            dict(
                label="🔄 Profil Proj 1",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=2.2, y=0.2, z=0.0), up=dict(x=0, y=0, z=1))}]
            ),
            dict(
                label="👁️ Przekrój En-Face",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=0, y=1, z=0))}]
            ),
        ]
    else:
        x_aspect = max(2.0, min(4.2, float(L_tot / (2.0 * max_r)) * 1.5))
        scene_aspect = dict(x=x_aspect, y=1.0, z=1.0)
        scene_camera = dict(
            eye=dict(x=0.0, y=2.2, z=0.4),
            up=dict(x=0, y=0, z=1)
        )
        xaxis_cfg = dict(
            title=dict(text="X: Oś naczynia (L: Prox → P: Dist) [mm]", font=dict(color="#38bdf8", size=11)),
            tickfont=dict(color="#64748b", size=9),
            backgroundcolor="#0b1120",
            gridcolor="#334155",
            showbackground=True,
            range=[-L_tot / 2.0 - 1.5, L_tot / 2.0 + 1.5]
        )
        yaxis_cfg = dict(
            title=dict(text="Y: Proj 1 (mm)", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#64748b", size=9),
            backgroundcolor="#0f172a",
            gridcolor="#334155",
            showbackground=True,
            range=[-max_r * 1.5, max_r * 1.5]
        )
        zaxis_cfg = dict(
            title=dict(text="Z: Proj 2 (mm)", font=dict(color="#94a3b8", size=11)),
            tickfont=dict(color="#64748b", size=9),
            backgroundcolor="#0f172a",
            gridcolor="#334155",
            showbackground=True,
            range=[-max_r * 1.5, max_r * 1.5]
        )
        camera_buttons = [
            dict(
                label="↔️ Profil (Horyzontalnie)",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=0.0, y=2.2, z=0.4), up=dict(x=0, y=0, z=1))}]
            ),
            dict(
                label="📐 Izometria",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=1.4, y=1.4, z=0.8), up=dict(x=0, y=0, z=1))}]
            ),
            dict(
                label="👁️ Przekrój En-Face",
                method="relayout",
                args=[{"scene.camera": dict(eye=dict(x=-2.4, y=0.0, z=0.0), up=dict(x=0, y=0, z=1))}]
            ),
        ]

    fig.update_layout(
        title=dict(
            text=f"<b>Rekonstrukcja 3D tętniaka: {pair.get('aha_label', 'Segment')}</b> | V = {simp['total_vol']:.1f} mm³ | Ekscentryczność: {simp['eccentricity']}",
            font=dict(color="#38bdf8", size=14)
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        margin=dict(l=10, r=10, t=50, b=10),
        height=650,
        showlegend=True,
        legend=dict(
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="#334155",
            borderwidth=1,
            x=0.02, y=0.98
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=1.09,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(15, 23, 42, 0.85)",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(color="#38bdf8", size=11),
                buttons=camera_buttons
            )
        ],
        scene=dict(
            xaxis=xaxis_cfg,
            yaxis=yaxis_cfg,
            zaxis=zaxis_cfg,
            aspectratio=scene_aspect,
            camera=scene_camera
        )
    )
    return fig

def export_aneurysm_to_stl(simp, n_phi=36, close_caps=True):
    """
    Generates a binary STL file bytes for the 3D reconstructed aneurysm lumen.
    Can be directly downloaded and opened in 3D Slicer, MeshMixer, or 3D printed.
    """
    try:
        D1 = np.array(simp["D1_slices"])
        D2 = np.array(simp["D2_slices"])
        s = np.array(simp["s_slices"])
    except (KeyError, TypeError):
        return b""

    n_s = len(s)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    verts = np.zeros((n_s, n_phi, 3), dtype=np.float32)
    for i in range(n_s):
        r1 = float(D1[i] / 2.0)
        r2 = float(D2[i] / 2.0)
        z = float(s[i])
        verts[i, :, 0] = r1 * np.cos(phi)
        verts[i, :, 1] = r2 * np.sin(phi)
        verts[i, :, 2] = z

    triangles = []
    for i in range(n_s - 1):
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            p00 = verts[i, j]
            p01 = verts[i, j_next]
            p10 = verts[i + 1, j]
            p11 = verts[i + 1, j_next]
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))

    if close_caps:
        c_prox = np.array([0.0, 0.0, float(s[0])], dtype=np.float32)
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            triangles.append((c_prox, verts[0, j_next], verts[0, j]))

        c_dist = np.array([0.0, 0.0, float(s[-1])], dtype=np.float32)
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            triangles.append((c_dist, verts[-1, j], verts[-1, j_next]))

    header = b"Coronary Artery Aneurysm 3D Reconstruction - AngioPY".ljust(80, b"\x00")[:80]
    n_tri = len(triangles)
    stl_buf = bytearray()
    stl_buf.extend(header)
    stl_buf.extend(struct.pack("<I", n_tri))

    for (v1, v2, v3) in triangles:
        edge1 = v2 - v1
        edge2 = v3 - v1
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len
        else:
            normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        stl_buf.extend(struct.pack("<3f", float(normal[0]), float(normal[1]), float(normal[2])))
        stl_buf.extend(struct.pack("<3f", float(v1[0]), float(v1[1]), float(v1[2])))
        stl_buf.extend(struct.pack("<3f", float(v2[0]), float(v2[1]), float(v2[2])))
        stl_buf.extend(struct.pack("<3f", float(v3[0]), float(v3[1]), float(v3[2])))
        stl_buf.extend(b"\x00\x00")

    return bytes(stl_buf)

def calibrate_catheter(frame_2d, pt1, pt2, catheter_mm):
    """
    Tracks edges of the catheter along the vector between pt1 and pt2
    using orthogonal gradient profiles with subpixel peak interpolation.
    """
    try:
        def get_subpixel_peak(idx, array):
            if idx == 0 or idx == len(array) - 1: return float(idx)
            alpha, beta, gamma = array[idx-1], array[idx], array[idx+1]
            denom = 2 * (alpha - 2*beta + gamma)
            return float(idx) if denom == 0 else idx + (alpha - gamma) / denom

        cx1, cy1 = int(round(pt1[0])), int(round(pt1[1]))
        cx2, cy2 = int(round(pt2[0])), int(round(pt2[1]))
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = np.hypot(dx, dy)
        tube_theta = np.arctan2(dy, dx)
        
        if dist < 6:
            return None
            
        mid_cx = (cx1 + cx2) / 2.0
        mid_cy = (cy1 + cy2) / 2.0
        cs_theta_init = tube_theta - np.pi / 2.0
        base_t = np.arange(-60, 60)
        
        tx = mid_cx + base_t * np.cos(cs_theta_init)
        ty = mid_cy + base_t * np.sin(cs_theta_init)
        coords = np.vstack((ty, tx))
        
        sf_f = frame_2d.astype(float)
        sf_min = sf_f.min()
        norm_frame = (sf_f - sf_min) / (sf_f.max() - sf_min + 1e-5) * 255.0
        
        profile = scipy.ndimage.map_coordinates(norm_frame, coords, mode='nearest')
        smoothed = np.convolve(profile, np.ones(3)/3.0, mode='same')
        grad = np.abs(np.gradient(smoothed))
        
        bestDiam = 15.0
        peaks = scipy.signal.find_peaks(grad, prominence=1.0, distance=4)[0]
        if len(peaks) >= 2:
            left_peaks = [p for p in peaks if p < 60]
            right_peaks = [p for p in peaks if p >= 60]
            if left_peaks and right_peaks:
                p1 = max(left_peaks)
                p2 = min(right_peaks)
                bestDiam = abs(get_subpixel_peak(p2, grad) - get_subpixel_peak(p1, grad))
                
        num_steps = max(3, int(dist / 5))
        detectedDiameters = []
        lw, rw = [], []
        caxis = tube_theta
        
        for step in range(num_steps + 1):
            frac = step / float(num_steps)
            ccx = cx1 + frac * dx
            ccy = cy1 + frac * dy
            cs_theta = caxis - np.pi / 2.0
            track_L = max(10, int(bestDiam * 1.3))
            track_t = np.arange(-track_L, track_L)
            tx = ccx + track_t * np.cos(cs_theta)
            ty = ccy + track_t * np.sin(cs_theta)
            t_coords = np.vstack((ty, tx))
            t_prof = scipy.ndimage.map_coordinates(norm_frame, t_coords, mode='nearest')
            t_smooth = np.convolve(t_prof, np.ones(3)/3.0, mode='same')
            t_grad = np.abs(np.gradient(t_smooth))
            
            t_peaks = scipy.signal.find_peaks(t_grad, prominence=1.0, distance=max(2, int(bestDiam * 0.3)))[0]
            if len(t_peaks) >= 2:
                left_peaks = [p for p in t_peaks if p < track_L]
                right_peaks = [p for p in t_peaks if p >= track_L]
                if left_peaks and right_peaks:
                    tp0 = max(left_peaks)
                    tp1 = min(right_peaks)
                    sp0 = get_subpixel_peak(tp0, t_grad)
                    sp1 = get_subpixel_peak(tp1, t_grad)
                    cur_diam = sp1 - sp0
                    if 4.0 <= cur_diam <= 80.0:
                        detectedDiameters.append(cur_diam)
                        woff_0 = -track_L + sp0
                        woff_1 = -track_L + sp1
                        wx0 = int(round(ccx + woff_0 * np.cos(cs_theta)))
                        wy0 = int(round(ccy + woff_0 * np.sin(cs_theta)))
                        wx1 = int(round(ccx + woff_1 * np.cos(cs_theta)))
                        wy1 = int(round(ccy + woff_1 * np.sin(cs_theta)))
                        lw.append((wx0, wy0))
                        rw.append((wx1, wy1))
                        
        if detectedDiameters and lw and rw:
            avg_diam_px = float(np.mean(detectedDiameters))
            mm_per_pixel = float(catheter_mm / avg_diam_px)
            mid_lw = lw[len(lw) // 2]
            mid_rw = rw[len(rw) // 2]
            mid_cx = (mid_lw[0] + mid_rw[0]) / 2.0
            mid_cy = (mid_lw[1] + mid_rw[1]) / 2.0
            cs_theta = tube_theta - np.pi / 2.0
            gx1 = mid_cx - (avg_diam_px / 2.0) * np.cos(cs_theta)
            gy1 = mid_cy - (avg_diam_px / 2.0) * np.sin(cs_theta)
            gx2 = mid_cx + (avg_diam_px / 2.0) * np.cos(cs_theta)
            gy2 = mid_cy + (avg_diam_px / 2.0) * np.sin(cs_theta)
            return {
                "avg_diam_px": avg_diam_px,
                "mm_per_pixel": mm_per_pixel,
                "lw": lw,
                "rw": rw,
                "click1": (cx1, cy1),
                "click2": (cx2, cy2),
                "ref_line": ((int(round(gx1)), int(round(gy1))), (int(round(gx2)), int(round(gy2))))
            }
    except Exception as e:
        print(f"Catheter calib error: {e}")
    return None

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
        
        clip = 8
        sp_x = sp_x[clip:-clip]
        sp_y = sp_y[clip:-clip]
        sp_dx = np.gradient(sp_x)
        sp_dy = np.gradient(sp_y)
        
        _diffs = np.hypot(np.diff(sp_x), np.diff(sp_y))
        cum_dist_px = np.concatenate([[0], np.cumsum(_diffs)])
        cum_dist_mm = cum_dist_px * mm_per_pixel
        
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
            
        max_idx = prox_idx + int(np.argmax(thickness_px[prox_idx:dist_idx+1]))
        
        ref_prox_mm = round(float(thickness_mm[prox_idx]), 2)
        ref_dist_mm = round(float(thickness_mm[dist_idx]), 2)
        max_diam_mm = round(float(thickness_mm[max_idx]), 2)
        interp_ref_mm = round((ref_prox_mm + ref_dist_mm) / 2.0, 2)
        
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

def render_aneurysm_overlay(base_img_512, mask_2d=None, profile=None, landmarks=None, default_dims=None, zoom=1.0, focus_x=256, focus_y=256, mld_idx=None):
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
        
        amber_layer = overlay.copy()
        amber_layer[mask_u8 > 0] = [245, 158, 11]
        cv2.addWeighted(amber_layer, 0.35, overlay, 0.65, 0, overlay)
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
                
                tick_len = 5 if is_bold else 3
                t1a = (int(round(p1[0] + tx/L * tick_len)), int(round(p1[1] + ty/L * tick_len)))
                t1b = (int(round(p1[0] - tx/L * tick_len)), int(round(p1[1] - ty/L * tick_len)))
                t2a = (int(round(p2[0] + tx/L * tick_len)), int(round(p2[1] + ty/L * tick_len)))
                t2b = (int(round(p2[0] - tx/L * tick_len)), int(round(p2[1] - ty/L * tick_len)))
                cv2.line(overlay, t1a, t1b, color, 1, lineType=cv2.LINE_AA)
                cv2.line(overlay, t2a, t2b, color, 1, lineType=cv2.LINE_AA)
                
                lx = int(max(p1[0], p2[0]) + 6)
                ly = int((p1[1] + p2[1]) / 2)
                lx = min(400, max(10, lx))
                ly = min(500, max(20, ly))
                cv2.putText(overlay, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        prox_mm = thick_px[p_idx] * mm_pp
        draw_caliper(p_idx, (50, 255, 50), f"Ref Prox: {prox_mm:.1f} mm")
        dist_mm = thick_px[d_idx] * mm_pp
        draw_caliper(d_idx, (50, 255, 50), f"Ref Dist: {dist_mm:.1f} mm")
        max_mm = thick_px[m_idx] * mm_pp
        draw_caliper(m_idx, (255, 50, 50), f"Dmax: {max_mm:.1f} mm", is_bold=True)
        
        if mld_idx is not None and 0 <= mld_idx < len(sp_x):
            mld_mm = thick_px[mld_idx] * mm_pp
            draw_caliper(mld_idx, (50, 150, 255), f"MLD: {mld_mm:.1f} mm")
    elif default_dims is not None:
        ref_p = default_dims.get("ref_prox", 3.0)
        ref_d = default_dims.get("ref_dist", 2.6)
        d_max = default_dims.get("max_diam", 6.2)
        mm_pp = default_dims.get("mm_pp", 0.20)
        
        def draw_simple_caliper(y, length_mm, color, label_text, is_bold=False):
            half_px = int(round((length_mm / mm_pp) / 2.0))
            p1 = (256 - half_px, y)
            p2 = (256 + half_px, y)
            thick = 3 if is_bold else 2
            cv2.line(overlay, p1, p2, color, thick, lineType=cv2.LINE_AA)
            cv2.line(overlay, (p1[0], y-4), (p1[0], y+4), color, 1, lineType=cv2.LINE_AA)
            cv2.line(overlay, (p2[0], y-4), (p2[0], y+4), color, 1, lineType=cv2.LINE_AA)
            cv2.putText(overlay, label_text, (p2[0]+8, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, label_text, (p2[0]+8, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
        draw_simple_caliper(190, ref_p, (50, 255, 50), f"Ref Prox: {ref_p:.1f} mm")
        draw_simple_caliper(256, d_max, (255, 50, 50), f"Dmax: {d_max:.1f} mm", is_bold=True)
        draw_simple_caliper(320, ref_d, (50, 255, 50), f"Ref Dist: {ref_d:.1f} mm")
            
    if zoom > 1.0:
        h, w = overlay.shape[:2]
        hz, wz = int(h / zoom), int(w / zoom)
        top = int(np.clip(focus_y - hz // 2, 0, h - hz))
        left = int(np.clip(focus_x - wz // 2, 0, w - wz))
        crop = overlay[top:top+hz, left:left+wz]
        overlay = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        
    return overlay

def get_norm_512(frame_pixels):
    norm = frame_pixels.astype(float)
    p_min, p_max = np.min(norm), np.max(norm)
    if p_max > p_min:
        norm = ((norm - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    else:
        norm = norm.astype(np.uint8)
    if norm.shape[0] != 512 or norm.shape[1] != 512:
        return cv2.resize(norm, (512, 512), interpolation=cv2.INTER_AREA)
    return norm

def get_series_gif(filepath, pixel_array, cine_rate=15):
    try:
        import tempfile
        tmp_gif_path = os.path.join(tempfile.gettempdir(), f"angio_preview_{os.path.basename(filepath)}_{os.path.getsize(filepath)}.gif")
        if not os.path.exists(tmp_gif_path):
            n_frames = pixel_array.shape[0]
            step = 1 if n_frames <= 30 else 2
            pil_frames = []
            for idx in range(0, n_frames, step):
                fr = pixel_array[idx].astype(float)
                p_min, p_max = fr.min(), fr.max()
                if p_max > p_min:
                    fr = ((fr - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
                else:
                    fr = fr.astype(np.uint8)
                if fr.shape[0] != 320 or fr.shape[1] != 320:
                    fr = cv2.resize(fr, (320, 320), interpolation=cv2.INTER_AREA)
                pil_frames.append(Image.fromarray(fr))
            if pil_frames:
                rate = float(cine_rate) if cine_rate > 0 else 15.0
                duration = int(1000.0 / (rate / step))
                pil_frames[0].save(tmp_gif_path, save_all=True, append_images=pil_frames[1:], duration=max(30, duration), loop=0)
        return tmp_gif_path if os.path.exists(tmp_gif_path) else None
    except Exception as e:
        print(f"Error generating GIF: {e}")
        return None

@st.cache_data(max_entries=150, ttl=3600, show_spinner=False)
def analyze_series_flow(dicom_path, file_size=None):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pixelArray = dcm.pixel_array
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        elif len(pixelArray.shape) == 3 and pixelArray.shape[2] in (3, 4):
            return 0, 0, 0
        if len(pixelArray.shape) == 2:
            pixelArray = np.expand_dims(pixelArray, axis=0)
        n_slices = pixelArray.shape[0]
        if n_slices <= 1:
            return 0, 0, 0
            
        pa_f = pixelArray.astype(np.float32)
        pmin, pmax = pa_f.min(), pa_f.max()
        if pmax > pmin:
            pa_f = (pa_f - pmin) / (pmax - pmin) * 255.0

        scores = []
        for i in range(n_slices):
            frame = pa_f[i]
            small = cv2.resize(frame, (256, 256))
            blurred = scipy.ndimage.gaussian_filter(small.astype(float), 2)
            grad = np.abs(np.gradient(blurred))
            scores.append(np.sum(grad))
            
        scores = np.array(scores)
        best_ix = int(np.argmax(scores))
        if best_ix < 3 and n_slices > 10:
            alt_best = int(np.argmax(scores[3:])) + 3
            if scores[alt_best] > scores[best_ix] * 0.8:
                best_ix = alt_best
                
        end_ix = n_slices - 1
        baseline = np.mean(scores[:3]) if len(scores) > 3 else scores[0]
        threshold = baseline + np.std(scores) * 1.5
        start_ix = 0
        for i in range(n_slices):
            if scores[i] > threshold:
                start_ix = i
                break
        if start_ix >= best_ix:
            start_ix = max(0, best_ix - 15)
            
        return best_ix, start_ix, end_ix
    except Exception as e:
        return 0, 0, 0

@st.cache_data(max_entries=150, ttl=3600, show_spinner=False)
def get_preview_image(dicom_path, start_ix, best_ix, end_ix, file_size=None):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pa = dcm.pixel_array
        if len(pa.shape) == 4: pa = pa[:, :, :, 0]
        elif len(pa.shape) == 3 and pa.shape[2] in (3, 4):
            pa = pa[:, :, 0]
            pa = np.expand_dims(pa, axis=0)
        if len(pa.shape) == 2: pa = np.expand_dims(pa, axis=0)
        
        n_slices = pa.shape[0]
        start_ix = max(0, min(n_slices - 1, start_ix))
        best_ix = max(0, min(n_slices - 1, best_ix))
        end_ix = max(0, min(n_slices - 1, end_ix))

        def _to_uint8(img):
            img_f = img.astype(np.float32)
            f_min, f_max = img_f.min(), img_f.max()
            if f_max > f_min:
                return ((img_f - f_min) / (f_max - f_min) * 255.0).astype(np.uint8)
            else:
                return np.zeros_like(img, dtype=np.uint8)
                
        frame_start = _to_uint8(pa[start_ix])
        frame_best  = _to_uint8(pa[best_ix])
        frame_end   = _to_uint8(pa[end_ix])
        img_start = cv2.resize(frame_start, (512, 512))
        img_best  = cv2.resize(frame_best,  (512, 512))
        img_end   = cv2.resize(frame_end,   (512, 512))
        return np.concatenate((img_start, img_best, img_end), axis=1)
    except Exception as e:
        return None

def get_best_or_saved_frame(active_pid, dfp, d_meta):
    """
    Returns the optimal frame index for a series:
    1. If user already chose a valid frame (> 0 or frame 0 has a saved mask), returns it.
    2. If any frame has an existing segmentation mask for this patient/series, returns that frame.
    3. Otherwise, automatically analyzes flow to detect the Peak QCA contrast frame.
    4. Falls back to middle frame if flow detection is inconclusive, never blank frame 0.
    """
    frame_key = f"caa_frame_{active_pid}_{os.path.basename(dfp)}"
    n_frames = d_meta.get("total_frames", 1)
    if n_frames <= 1:
        st.session_state[frame_key] = 0
        return 0

    # 1. Check current frame in session state
    if frame_key in st.session_state:
        cur_fr = st.session_state[frame_key]
        if 0 <= cur_fr < n_frames:
            case_key = f"{active_pid}_{os.path.basename(dfp)}_{cur_fr}"
            if cur_fr > 0 or st.session_state.get(f"caa_mask_{case_key}") is not None:
                return cur_fr

    # 2. Check if any frame has a saved mask
    base_prefix = f"caa_mask_{active_pid}_{os.path.basename(dfp)}_"
    saved_frames = []
    for k in list(st.session_state.keys()):
        if k.startswith(base_prefix) and st.session_state.get(k) is not None:
            tail = k[len(base_prefix):]
            if tail.isdigit():
                saved_frames.append(int(tail))
    if saved_frames:
        chosen_fr = min(max(0, saved_frames[0]), n_frames - 1)
        st.session_state[frame_key] = chosen_fr
        return chosen_fr

    # 3. Peak QCA flow analysis
    fsize = os.path.getsize(dfp) if os.path.exists(dfp) else None
    b_ix, _, _ = analyze_series_flow(dfp, fsize)
    if b_ix <= 0 and n_frames > 1:
        b_ix = min(int(n_frames / 2), n_frames - 1)
    chosen_fr = max(0, min(b_ix, n_frames - 1))
    st.session_state[frame_key] = chosen_fr
    return chosen_fr

def find_biplane_pairs(series_map, series_meta):
    """
    Finds groups of sequences chosen for analysis that share the exact same (vessel_system, aha_code).
    Returns list of matched biplane pairs.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for name, (dfp, d_meta) in series_map.items():
        m = series_meta.get(name, {})
        if m.get("chosen_for_analysis") and m.get("vessel_system") and m.get("aha_code"):
            groups[(m["vessel_system"], m["aha_code"], m.get("aha_label", ""))].append((name, dfp, d_meta))
            
    pairs = []
    for (vsys, code, lbl), items in groups.items():
        if len(items) >= 2:
            best_p1, best_p2 = items[0], items[1]
            max_diff = compute_3d_angle_diff(best_p1[2]["primary_angle"], best_p1[2]["secondary_angle"], best_p2[2]["primary_angle"], best_p2[2]["secondary_angle"])
            if len(items) > 2:
                for i in range(len(items)):
                    for j in range(i+1, len(items)):
                        diff = compute_3d_angle_diff(items[i][2]["primary_angle"], items[i][2]["secondary_angle"], items[j][2]["primary_angle"], items[j][2]["secondary_angle"])
                        if diff > max_diff:
                            max_diff = diff
                            best_p1, best_p2 = items[i], items[j]
            pairs.append({
                "system": vsys,
                "aha_code": code,
                "aha_label": lbl,
                "p1_name": best_p1[0],
                "p1_dfp": best_p1[1],
                "p1_meta": best_p1[2],
                "p2_name": best_p2[0],
                "p2_dfp": best_p2[1],
                "p2_meta": best_p2[2],
                "angle_diff": max_diff,
                "all_items": items
            })
    return pairs

@st.fragment
def render_series_card(active_pid, name, dfp, d_meta, series_meta, meta_store_key, card_idx=0, prefix="card"):
    file_id = hashlib.md5(str(name).encode("utf-8")).hexdigest()[:10]
    
    if meta_store_key in st.session_state:
        series_meta = st.session_state[meta_store_key]
        
    m = series_meta.setdefault(name, {
        "vessel_system": ALL_SYSTEM_NAMES[0],
        "aha_code": _seg_codes(ALL_SYSTEM_NAMES[0])[1] if len(_seg_codes(ALL_SYSTEM_NAMES[0])) > 1 else _seg_codes(ALL_SYSTEM_NAMES[0])[0],
        "aha_label": _seg_labels(ALL_SYSTEM_NAMES[0])[1] if len(_seg_labels(ALL_SYSTEM_NAMES[0])) > 1 else _seg_labels(ALL_SYSTEM_NAMES[0])[0],
        "chosen_for_analysis": False
    })
    
    chk_key = f"chk_chosen_{active_pid}_{file_id}"
    sys_key = f"vessel_sys_{active_pid}_{file_id}"
    
    # Synchronize chosen status from session_state widget if present
    if chk_key in st.session_state:
        m["chosen_for_analysis"] = bool(st.session_state[chk_key])
        
    chosen = m.get("chosen_for_analysis", False)
    chosen_badge = "⭐ " if chosen else ""
    
    active_fr = get_best_or_saved_frame(active_pid, dfp, d_meta)
    has_mask = bool(st.session_state.get(f"caa_mask_{active_pid}_{os.path.basename(dfp)}_{active_fr}") is not None)
    status_dot = "🟢" if has_mask else "⚪"
    
    with st.expander(f"{status_dot} {chosen_badge}Seria: {d_meta['series_desc']} (Kąty: {d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°, {d_meta['total_frames']} klatek)", expanded=True):
        # 1. Controls row: Chosen checkbox + Vessel System & AHA Segment
        c_top_chk, c_v1, c_v2 = st.columns([1.2, 1.8, 2.0])
        with c_top_chk:
            new_chosen = st.checkbox("⭐ Wybierz do analizy", value=chosen, key=chk_key)
            if new_chosen != chosen:
                m["chosen_for_analysis"] = new_chosen
                st.session_state[meta_store_key] = series_meta
                
        with c_v1:
            if sys_key in st.session_state and st.session_state[sys_key] in ALL_SYSTEM_NAMES:
                m["vessel_system"] = st.session_state[sys_key]
            cur_sys = m.get("vessel_system") or ALL_SYSTEM_NAMES[0]
            if cur_sys not in ALL_SYSTEM_NAMES:
                cur_sys = ALL_SYSTEM_NAMES[0]
            chosen_sys = st.selectbox("Naczynie:", ALL_SYSTEM_NAMES, index=ALL_SYSTEM_NAMES.index(cur_sys), key=sys_key)
            if chosen_sys != cur_sys:
                m["vessel_system"] = chosen_sys
                m["aha_label"] = _seg_labels(chosen_sys)[0]
                m["aha_code"] = _seg_codes(chosen_sys)[0]
                st.session_state[meta_store_key] = series_meta
                st.rerun(scope="fragment")
                
        with c_v2:
            seg_labels = _seg_labels(cur_sys)
            seg_codes = _seg_codes(cur_sys)
            seg_key = f"aha_seg_{active_pid}_{file_id}_{cur_sys}"
            if seg_key in st.session_state and st.session_state[seg_key] in seg_labels:
                m["aha_label"] = st.session_state[seg_key]
                m["aha_code"] = seg_codes[seg_labels.index(m["aha_label"])]
            cur_lbl = m.get("aha_label") or seg_labels[0]
            if cur_lbl not in seg_labels:
                cur_lbl = seg_labels[0]
            chosen_lbl = st.selectbox("Segment AHA:", seg_labels, index=seg_labels.index(cur_lbl), key=seg_key)
            if chosen_lbl != cur_lbl:
                m["aha_label"] = chosen_lbl
                m["aha_code"] = seg_codes[seg_labels.index(chosen_lbl)]
                st.session_state[meta_store_key] = series_meta

        # 2. Preview image row: 3-panel strip (Start | Peak QCA | Last Frame) or animated CINE
        play_state_key = f"play_gif_{active_pid}_{file_id}"
        play_state = st.session_state.get(play_state_key, False)
        if play_state:
            gif_path = get_series_gif(dfp, d_meta["pixels"], d_meta.get("cine_rate", 15))
            if gif_path:
                safe_display_image(gif_path, caption=f"Animacja sekwencji | {d_meta['total_frames']} klatek")
            else:
                norm_f = get_norm_512(d_meta["pixels"][min(int(d_meta['total_frames']/2), d_meta['total_frames']-1)])
                safe_display_image(norm_f, caption=f"Klatka środkowa")
        else:
            fsize = os.path.getsize(dfp) if os.path.exists(dfp) else None
            b_ix, s_ix, e_ix = analyze_series_flow(dfp, fsize)
            combined = get_preview_image(dfp, s_ix, b_ix, e_ix, fsize)
            if combined is not None:
                safe_display_image(Image.fromarray(combined), caption=f"Start (Klatka {s_ix+1}) | Peak QCA (Klatka {b_ix+1}) | Last Frame (Klatka {e_ix+1})")
            else:
                norm_f = get_norm_512(d_meta["pixels"][min(int(d_meta['total_frames']/2), d_meta['total_frames']-1)])
                safe_display_image(norm_f, caption=f"Klatka referencyjna (Kąty: {d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°)")

        # 3. Bottom row: metadata & action buttons
        c_b1, c_b2, c_b3 = st.columns([1, 1.8, 1.4])
        with c_b1:
            btn_lbl = "⏹️ Stop" if play_state else "🎥 Odtwórz CINE"
            if st.button(btn_lbl, key=f"btn_play_{active_pid}_{file_id}", use_container_width=True):
                st.session_state[play_state_key] = not play_state
                st.rerun(scope="fragment")
        with c_b2:
            if has_mask:
                prof = st.session_state.get(f"caa_prof_{active_pid}_{os.path.basename(dfp)}_{active_fr}")
                dmax_txt = f"{prof['max_diam_mm']} mm" if prof else ""
                st.success(f"🟢 Obrysowana (Klatka {active_fr + 1}, Dmax={dmax_txt})")
            else:
                st.caption(f"Kąty: **{d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°** | Klatki: **{d_meta['total_frames']}** | Skala: **{d_meta['spacing']:.3f} mm**")
        with c_b3:
            if st.button("🎯 Obrysuj tę projekcję", key=f"btn_delineate_{active_pid}_{file_id}", type="primary", use_container_width=True):
                st.session_state["caa_active_series_name"] = name
                fsize = os.path.getsize(dfp) if os.path.exists(dfp) else None
                b_ix, _, _ = analyze_series_flow(dfp, fsize)
                st.session_state[f"caa_frame_{active_pid}_{os.path.basename(dfp)}"] = b_ix
                st.session_state["caa_target_view"] = "single_delineation"
                st.rerun(scope="app")

def reset_patient_workspace(active_pid):
    """
    Completely resets all series choices, AHA assignments, calibrations, masks,
    profiles, landmarks, and paired views for the given patient.
    """
    meta_store_key = f"caa_series_meta_{active_pid}"
    st.session_state.pop(meta_store_key, None)
    
    sort_applied_key = f"caa_sort_applied_{active_pid}"
    st.session_state.pop(sort_applied_key, None)
    
    keys_to_clear = [k for k in list(st.session_state.keys()) if active_pid in k and (
        k.startswith("caa_") or k.startswith("chk_chosen_") or k.startswith("vessel_sys_") or k.startswith("aha_seg_") or k.startswith("play_gif_") or k.startswith("btn_")
    )]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
        
    for k in ["caa_active_pair", "caa_active_biplane_pair", "caa_active_series_name", "caa_target_view"]:
        st.session_state.pop(k, None)

# ── 1. WIDOK: PRZEGLĄD WSZYSTKICH PROJEKCJI Z KORONAROGRAFII (GALLERY) ────────
def render_projections_gallery(active_pid, series_map):
    meta_store_key = f"caa_series_meta_{active_pid}"
    if meta_store_key not in st.session_state:
        st.session_state[meta_store_key] = {}
        
    series_meta = st.session_state[meta_store_key]
    
    # Initialize series metadata defaults if missing
    for idx, (name, (dfp, d_meta)) in enumerate(series_map.items()):
        if name not in series_meta:
            series_meta[name] = {
                "vessel_system": ALL_SYSTEM_NAMES[0],
                "aha_code": _seg_codes(ALL_SYSTEM_NAMES[0])[1] if len(_seg_codes(ALL_SYSTEM_NAMES[0])) > 1 else _seg_codes(ALL_SYSTEM_NAMES[0])[0],
                "aha_label": _seg_labels(ALL_SYSTEM_NAMES[0])[1] if len(_seg_labels(ALL_SYSTEM_NAMES[0])) > 1 else _seg_labels(ALL_SYSTEM_NAMES[0])[0],
                "chosen_for_analysis": False
            }

    sort_applied_key = f"caa_sort_applied_{active_pid}"
    sort_applied = st.session_state.get(sort_applied_key, False)
    
    all_series_items = list(series_map.items())
    chosen_items = [(name, dfp, d_meta) for (name, (dfp, d_meta)) in all_series_items if series_meta.get(name, {}).get("chosen_for_analysis")]
    unchosen_items = [(name, dfp, d_meta) for (name, (dfp, d_meta)) in all_series_items if not series_meta.get(name, {}).get("chosen_for_analysis")]
    n_chosen = len(chosen_items)
    
    # Top header with prominent Sortuj action
    c_hdr1, c_hdr2, c_hdr3, c_hdr4 = st.columns([2.2, 1.2, 1.1, 0.9])
    with c_hdr1:
        st.markdown(f"### 🗂️ Przegląd projekcji koronarografii *({len(series_map)})*")
    with c_hdr2:
        if not sort_applied:
            btn_txt = f"🔀 Sortuj ({n_chosen})" if n_chosen > 0 else "🔀 Sortuj"
            if st.button(btn_txt, key=f"btn_sort_top_{active_pid}", type="primary" if n_chosen > 0 else "secondary", use_container_width=True, help="Układa wybrane projekcje w pary do analizy"):
                st.session_state[sort_applied_key] = True
                st.rerun()
        else:
            if st.button(f"🔀 Zaktualizuj ({n_chosen})", key=f"btn_resort_top_{active_pid}", type="primary", use_container_width=True, help="Odświeża sortowanie i łączenie w pary"):
                st.session_state[sort_applied_key] = True
                st.rerun()
    with c_hdr3:
        if sort_applied:
            if st.button("↩️ Całe badanie", key=f"btn_unsort_top_{active_pid}", help="Pokaż wszystkie projekcje po kolei bez podziału", use_container_width=True):
                st.session_state[sort_applied_key] = False
                st.rerun()
        else:
            st.caption(f"Wybrano: **{n_chosen}** projekcji")
    with c_hdr4:
        if st.button("🔄 Resetuj", key=f"btn_reset_all_choices_{active_pid}", help="Cofa wszystkie zaznaczenia, kalibracje i obrysy dla tego pacjenta", use_container_width=True):
            reset_patient_workspace(active_pid)
            st.rerun()

    # If sorted mode: display pairs, chosen ordered jedne pod drugą, and unchosen
    if sort_applied:
        # Auto-detect biplane pairs among chosen
        detected_pairs = find_biplane_pairs(series_map, series_meta)
        
        if detected_pairs:
            st.markdown("---")
            for p_idx, pair in enumerate(detected_pairs):
                diff_deg = pair["angle_diff"]
                is_valid_angle = (diff_deg >= 30.0)
                status_icon = "✅" if is_valid_angle else "⚠️"
                
                p1_active_frame = get_best_or_saved_frame(active_pid, pair['p1_dfp'], pair['p1_meta'])
                p2_active_frame = get_best_or_saved_frame(active_pid, pair['p2_dfp'], pair['p2_meta'])
                
                p1_has_mask = bool(st.session_state.get(f"caa_mask_{active_pid}_{os.path.basename(pair['p1_dfp'])}_{p1_active_frame}") is not None)
                p2_has_mask = bool(st.session_state.get(f"caa_mask_{active_pid}_{os.path.basename(pair['p2_dfp'])}_{p2_active_frame}") is not None)
                
                p1_badge = "🟢 Obrysowana" if p1_has_mask else "⚪ Do obrysowania"
                p2_badge = "🟢 Obrysowana" if p2_has_mask else "⚪ Do obrysowania"
                
                st.markdown(f"""
                <div style='background-color: #042f2e; border: 1px solid #0f766e; border-left: 6px solid #14b8a6; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px;'>
                    <div style='font-size: 16px; font-weight: 700; color: #5eead4;'>
                        🎉 Wykryto parę do rekonstrukcji Simpsona 3D: <u>{pair['aha_label']}</u>
                    </div>
                    <div style='margin-top: 6px; font-size: 14px; color: #ccfbf1;'>
                        📹 <b>Projekcja 1:</b> {pair['p1_meta']['series_desc']} ({pair['p1_meta']['primary_angle']:+.1f}° / {pair['p1_meta']['secondary_angle']:+.1f}°) — <span style='color: {"#34d399" if p1_has_mask else "#facc15"}; font-weight: 600;'>{p1_badge}</span><br/>
                        🌐 <b>Projekcja 2:</b> {pair['p2_meta']['series_desc']} ({pair['p2_meta']['primary_angle']:+.1f}° / {pair['p2_meta']['secondary_angle']:+.1f}°) — <span style='color: {"#34d399" if p2_has_mask else "#facc15"}; font-weight: 600;'>{p2_badge}</span><br/>
                        📐 <b>Różnica kątów w przestrzeni 3D:</b> <b>{diff_deg:.1f}°</b> {status_icon} {"(Spełnia warunek ≥ 30° dla reguły Simpsona)" if is_valid_angle else "(Zalecane ≥ 30° dla optymalnej dokładności 3D)"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                c_pair_btn0, c_pair_btn1, c_pair_btn2, c_pair_btn3 = st.columns([1.6, 1.4, 1.0, 1.0])
                with c_pair_btn0:
                    if st.button(f"✨ Obrysuj razem w parze ({pair['aha_label']})", key=f"btn_pair_delineate_top_{p_idx}", type="primary", use_container_width=True):
                        st.session_state["caa_active_pair"] = pair
                        st.session_state["caa_target_view"] = "paired_delineation"
                        st.session_state[f"caa_pair_substep_{active_pid}"] = "1️⃣ Kalibracja cewnika (P1 i P2)"
                        st.rerun()
                with c_pair_btn1:
                    if p1_has_mask and p2_has_mask:
                        btn_text = f"🚀 Wyniki Simpsona 3D"
                        btn_type = "primary"
                    else:
                        btn_text = f"🔍 Widok Simpsona 3D"
                        btn_type = "secondary"
                    if st.button(btn_text, key=f"btn_open_biplane_pair_{p_idx}", type=btn_type, use_container_width=True):
                        st.session_state["caa_active_biplane_pair"] = pair
                        st.session_state["caa_target_view"] = "biplane_simpson"
                        st.rerun()
                with c_pair_btn2:
                    if st.button(f"🎯 Projekcja 1", key=f"btn_outline_p1_{p_idx}", use_container_width=True):
                        st.session_state["caa_active_series_name"] = pair["p1_name"]
                        st.session_state["caa_target_view"] = "single_delineation"
                        st.rerun()
                with c_pair_btn3:
                    if st.button(f"🎯 Projekcja 2", key=f"btn_outline_p2_{p_idx}", use_container_width=True):
                        st.session_state["caa_active_series_name"] = pair["p2_name"]
                        st.session_state["caa_target_view"] = "single_delineation"
                        st.rerun()
            st.markdown("---")

        # Sort chosen items by vessel system and AHA segment
        def _item_sort_key(item):
            name, dfp, d_meta = item
            m = series_meta.get(name, {})
            vsys = m.get("vessel_system", ALL_SYSTEM_NAMES[0])
            code = m.get("aha_code", "99")
            v_order = {"LM & LAD – Left Main & Left Anterior Descending": 1, "LCx – Left Circumflex": 2, "RCA – Right Coronary Artery": 3}.get(vsys, 99)
            try:
                s_order = _seg_codes(vsys).index(code)
            except ValueError:
                s_order = 99
            return (v_order, s_order, name)

        chosen_items.sort(key=_item_sort_key)

        # 1. Chosen section: sorted and placed one under the other (jedne pod drugą)
        st.markdown(f"#### 🎯 Projekcje wybrane do analizy ({len(chosen_items)}):")
        st.caption("Wybrane sekwencje są posortowane anatomicznie i ułożone jedne pod drugą. Projekcje dla tego samego segmentu tworzą parę do jednoczesnej analizy.")

        if chosen_items:
            from itertools import groupby
            for (vsys, code), group_iter in groupby(chosen_items, key=lambda it: (series_meta.get(it[0], {}).get("vessel_system"), series_meta.get(it[0], {}).get("aha_code"))):
                group_list = list(group_iter)
                group_label = series_meta.get(group_list[0][0], {}).get("aha_label", f"{vsys} Seg {code}")
                
                if len(group_list) >= 2:
                    p1_it, p2_it = group_list[0], group_list[1]
                    p_diff = compute_3d_angle_diff(p1_it[2]["primary_angle"], p1_it[2]["secondary_angle"], p2_it[2]["primary_angle"], p2_it[2]["secondary_angle"])
                    
                    c_grp1, c_grp2 = st.columns([2.5, 1.5])
                    with c_grp1:
                        st.markdown(f"##### 🔗 Para biplanarna: `{group_label}` (Kąt 3D: **{p_diff:.1f}°**)")
                    with c_grp2:
                        matched_pair_obj = None
                        if detected_pairs:
                            for dp in detected_pairs:
                                if dp["system"] == vsys and dp["aha_code"] == code:
                                    matched_pair_obj = dp
                                    break
                        if not matched_pair_obj:
                            matched_pair_obj = {
                                "system": vsys,
                                "aha_code": code,
                                "aha_label": group_label,
                                "p1_name": p1_it[0], "p1_dfp": p1_it[1], "p1_meta": p1_it[2],
                                "p2_name": p2_it[0], "p2_dfp": p2_it[1], "p2_meta": p2_it[2],
                                "angle_diff": p_diff
                            }
                        if st.button(f"✨ Obrysuj razem dla {group_label}", key=f"btn_inline_pair_{vsys}_{code}", type="primary", use_container_width=True):
                            st.session_state["caa_active_pair"] = matched_pair_obj
                            st.session_state["caa_target_view"] = "paired_delineation"
                            st.session_state[f"caa_pair_substep_{active_pid}"] = "1️⃣ Kalibracja cewnika (P1 i P2)"
                            st.rerun()

                for c_idx, (name, dfp, d_meta) in enumerate(group_list):
                    render_series_card(active_pid, name, dfp, d_meta, series_meta, meta_store_key, card_idx=f"chosen_{c_idx}_{name[:12]}", prefix="chosen")
        else:
            st.info("ℹ️ Nie wybrano jeszcze żadnej projekcji do analizy. Zaznacz '⭐ Wybierz do analizy' przy co najmniej 1–2 projekcjach poniżej, a następnie kliknij 'Zaktualizuj'.")

        # 2. Unchosen section
        if unchosen_items:
            st.markdown("---")
            section_title = f"🗂️ Wszystkie projekcje z badania ({len(unchosen_items)}):" if len(chosen_items) == 0 else f"🗂️ Wszystkie pozostałe projekcje z badania ({len(unchosen_items)}):"
            st.markdown(f"#### {section_title}")
            for u_idx, (name, dfp, d_meta) in enumerate(unchosen_items):
                render_series_card(active_pid, name, dfp, d_meta, series_meta, meta_store_key, card_idx=f"unchosen_{u_idx}_{name[:12]}", prefix="unchosen")

    else:
        # Selection mode: All series in sequential order without moving around
        st.info("💡 **Tryb wyboru projekcji:** Zaznacz **⭐ Wybierz do analizy**, wskaż naczynie i segment dla interesujących Cię projekcji (wybór nie przeładowuje widoku). Gdy skończysz wybierać, kliknij **🔀 Sortuj** na górze lub na dole, aby ułożyć je w pary do analizy.")
        
        for s_idx, (name, (dfp, d_meta)) in enumerate(all_series_items):
            render_series_card(active_pid, name, dfp, d_meta, series_meta, meta_store_key, card_idx=f"seq_{s_idx}_{name[:12]}", prefix="seq")
            
        st.markdown("---")
        c_bot1, c_bot2 = st.columns([3, 1.5])
        with c_bot1:
            st.markdown("Gotowe? Kliknij **Sortuj**, aby ułożyć wybrane projekcje w pary do analizy Simpsona 3D:")
        with c_bot2:
            if st.button("🔀 Sortuj wybrane do analizy", key=f"btn_sort_bottom_{active_pid}", type="primary", use_container_width=True):
                st.session_state[sort_applied_key] = True
                st.rerun()

# ── 2. WIDOK: OBRYSOWANIE POJEDYNCZEJ PROJEKCJI ──────────────────────────────
def render_single_delineation_view(active_pid, series_map):
    active_name = st.session_state.get("caa_active_series_name")
    if not active_name or active_name not in series_map:
        active_name = list(series_map.keys())[0]
        st.session_state["caa_active_series_name"] = active_name
        
    dfp, d_meta = series_map[active_name]
    meta_store = st.session_state.get(f"caa_series_meta_{active_pid}", {})
    s_meta = meta_store.get(active_name, {})
    
    c_top1, c_top2 = st.columns([3, 1])
    with c_top1:
        st.markdown(f"### 🎯 Delineacja Projekcji: **{d_meta['series_desc']}**")
        st.caption(f"Kąty: LAO/RAO **{d_meta['primary_angle']:+.1f}°**, CRA/CAU **{d_meta['secondary_angle']:+.1f}°** | Przypisany segment: **{s_meta.get('aha_label', 'Nieustalony')}**")
    with c_top2:
        if st.button("🔙 Wróć do przeglądu (Gallery)", use_container_width=True):
            st.session_state["caa_target_view"] = "gallery"
            st.rerun()
            
    # Check if this series is part of a biplane pair
    detected_pairs = find_biplane_pairs(series_map, meta_store)
    matching_pair = None
    for p in detected_pairs:
        if p["p1_name"] == active_name or p["p2_name"] == active_name:
            matching_pair = p
            break
            
    if matching_pair:
        other_name = matching_pair["p2_name"] if matching_pair["p1_name"] == active_name else matching_pair["p1_name"]
        other_dfp, other_meta = series_map[other_name]
        st.info(f"💡 Ta projekcja tworzy parę biplanarną z: **{other_meta['series_desc']}** ({other_meta['primary_angle']:+.1f}° / {other_meta['secondary_angle']:+.1f}°) dla segmentu **{matching_pair['aha_label']}** (Różnica kątów 3D: **{matching_pair['angle_diff']:.1f}°**).")
        
    # Cine Frame Slider
    n_frames = d_meta["total_frames"]
    frame_key = f"caa_frame_{active_pid}_{os.path.basename(dfp)}"
    if frame_key not in st.session_state:
        st.session_state[frame_key] = min(int(n_frames / 2), n_frames - 1)
        
    frame_slider = st.slider("Numer klatki (Cine Frame):", min_value=0, max_value=max(0, n_frames - 1), value=st.session_state[frame_key], key=f"sl_{frame_key}")
    st.session_state[frame_key] = frame_slider
    
    frame_pixels = d_meta["pixels"][frame_slider]
    norm_512 = get_norm_512(frame_pixels)
    norm_rgb = cv2.cvtColor(norm_512, cv2.COLOR_GRAY2RGB)
    dicom_mm_per_pixel = d_meta["spacing"] * (d_meta["pixels"].shape[-1] / 512.0)
    
    case_key = f"{active_pid}_{os.path.basename(dfp)}_{frame_slider}"
    mask_key = f"caa_mask_{case_key}"
    prof_key = f"caa_prof_{case_key}"
    lm_key = f"caa_lm_{case_key}"
    calib_key = f"caa_calib_{active_pid}_{os.path.basename(dfp)}"
    
    calib_info = st.session_state.get(calib_key, None)
    if calib_info is not None:
        active_mm_pp = calib_info["mm_per_pixel"]
        calib_badge = f"🟢 Cewnik {calib_info.get('catheter_name', 'cewnik')}: {calib_info['avg_diam_px']:.1f} px = {calib_info.get('catheter_mm', 2.0):.2f} mm → **{active_mm_pp:.4f} mm/px**"
    else:
        active_mm_pp = dicom_mm_per_pixel
        calib_badge = f"ℹ️ Brak kalibracji cewnika – użyto metadanych DICOM: **{active_mm_pp:.4f} mm/px**"

    active_mask = st.session_state.get(mask_key, None)
    active_profile = st.session_state.get(prof_key, None)
    active_landmarks = st.session_state.get(lm_key, None)
    
    # Sub-steps
    st.markdown("---")
    substep_key = f"caa_substep_{active_pid}_{os.path.basename(dfp)}"
    if "caa_target_substep" in st.session_state:
        st.session_state[substep_key] = st.session_state.pop("caa_target_substep")
    if substep_key not in st.session_state:
        st.session_state[substep_key] = "1️⃣ Kalibracja cewnika"
        
    current_substep = st.radio(
        "Kolejność procedury:",
        ["1️⃣ Kalibracja cewnika", "2️⃣ Obrysowanie tętniaka (AI)", "3️⃣ Pomiary i kalipery"],
        horizontal=True,
        key=substep_key
    )
    
    col_vis, col_params = st.columns([1.3, 1])
    
    # ── KROK 1: KALIBRACJA CEWNIKA ──
    if current_substep == "1️⃣ Kalibracja cewnika":
        with col_vis:
            st.markdown("#### 📏 Kalibracja cewnika naczyniowego (Auto-QCA)")
            st.markdown("""
            <div style='background-color: #042f2e; border-left: 4px solid #14b8a6; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px; font-size: 14px; color: #ccfbf1;'>
                1. Kliknij <b>2 punkty wzdłuż trzonu cewnika</b> na obrazie (czerwone kropki).<br/>
                2. Silnik subpikselowy samoczynnie wykryje krawędzie cewnika i przeliczy skalę <code>mm/px</code>.
            </div>
            """, unsafe_allow_html=True)
            
            calib_bg = norm_rgb.copy()
            if calib_info is not None:
                if "lw" in calib_info and "rw" in calib_info:
                    for pt in calib_info["lw"]:
                        cv2.circle(calib_bg, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 0), -1)
                    for pt in calib_info["rw"]:
                        cv2.circle(calib_bg, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 0), -1)
                if "click1" in calib_info and "click2" in calib_info:
                    p1_i, p2_i = calib_info["click1"], calib_info["click2"]
                    cv2.line(calib_bg, p1_i, p2_i, (0, 255, 0), 1)
                    cv2.circle(calib_bg, p1_i, 3, (0, 0, 255), -1)
                    cv2.circle(calib_bg, p2_i, 3, (0, 0, 255), -1)
                    
            calib_canvas_key = f"calib_cv_{case_key}_{st.session_state.get('caa_calib_sfx', 0)}"
            c_canvas = st_canvas(
                fill_color="#ff0000",
                stroke_width=0,
                stroke_color="#ff0000",
                background_color="black",
                background_image=Image.fromarray(calib_bg),
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode="point",
                point_display_radius=2,
                key=calib_canvas_key
            )
            
        with col_params:
            st.markdown("#### Parametry cewnika")
            CATHETER_SIZES = {
                "6F = 1.98 mm": 1.98,
                "5F = 1.67 mm": 1.67,
                "7F = 2.33 mm": 2.33,
                "4F = 1.35 mm": 1.35,
                "8F = 2.67 mm": 2.67,
                "Własny rozmiar (Custom mm)": None
            }
            cat_choice = st.selectbox("Rozmiar cewnika:", list(CATHETER_SIZES.keys()), index=0, key=f"caa_cat_{case_key}")
            if cat_choice == "Własny rozmiar (Custom mm)":
                catheter_mm = st.number_input("Średnica cewnika [mm]:", min_value=0.5, max_value=5.0, value=1.98, step=0.05)
            else:
                catheter_mm = CATHETER_SIZES[cat_choice]
                
            # If catheter size changed on existing calibration:
            if calib_info is not None and (calib_info.get("catheter_mm") != catheter_mm or calib_info.get("catheter_name") != cat_choice):
                calib_info["catheter_name"] = cat_choice
                calib_info["catheter_mm"] = catheter_mm
                calib_info["mm_per_pixel"] = float(catheter_mm / calib_info["avg_diam_px"])
                st.session_state[calib_key] = calib_info
                if active_mask is not None:
                    prof = extract_aneurysm_profile(active_mask, calib_info["mm_per_pixel"])
                    if prof:
                        st.session_state[prof_key] = prof
                st.rerun()

            # Automatic detection as soon as 2 points exist:
            if c_canvas.json_data is not None and "objects" in c_canvas.json_data:
                objs = c_canvas.json_data["objects"]
                if len(objs) >= 2:
                    r1 = float(objs[0].get('radius', 2))
                    p1 = (float(objs[0].get('left', 0)) + r1, float(objs[0].get('top', 0)) + r1)
                    r2 = float(objs[1].get('radius', 2))
                    p2 = (float(objs[1].get('left', 0)) + r2, float(objs[1].get('top', 0)) + r2)
                    
                    eval_sig = (round(p1[0], 1), round(p1[1], 1), round(p2[0], 1), round(p2[1], 1), catheter_mm, cat_choice)
                    last_sig = st.session_state.get(f"caa_last_calib_eval_{case_key}", None)
                    
                    if last_sig != eval_sig:
                        st.session_state[f"caa_last_calib_eval_{case_key}"] = eval_sig
                        res = calibrate_catheter(norm_512, p1, p2, catheter_mm)
                        if res is not None:
                            res["catheter_name"] = cat_choice
                            res["catheter_mm"] = catheter_mm
                            st.session_state[calib_key] = res
                            if active_mask is not None:
                                prof = extract_aneurysm_profile(active_mask, res["mm_per_pixel"])
                                if prof:
                                    st.session_state[prof_key] = prof
                            st.session_state["caa_calib_sfx"] = st.session_state.get("caa_calib_sfx", 0) + 1
                            st.rerun()
                        else:
                            st.error("⚠️ Nie udało się precyzyjnie wykryć krawędzi cewnika. Wskaż 2 punkty w prostym, widocznym odcinku cewnika.")
                                
            if calib_info is not None:
                st.success(f"✅ Kalibracja ({calib_info.get('catheter_name', cat_choice)}): {calib_info['avg_diam_px']:.1f} px = {calib_info['catheter_mm']:.2f} mm → **{calib_info['mm_per_pixel']:.4f} mm/px**")
                if st.button("➡️ Przejdź do obrysowania tętniaka (Krok 2)", type="primary", use_container_width=True):
                    st.session_state["caa_target_substep"] = "2️⃣ Obrysowanie tętniaka (AI)"
                    st.rerun()
                if st.button("🔄 Skasuj i powtórz kalibrację", use_container_width=True):
                    st.session_state.pop(calib_key, None)
                    st.session_state.pop(f"caa_last_calib_eval_{case_key}", None)
                    st.session_state["caa_calib_sfx"] = st.session_state.get("caa_calib_sfx", 0) + 1
                    st.rerun()
            else:
                st.info(calib_badge)
                st.caption("Wskaż 2 punkty na cewniku na obrazie obok — system samoczynnie wykryje średnicę cewnika.")
                if st.button("⚡ Użyj kalibracji DICOM i przejdź do Kroku 2", use_container_width=True):
                    st.session_state["caa_target_substep"] = "2️⃣ Obrysowanie tętniaka (AI)"
                    st.rerun()

    # ── KROK 2: OBRYSOWANIE TĘTNIAKA (AI) ──
    elif current_substep == "2️⃣ Obrysowanie tętniaka (AI)":
        with col_vis:
            st.markdown("#### 🎯 Wskazywanie tętnicy i segmentacja AI (angioPy)")
            st.caption(calib_badge)
            st.markdown("""
            <div style='background-color: #0f172a; border-left: 4px solid #38bdf8; padding: 10px 14px; border-radius: 4px; margin-bottom: 12px;'>
                1. Kliknij <b>2 do 4 czerwonych kropek</b> wzdłuż światła naczynia (początek, wybrzuszenie tętniaka, koniec).<br/>
                2. Kliknij zielony przycisk <b>'🚀 Segmentuj tętniak (AI)'</b>.
            </div>
            """, unsafe_allow_html=True)
            
            canvas_key = f"seg_cv_{case_key}_{st.session_state.get('caa_cv_sfx', 0)}"
            annot_bg = norm_rgb.copy()
            if active_mask is not None and np.sum(active_mask) > 0:
                v_mask = (active_mask > 0).astype(np.uint8) * 255
                contour_preview = angioPyFunctions.maskOutliner(labelledArtery=v_mask, outlineThickness=1)
                annot_bg[contour_preview, :] = [0, 255, 0]
            pil_for_canvas = Image.fromarray(annot_bg)
            
            annotation_canvas = st_canvas(
                fill_color="#ff0000",
                stroke_width=0,
                stroke_color="#ff0000",
                background_color="black",
                background_image=pil_for_canvas,
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode="point",
                point_display_radius=2,
                key=canvas_key
            )
            
            c_seg1, c_seg2 = st.columns(2)
            with c_seg1:
                if st.button("🚀 Segmentuj tętniak (AI)", type="primary", key=f"btn_run_seg_{case_key}", use_container_width=True):
                    if annotation_canvas.json_data is not None and "objects" in annotation_canvas.json_data:
                        objs = annotation_canvas.json_data["objects"]
                        if len(objs) >= 2:
                            with st.spinner(f"Segmentacja naczynia silnikiem angioPy na podstawie {len(objs)} punktów..."):
                                try:
                                    pts = []
                                    for o in objs:
                                        r = float(o.get('radius', 2))
                                        cy = float(o.get('top', 0)) + r
                                        cx = float(o.get('left', 0)) + r
                                        pts.append([cy, cx])
                                    pts = np.array(pts, dtype=np.float32)
                                    
                                    mask = angioPyFunctions.arterySegmentation(norm_512, pts)
                                    mask_bin = (mask > 0).astype(np.uint8)
                                    
                                    prof = extract_aneurysm_profile(mask_bin, active_mm_pp)
                                    if prof is not None:
                                        st.session_state[mask_key] = mask_bin
                                        st.session_state[prof_key] = prof
                                        st.session_state[lm_key] = {
                                            "prox": prof["prox_idx"],
                                            "dist": prof["dist_idx"],
                                            "max": prof["max_idx"]
                                        }
                                        st.session_state["caa_target_substep"] = "3️⃣ Pomiary i kalipery"
                                        st.success("✅ Sukces: Tętniak został obrysowany!")
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error("Nie udało się wyznaczyć osi naczynia. Upewnij się, że punkty leżą wzdłuż światła tętnicy.")
                                except Exception as e:
                                    st.error(f"Błąd podczas segmentacji: {e}")
                        else:
                            st.warning("Kliknij co najmniej 2 punkty na naczyniu przed uruchomieniem.")
                    else:
                        st.warning("Kliknij najpierw punkty na obrazie.")
                            
            with c_seg2:
                if st.button("🗑️ Wyczyść punkty", key=f"btn_clr_{case_key}", use_container_width=True):
                    st.session_state["caa_cv_sfx"] = st.session_state.get("caa_cv_sfx", 0) + 1
                    st.rerun()

        with col_params:
            st.markdown("#### Status obrysu")
            if active_mask is not None:
                st.success("✅ Obrys tętniaka jest aktywny.")
                if st.button("👁️ Przejdź do pomiarów i kaliperów (Krok 3)", type="primary", use_container_width=True):
                    st.session_state["caa_target_substep"] = "3️⃣ Pomiary i kalipery"
                    st.rerun()
            else:
                st.info("Brak obrysu. Wskaż punkty na naczyniu i kliknij przycisk po lewej stronie.")

    # ── KROK 3: POMIARY I KALIPERY ──
    elif current_substep == "3️⃣ Pomiary i kalipery":
        with col_vis:
            st.markdown("#### 🔬 Podgląd obrysu, osi naczynia i kaliperów")
            st.caption(calib_badge)
            
            default_dims = {
                "ref_prox": 3.0,
                "ref_dist": 2.6,
                "max_diam": 6.2,
                "mm_pp": active_mm_pp
            }
            overlay_img = render_aneurysm_overlay(norm_512, active_mask, active_profile, active_landmarks, default_dims=default_dims)
            safe_display_image(overlay_img, caption=f"Klatka {frame_slider + 1}/{n_frames} | {d_meta['series_desc']} | Kąty: {d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°")
            
            if active_mask is not None and active_profile is not None:
                with st.expander("📍 Korekta pozycji znaczników (Landmarks)", expanded=True):
                    N_pts = len(active_profile["sp_x"])
                    p_def = active_landmarks["prox"] if active_landmarks else active_profile["prox_idx"]
                    d_def = active_landmarks["dist"] if active_landmarks else active_profile["dist_idx"]
                    m_def = active_landmarks["max"] if active_landmarks else active_profile["max_idx"]
                    
                    c_sl1, c_sl2, c_sl3 = st.columns(3)
                    prox_override = c_sl1.slider("Ref Proksymalna", 0, N_pts - 1, int(p_def), key=f"sl_p_{case_key}")
                    dist_override = c_sl2.slider("Ref Dystalna", 0, N_pts - 1, int(d_def), key=f"sl_d_{case_key}")
                    max_override = c_sl3.slider("Max Dilation (Dmax)", 0, N_pts - 1, int(m_def), key=f"sl_m_{case_key}")
                    
                    c_b1, c_b2 = st.columns(2)
                    if c_b1.button("✅ Zastosuj pozycje znaczników", key=f"btn_apply_lm_{case_key}", use_container_width=True):
                        st.session_state[lm_key] = {"prox": prox_override, "dist": dist_override, "max": max_override}
                        st.rerun()
                    if c_b2.button("↩ Przywróć wykryte automatycznie", key=f"btn_reset_lm_{case_key}", use_container_width=True):
                        st.session_state[lm_key] = {"prox": active_profile["prox_idx"], "dist": active_profile["dist_idx"], "max": active_profile["max_idx"]}
                        st.rerun()

        with col_params:
            st.markdown("#### Pomiary i wskaźnik rozszerzenia")
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
                calc_ref_mean = round((calc_ref_prox + calc_ref_dist) / 2.0, 2)
                calc_len = round(float(abs(active_profile["cum_dist_mm"][d_idx] - active_profile["cum_dist_mm"][p_idx])), 2)
                if calc_len < 1.0: calc_len = 5.0
                
                exp_ratio = round(calc_max_diam / calc_ref_mean, 2) if calc_ref_mean > 0 else 1.0
                pct_dil = round((exp_ratio - 1.0) * 100.0, 1)
                
                st.metric("Maksymalna średnica (Dmax):", f"{calc_max_diam} mm")
                st.metric("Referencja uśredniona (Ref):", f"{calc_ref_mean} mm", delta=f"Proks: {calc_ref_prox} | Dyst: {calc_ref_dist}")
                st.metric("Długość naczynia (Length):", f"{calc_len} mm")
                st.markdown(f"""
                <div style='background-color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155; margin-top: 10px;'>
                    <div style='font-size: 13px; color: #94a3b8;'>Ekscentryczność / Wskaźnik poszerzenia:</div>
                    <div style='font-size: 22px; font-weight: 700; color: #38bdf8;'>{exp_ratio}x <span style='font-size: 14px; color: #cbd5e1;'>({pct_dil:+.1f}%)</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Wskaż punkty i wykonaj segmentację, aby wyliczyć wymiary.")

            st.markdown("---")
            st.markdown("#### 🧭 Nawigacja:")
            if matching_pair:
                other_name = matching_pair["p2_name"] if matching_pair["p1_name"] == active_name else matching_pair["p1_name"]
                other_dfp, other_meta = series_map[other_name]
                other_active_fr = st.session_state.get(f"caa_frame_{active_pid}_{os.path.basename(other_dfp)}", 0)
                other_has_mask = bool(st.session_state.get(f"caa_mask_{active_pid}_{os.path.basename(other_dfp)}_{other_active_fr}") is not None)
                
                if not other_has_mask:
                    if st.button(f"➡️ Przejdź do obrysowania drugiej projekcji ({other_meta['series_desc']})", type="primary", use_container_width=True):
                        st.session_state["caa_active_series_name"] = other_name
                        st.session_state["caa_target_view"] = "single_delineation"
                        st.rerun()
                else:
                    if st.button("🌐 Oblicz objętość Simpsona 3D z obu obrysów", type="primary", use_container_width=True):
                        st.session_state["caa_active_biplane_pair"] = matching_pair
                        st.session_state["caa_target_view"] = "biplane_simpson"
                        st.rerun()
                        
            if st.button("📋 Wróć do przeglądu projekcji (Gallery)", use_container_width=True):
                st.session_state["caa_target_view"] = "gallery"
                st.rerun()

def render_catheter_calibration_widget(active_pid, p_name, p_dfp, p_meta, tag="P1"):
    dfp = p_dfp
    d_meta = p_meta
    n_frames = d_meta["total_frames"]
    frame_key = f"caa_frame_{active_pid}_{os.path.basename(dfp)}"
    frame_slider = get_best_or_saved_frame(active_pid, dfp, d_meta)
        
    c_sl1, c_sl2, c_sl3, c_sl4 = st.columns([3.2, 0.6, 0.6, 1.6])
    cal_slider_key = f"sl_frame_cal_{tag}_{active_pid}_{os.path.basename(dfp)}"
    if cal_slider_key in st.session_state and st.session_state[cal_slider_key] != frame_slider:
        st.session_state[cal_slider_key] = frame_slider

    with c_sl1:
        new_frame = st.slider(
            f"🎬 Klatka do kalibracji ({tag}):",
            min_value=0,
            max_value=max(0, n_frames - 1),
            value=frame_slider,
            key=cal_slider_key,
            help="Przesuń suwak, aby wybrać klatkę z najlepiej widocznym cewnikiem"
        )
        if new_frame != frame_slider:
            st.session_state[frame_key] = new_frame
            frame_slider = new_frame
            st.rerun()
            
    with c_sl2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("◀️", key=f"btn_prev_cal_fr_{tag}_{active_pid}_{os.path.basename(dfp)}", help="Poprzednia klatka (-1)"):
            if frame_slider > 0:
                st.session_state[frame_key] = frame_slider - 1
                if cal_slider_key in st.session_state:
                    st.session_state[cal_slider_key] = frame_slider - 1
                st.rerun()
                
    with c_sl3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("▶️", key=f"btn_next_cal_fr_{tag}_{active_pid}_{os.path.basename(dfp)}", help="Następna klatka (+1)"):
            if frame_slider < n_frames - 1:
                st.session_state[frame_key] = frame_slider + 1
                if cal_slider_key in st.session_state:
                    st.session_state[cal_slider_key] = frame_slider + 1
                st.rerun()

    with c_sl4:
        st.markdown(f"**Klatka {frame_slider + 1} / {n_frames}**")
        st.caption(f"Kąty: **{d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°**")
        
    frame_pixels = d_meta["pixels"][frame_slider]
    norm_512 = get_norm_512(frame_pixels)
    norm_rgb = cv2.cvtColor(norm_512, cv2.COLOR_GRAY2RGB)
    dicom_mm_pp = d_meta["spacing"] * (d_meta["pixels"].shape[-1] / 512.0)
    
    case_key = f"{active_pid}_{os.path.basename(dfp)}_{frame_slider}"
    calib_key = f"caa_calib_{active_pid}_{os.path.basename(dfp)}"
    calib_info = st.session_state.get(calib_key, None)
    
    col_vis, col_params = st.columns([1.3, 1])
    with col_vis:
        st.markdown(f"##### 📏 Wskaż 2 punkty na cewniku dla {tag} ({d_meta['series_desc']})")
        calib_bg = norm_rgb.copy()
        if calib_info is not None:
            if "lw" in calib_info and "rw" in calib_info:
                for pt in calib_info["lw"]:
                    cv2.circle(calib_bg, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 0), -1)
                for pt in calib_info["rw"]:
                    cv2.circle(calib_bg, (int(round(pt[0])), int(round(pt[1]))), 2, (0, 255, 0), -1)
            if "click1" in calib_info and "click2" in calib_info:
                p1_i, p2_i = calib_info["click1"], calib_info["click2"]
                cv2.line(calib_bg, p1_i, p2_i, (0, 255, 0), 1)
                cv2.circle(calib_bg, p1_i, 3, (0, 0, 255), -1)
                cv2.circle(calib_bg, p2_i, 3, (0, 0, 255), -1)
                
        calib_canvas_key = f"calib_cv_{tag}_{case_key}_{st.session_state.get('caa_calib_sfx', 0)}"
        c_canvas = st_canvas(
            fill_color="#ff0000",
            stroke_width=0,
            stroke_color="#ff0000",
            background_color="black",
            background_image=Image.fromarray(calib_bg),
            update_streamlit=True,
            height=512,
            width=512,
            drawing_mode="point",
            point_display_radius=2,
            key=calib_canvas_key
        )
        
    with col_params:
        st.markdown(f"##### Parametry cewnika ({tag})")
        CATHETER_SIZES = {
            "6F = 1.98 mm": 1.98,
            "5F = 1.67 mm": 1.67,
            "7F = 2.33 mm": 2.33,
            "4F = 1.35 mm": 1.35,
            "8F = 2.67 mm": 2.67,
            "Własny rozmiar (Custom mm)": None
        }
        cat_choice = st.selectbox(f"Rozmiar cewnika ({tag}):", list(CATHETER_SIZES.keys()), index=0, key=f"caa_cat_{tag}_{case_key}")
        if cat_choice == "Własny rozmiar (Custom mm)":
            catheter_mm = st.number_input(f"Średnica cewnika [mm] ({tag}):", min_value=0.5, max_value=5.0, value=1.98, step=0.05, key=f"caa_cat_mm_{tag}_{case_key}")
        else:
            catheter_mm = CATHETER_SIZES[cat_choice]
            
        if calib_info is not None and (calib_info.get("catheter_mm") != catheter_mm or calib_info.get("catheter_name") != cat_choice):
            calib_info["catheter_name"] = cat_choice
            calib_info["catheter_mm"] = catheter_mm
            calib_info["mm_per_pixel"] = float(catheter_mm / calib_info["avg_diam_px"])
            st.session_state[calib_key] = calib_info
            st.rerun()

        if c_canvas.json_data is not None and "objects" in c_canvas.json_data:
            objs = c_canvas.json_data["objects"]
            if len(objs) >= 2:
                r1 = float(objs[0].get('radius', 2))
                p1 = (float(objs[0].get('left', 0)) + r1, float(objs[0].get('top', 0)) + r1)
                r2 = float(objs[1].get('radius', 2))
                p2 = (float(objs[1].get('left', 0)) + r2, float(objs[1].get('top', 0)) + r2)
                
                eval_sig = (round(p1[0], 1), round(p1[1], 1), round(p2[0], 1), round(p2[1], 1), catheter_mm, cat_choice)
                last_sig = st.session_state.get(f"caa_last_calib_eval_{tag}_{case_key}", None)
                
                if last_sig != eval_sig:
                    st.session_state[f"caa_last_calib_eval_{tag}_{case_key}"] = eval_sig
                    res = calibrate_catheter(norm_512, p1, p2, catheter_mm)
                    if res is not None:
                        res["catheter_name"] = cat_choice
                        res["catheter_mm"] = catheter_mm
                        st.session_state[calib_key] = res
                        st.session_state["caa_calib_sfx"] = st.session_state.get("caa_calib_sfx", 0) + 1
                        st.rerun()
                    else:
                        st.error("⚠️ Nie udało się precyzyjnie wykryć krawędzi cewnika. Wskaż 2 punkty w prostym, widocznym odcinku cewnika.")
                        
        if calib_info is not None:
            st.success(f"✅ Skalibrowano ({calib_info.get('catheter_name', cat_choice)}): **{calib_info['mm_per_pixel']:.4f} mm/px**")
            if st.button(f"🔄 Skasuj kalibrację {tag}", key=f"btn_reset_cal_{tag}_{case_key}", use_container_width=True):
                st.session_state.pop(calib_key, None)
                st.session_state.pop(f"caa_last_calib_eval_{tag}_{case_key}", None)
                st.session_state["caa_calib_sfx"] = st.session_state.get("caa_calib_sfx", 0) + 1
                st.rerun()
        else:
            st.info(f"ℹ️ Domyślna skala DICOM: **{dicom_mm_pp:.4f} mm/px**")
            if st.button(f"⚡ Zaakceptuj skalę DICOM dla {tag}", key=f"btn_acc_dcm_{tag}_{case_key}", use_container_width=True):
                st.session_state[calib_key] = {
                    "mm_per_pixel": dicom_mm_pp,
                    "catheter_name": "DICOM Header",
                    "catheter_mm": 2.0,
                    "avg_diam_px": 2.0 / dicom_mm_pp
                }
                st.rerun()

def render_artery_segmentation_widget(active_pid, p_name, p_dfp, p_meta, tag="P1"):
    dfp = p_dfp
    d_meta = p_meta
    n_frames = d_meta["total_frames"]
    frame_key = f"caa_frame_{active_pid}_{os.path.basename(dfp)}"
    frame_slider = get_best_or_saved_frame(active_pid, dfp, d_meta)
    
    calib_key = f"caa_calib_{active_pid}_{os.path.basename(dfp)}"
    calib_info = st.session_state.get(calib_key)
    dicom_mm_pp = d_meta["spacing"] * (d_meta["pixels"].shape[-1] / 512.0)
    active_mm_pp = calib_info["mm_per_pixel"] if calib_info else dicom_mm_pp

    # Frame selection bar: slider + step buttons + Peak QCA
    c_sl1, c_sl2, c_sl3, c_sl4 = st.columns([3.2, 0.6, 0.6, 1.6])
    seg_slider_key = f"sl_frame_seg_{tag}_{active_pid}_{os.path.basename(dfp)}"
    if seg_slider_key in st.session_state and st.session_state[seg_slider_key] != frame_slider:
        st.session_state[seg_slider_key] = frame_slider

    with c_sl1:
        new_frame = st.slider(
            f"🎬 Przesuń, aby wybrać klatkę do obrysowania ({tag}):",
            min_value=0,
            max_value=max(0, n_frames - 1),
            value=frame_slider,
            key=seg_slider_key,
            help="Przesuń suwak, aby wybrać klatkę z optymalnym wypełnieniem tętnicy kontrastem"
        )
        if new_frame != frame_slider:
            st.session_state[frame_key] = new_frame
            frame_slider = new_frame
            st.rerun()
            
    with c_sl2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("◀️", key=f"btn_prev_fr_{tag}_{active_pid}_{os.path.basename(dfp)}", help="Poprzednia klatka (-1)"):
            if frame_slider > 0:
                st.session_state[frame_key] = frame_slider - 1
                if seg_slider_key in st.session_state:
                    st.session_state[seg_slider_key] = frame_slider - 1
                st.rerun()
                
    with c_sl3:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("▶️", key=f"btn_next_fr_{tag}_{active_pid}_{os.path.basename(dfp)}", help="Następna klatka (+1)"):
            if frame_slider < n_frames - 1:
                st.session_state[frame_key] = frame_slider + 1
                if seg_slider_key in st.session_state:
                    st.session_state[seg_slider_key] = frame_slider + 1
                st.rerun()

    with c_sl4:
        st.markdown(f"**Klatka {frame_slider + 1} / {n_frames}**")
        fsize = os.path.getsize(dfp) if os.path.exists(dfp) else None
        b_ix, _, _ = analyze_series_flow(dfp, fsize)
        if b_ix > 0 and b_ix != frame_slider:
            if st.button(f"🎯 Peak QCA ({b_ix+1})", key=f"btn_peak_{tag}_{active_pid}_{os.path.basename(dfp)}", use_container_width=True, help="Skocz do klatki optymalnego kontrastu"):
                st.session_state[frame_key] = b_ix
                if seg_slider_key in st.session_state:
                    st.session_state[seg_slider_key] = b_ix
                st.rerun()
        else:
            st.caption(f"Kąty: {d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°")
    
    frame_pixels = d_meta["pixels"][frame_slider]
    norm_512 = get_norm_512(frame_pixels)
    norm_rgb = cv2.cvtColor(norm_512, cv2.COLOR_GRAY2RGB)
    
    case_key = f"{active_pid}_{os.path.basename(dfp)}_{frame_slider}"
    mask_key = f"caa_mask_{case_key}"
    prof_key = f"caa_prof_{case_key}"
    lm_key = f"caa_lm_{case_key}"
    
    active_mask = st.session_state.get(mask_key)
    active_profile = st.session_state.get(prof_key)
    active_landmarks = st.session_state.get(lm_key)
    
    col_vis, col_params = st.columns([1.3, 1])
    with col_vis:
        st.markdown(f"##### 🎯 Obrys tętniaka dla {tag} ({d_meta['series_desc']}, Klatka {frame_slider+1})")
        st.caption(f"Skala: **{active_mm_pp:.4f} mm/px** | Kąty: **{d_meta['primary_angle']:+.1f}° / {d_meta['secondary_angle']:+.1f}°**")
        
        annot_bg = norm_rgb.copy()
        if active_mask is not None and np.sum(active_mask) > 0:
            v_mask = (active_mask > 0).astype(np.uint8) * 255
            contour_preview = angioPyFunctions.maskOutliner(labelledArtery=v_mask, outlineThickness=1)
            annot_bg[contour_preview, :] = [0, 255, 0]
            
        canvas_key = f"seg_cv_{tag}_{case_key}_{st.session_state.get('caa_cv_sfx', 0)}"
        annotation_canvas = st_canvas(
            fill_color="#ff0000",
            stroke_width=0,
            stroke_color="#ff0000",
            background_color="black",
            background_image=Image.fromarray(annot_bg),
            update_streamlit=True,
            height=512,
            width=512,
            drawing_mode="point",
            point_display_radius=2,
            key=canvas_key
        )
        
        c_seg1, c_seg2 = st.columns(2)
        with c_seg1:
            if st.button(f"🚀 Segmentuj tętniak {tag} (AI)", type="primary", key=f"btn_seg_{tag}_{case_key}", use_container_width=True):
                if annotation_canvas.json_data is not None and "objects" in annotation_canvas.json_data:
                    objs = annotation_canvas.json_data["objects"]
                    if len(objs) >= 2:
                        with st.spinner(f"Segmentacja naczynia silnikiem angioPy dla {tag}..."):
                            try:
                                pts = []
                                for o in objs:
                                    r = float(o.get('radius', 2))
                                    cy = float(o.get('top', 0)) + r
                                    cx = float(o.get('left', 0)) + r
                                    pts.append([cy, cx])
                                pts = np.array(pts, dtype=np.float32)
                                
                                mask = angioPyFunctions.arterySegmentation(norm_512, pts)
                                mask_bin = (mask > 0).astype(np.uint8)
                                prof = extract_aneurysm_profile(mask_bin, active_mm_pp)
                                if prof is not None:
                                    st.session_state[mask_key] = mask_bin
                                    st.session_state[prof_key] = prof
                                    st.session_state[lm_key] = {
                                        "prox": prof["prox_idx"],
                                        "dist": prof["dist_idx"],
                                        "max": prof["max_idx"]
                                    }
                                    st.success(f"✅ Sukces: Tętniak w {tag} został obrysowany!")
                                    st.rerun()
                                else:
                                    st.error("Nie udało się wyznaczyć osi naczynia. Upewnij się, że punkty leżą wzdłuż światła tętnicy.")
                            except Exception as e:
                                st.error(f"Błąd podczas segmentacji: {e}")
                    else:
                        st.warning("Kliknij co najmniej 2 punkty na naczyniu przed uruchomieniem.")
                else:
                    st.warning("Wskaż punkty na naczyniu na obrazie obok.")
        with c_seg2:
            if st.button(f"🗑️ Wyczyść punkty {tag}", key=f"btn_clr_{tag}_{case_key}", use_container_width=True):
                st.session_state["caa_cv_sfx"] = st.session_state.get("caa_cv_sfx", 0) + 1
                st.rerun()
            if active_mask is not None:
                if st.button(f"🔄 Usuń obrys {tag}", key=f"btn_del_mask_{tag}_{case_key}", use_container_width=True, help="Usuwa obrys tętniaka dla tej klatki"):
                    st.session_state.pop(mask_key, None)
                    st.session_state.pop(prof_key, None)
                    st.session_state.pop(lm_key, None)
                    st.session_state["caa_cv_sfx"] = st.session_state.get("caa_cv_sfx", 0) + 1
                    st.rerun()

    with col_params:
        st.markdown(f"##### Pomiary {tag}")
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
            calc_ref_mean = round((calc_ref_prox + calc_ref_dist) / 2.0, 2)
            calc_len = round(float(abs(active_profile["cum_dist_mm"][d_idx] - active_profile["cum_dist_mm"][p_idx])), 2)
            if calc_len < 1.0: calc_len = 5.0
            exp_ratio = round(calc_max_diam / calc_ref_mean, 2) if calc_ref_mean > 0 else 1.0
            
            st.metric(f"Dmax ({tag}):", f"{calc_max_diam} mm")
            st.metric(f"Ref ({tag}):", f"{calc_ref_mean} mm")
            st.metric(f"Wskaźnik poszerzenia ({tag}):", f"{exp_ratio}x")
            
            # Kaliper overlay preview
            default_dims = {"ref_prox": calc_ref_prox, "ref_dist": calc_ref_dist, "max_diam": calc_max_diam, "mm_pp": active_mm_pp}
            overlay_img = render_aneurysm_overlay(norm_512, active_mask, active_profile, active_landmarks, default_dims=default_dims)
            safe_display_image(overlay_img, caption=f"Obrys i kalipery {tag}")
            
            with st.expander(f"📍 Korekta znaczników {tag}", expanded=False):
                N_pts = len(active_profile["sp_x"])
                prox_override = st.slider("Ref Proksymalna", 0, N_pts - 1, int(p_idx), key=f"sl_p_p_{tag}_{case_key}")
                dist_override = st.slider("Ref Dystalna", 0, N_pts - 1, int(d_idx), key=f"sl_d_p_{tag}_{case_key}")
                max_override = st.slider("Max Dilation (Dmax)", 0, N_pts - 1, int(m_idx), key=f"sl_m_p_{tag}_{case_key}")
                if st.button(f"✅ Zastosuj pozycje {tag}", key=f"btn_apply_lm_p_{tag}_{case_key}", use_container_width=True):
                    st.session_state[lm_key] = {"prox": prox_override, "dist": dist_override, "max": max_override}
                    st.rerun()
        else:
            st.info(f"Wskaż 2 do 4 czerwonych punktów wzdłuż światła naczynia i kliknij 'Segmentuj tętniak {tag} (AI)'.")

def render_biplane_results_content(active_pid, pair, series_map):
    p1_name, p1_dfp, p1_meta = pair["p1_name"], pair["p1_dfp"], pair["p1_meta"]
    p2_name, p2_dfp, p2_meta = pair["p2_name"], pair["p2_dfp"], pair["p2_meta"]
    angle_diff = pair.get("angle_diff", 0.0)
    
    p1_fr = get_best_or_saved_frame(active_pid, p1_dfp, p1_meta)
    p2_fr = get_best_or_saved_frame(active_pid, p2_dfp, p2_meta)
    
    p1_case_key = f"{active_pid}_{os.path.basename(p1_dfp)}_{p1_fr}"
    p2_case_key = f"{active_pid}_{os.path.basename(p2_dfp)}_{p2_fr}"
    
    prof1 = st.session_state.get(f"caa_prof_{p1_case_key}")
    mask1 = st.session_state.get(f"caa_mask_{p1_case_key}")
    lm1 = st.session_state.get(f"caa_lm_{p1_case_key}")
    
    prof2 = st.session_state.get(f"caa_prof_{p2_case_key}")
    mask2 = st.session_state.get(f"caa_mask_{p2_case_key}")
    lm2 = st.session_state.get(f"caa_lm_{p2_case_key}")
    
    is_ready = (prof1 is not None and prof2 is not None and mask1 is not None and mask2 is not None)
    if not is_ready:
        st.warning(f"⚠️ Do wyliczenia objętości metodą Simpsona 3D wymagane jest obrysowanie **obu** projekcji z pary!")
        c_alt1, c_alt2 = st.columns(2)
        with c_alt1:
            if prof1 is None:
                st.error(f"❌ Projekcja 1 ({p1_meta['series_desc']}) nie została jeszcze obrysowana.")
            else:
                st.success(f"✅ Projekcja 1 ({p1_meta['series_desc']}) jest obrysowana.")
        with c_alt2:
            if prof2 is None:
                st.error(f"❌ Projekcja 2 ({p2_meta['series_desc']}) nie została jeszcze obrysowana.")
            else:
                st.success(f"✅ Projekcja 2 ({p2_meta['series_desc']}) jest obrysowana.")
        return False

    simp = compute_biplane_simpsons_volumetry(
        thick1=prof1["thickness_mm"],
        cum_dist1=prof1["cum_dist_mm"],
        prox1=lm1.get("prox", prof1["prox_idx"]) if lm1 else prof1["prox_idx"],
        dist1=lm1.get("dist", prof1["dist_idx"]) if lm1 else prof1["dist_idx"],
        max1=lm1.get("max", prof1["max_idx"]) if lm1 else prof1["max_idx"],
        thick2=prof2["thickness_mm"],
        cum_dist2=prof2["cum_dist_mm"],
        prox2=lm2.get("prox", prof2["prox_idx"]) if lm2 else prof2["prox_idx"],
        dist2=lm2.get("dist", prof2["dist_idx"]) if lm2 else prof2["dist_idx"],
        max2=lm2.get("max", prof2["max_idx"]) if lm2 else prof2["max_idx"],
        n_slices=30
    )
    if simp is None:
        st.error("Wystąpił błąd podczas integracji numerycznej profili.")
        return False

    st.markdown("#### 🔬 Porównanie biplanarne obu obrysowanych projekcji (Side-by-Side):")
    col_ov1, col_ov2 = st.columns(2)
    
    norm_512_p1 = get_norm_512(p1_meta["pixels"][p1_fr])
    ov1 = render_aneurysm_overlay(norm_512_p1, mask1, prof1, lm1)
    
    norm_512_p2 = get_norm_512(p2_meta["pixels"][p2_fr])
    ov2 = render_aneurysm_overlay(norm_512_p2, mask2, prof2, lm2)
    
    with col_ov1:
        st.markdown(f"**Projekcja 1:** `{p1_meta['series_desc']}` (Klatka {p1_fr+1})")
        st.caption(f"Kąty: **{p1_meta['primary_angle']:+.1f}° / {p1_meta['secondary_angle']:+.1f}°** | Skala: **{prof1['mm_per_pixel']:.4f} mm/px**")
        safe_display_image(ov1)
        st.markdown(f"📏 Dmax: **{simp['d1_max']:.1f} mm** | Ref: **{simp['ref1_mean']:.1f} mm** | Ratio: **{simp['dilation_ratio_1']}x** | L1: **{simp['L1']:.1f} mm**")
        
    with col_ov2:
        st.markdown(f"**Projekcja 2:** `{p2_meta['series_desc']}` (Klatka {p2_fr+1})")
        st.caption(f"Kąty: **{p2_meta['primary_angle']:+.1f}° / {p2_meta['secondary_angle']:+.1f}°** | Skala: **{prof2['mm_per_pixel']:.4f} mm/px**")
        safe_display_image(ov2)
        st.markdown(f"📏 Dmax: **{simp['d2_max']:.1f} mm** | Ref: **{simp['ref2_mean']:.1f} mm** | Ratio: **{simp['dilation_ratio_2']}x** | L2: **{simp['L2']:.1f} mm**")
        
    st.markdown("---")
    st.markdown(f"""
    <div style='background-color: #042f2e; border: 1px solid #0f766e; border-left: 6px solid #14b8a6; padding: 16px 20px; border-radius: 8px; margin-bottom: 16px;'>
        <div style='font-size: 15px; font-weight: 700; color: #2dd4bf;'>
            📐 WYNIKI OBJĘTOŚCI 3D SIMPSONA (Z OBU RZECZYWISTYCH OBRYSÓW):
        </div>
        <div style='font-size: 28px; font-weight: 800; color: #5eead4; margin-top: 4px;'>
            {simp['total_vol']:.1f} mm³ <span style='font-size: 16px; font-weight: normal; color: #ccfbf1;'>({simp['total_vol']:.1f} μl)</span>
        </div>
        <div style='font-size: 15px; color: #a7f3d0; margin-top: 6px;'>
            • Nadmiarowa objętość tętniaka (≥ 1.2x Ref): <b>{simp['excess_vol']:.1f} mm³</b> ({simp['excess_vol']:.1f} μl)<br/>
            • Długość właściwego worka tętniaka (≥ 1.2x Ref): <b>{simp.get('aneurysm_len', simp['L']):.1f} mm</b> (cały segment: {simp['L']:.1f} mm)<br/>
            • Objętość zdrowego naczynia referencyjnego: <b>{simp['ref_vol']:.1f} mm³</b><br/>
            • Różnica kątów w przestrzeni: <b>{angle_diff:.1f}°</b> {"✅ (Spełnia warunek ≥ 30°)" if angle_diff>=30 else "⚠️ (< 30°)"}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("3D Eccentricity (Ekscentryczność):", f"{simp['eccentricity']}", help="e = sqrt(1 - (b/a)^2)")
    with c_m2:
        st.metric("Eliptyczność przekroju (a/b):", f"{simp['ellipticity']}")
    with c_m3:
        st.metric("Długość worka (≥1.2x):", f"{simp.get('aneurysm_len', simp['L']):.1f} mm", delta=f"Cały seg: {simp['L']:.1f} mm")
    with c_m4:
        st.metric("Morfologia:", simp["morphology"])

    # ── Interaktywna rekonstrukcja 3D tętniaka ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 🌐 Interaktywna rekonstrukcja 3D tętniaka wieńcowego:")
    st.caption("Trójwymiarowy model światła naczynia zrekonstruowany z obu obrysów (metoda przekrojów eliptycznych Simpsona). **Możesz swobodnie obracać model myszką w 360°, przybliżać kółkiem myszy i badać geometrię worka tętniaka.**")

    c_3d_ctrl0, c_3d_ctrl1, c_3d_ctrl2, c_3d_ctrl3, c_3d_ctrl4 = st.columns([1.3, 1.4, 1.0, 1.0, 1.2])
    with c_3d_ctrl0:
        orient_choice = st.radio(
            "Orientacja naczynia:",
            options=["↕️ Wertykalnie (Pion)", "↔️ Horyzontalnie"],
            index=0,
            horizontal=True,
            key=f"caa_3d_orient_{pair['aha_code']}",
            help="Orientacja wertykalna (pionowa) odpowiada naturalnemu przebiegowi naczynia w pracowni hemodynamicznej (od góry: wlot proksymalny -> w dół: wylot dystalny)"
        )
        orient_param = "vertical" if "Wertykalnie" in orient_choice else "horizontal"
    with c_3d_ctrl1:
        color_mode = st.radio(
            "Kolory naczynia 3D:",
            options=["🎯 Strefy kliniczne (bez gradientu)", "Średnica [mm]", "Rozstrzeń (gradient)"],
            index=0,
            horizontal=True,
            key=f"caa_3d_colormode_{pair['aha_code']}",
            help="Strefy kliniczne: 🟢 Zielony = Zdrowe naczynie (< 1.2x Ref) | 🟡 Żółty = Szyja tętniaka (1.2x - 1.4x Ref) | 🔴 Czerwony = Worek tętniaka (≥ 1.4x Ref)"
        )
        if "Strefy" in color_mode:
            col_param = "zones"
        elif "Średnica" in color_mode:
            col_param = "diameter"
        else:
            col_param = "dilation"
    with c_3d_ctrl2:
        show_ghost_ref = st.checkbox("Pokaż Ref (Ghost)", value=True, key=f"caa_3d_ref_{pair['aha_code']}", help="Półprzezroczysta powłoka pokazująca referencyjny kształt zdrowego naczynia")
    with c_3d_ctrl3:
        show_rings = st.checkbox("Pierścienie kaliperów", value=True, key=f"caa_3d_rings_{pair['aha_code']}", help="Wyświetla pierścienie na poziomie Ref Prox, Dmax i Ref Dist")
    with c_3d_ctrl4:
        stl_data = export_aneurysm_to_stl(simp)
        st.download_button(
            label="📥 Pobierz 3D (STL)",
            data=stl_data,
            file_name=f"aneurysm_3d_{active_pid}_{pair.get('aha_code', 'seg')}.stl",
            mime="application/sla",
            use_container_width=True,
            help="Pobierz plik w formacie STL do druku 3D lub przeglądania w programach CAD / 3D Slicer"
        )

    fig_3d = generate_aneurysm_3d_figure(
        simp=simp,
        pair=pair,
        color_mode=col_param,
        orientation=orient_param,
        show_ref=show_ghost_ref,
        show_rings=show_rings
    )
    if fig_3d is not None:
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📝 Dokumentacja kliniczna i zapis do bazy:")
    c_doc1, c_doc2 = st.columns(2)
    with c_doc1:
        vessel = st.selectbox("Coronary Vessel:", ["LM (Left Main)", "LAD", "LCx", "RCA"], index=1 if "LAD" in pair["aha_label"] else (2 if "LCx" in pair["aha_label"] else 3), key=f"caa_save_vessel_{pair['aha_code']}")
        aha_segment = st.text_input("Segment AHA:", value=pair["aha_label"], key=f"caa_save_aha_{pair['aha_code']}")
    with c_doc2:
        thrombus = st.selectbox("Obecność skrzepliny (Thrombus):", ["Brak (None)", "Obecna (Present)", "Podejrzenie (Suspected)"], key=f"caa_save_thrombus_{pair['aha_code']}")
        calcification = st.selectbox("Zwapnienia ściany (Calcification):", ["Brak (None)", "Łagodne (Mild)", "Masywne (Severe)"], key=f"caa_save_calc_{pair['aha_code']}")

    has_stenosis = st.checkbox("Współistniejące zwężenie w obrębie tętniaka", key=f"caa_save_stenosis_{pair['aha_code']}")
    mld_val = None
    pct_stenosis = 0.0
    if has_stenosis:
        c_st1, c_st2 = st.columns(2)
        with c_st1:
            ref_avg = (simp['ref1_mean'] + simp['ref2_mean']) / 2.0
            mld_val = st.number_input("Minimal Lumen Diameter (MLD) [mm]:", min_value=0.2, max_value=float(ref_avg), value=min(1.5, float(ref_avg)), step=0.1, key=f"caa_mld_{pair['aha_code']}")
        with c_st2:
            pct_stenosis = round((1.0 - (mld_val / ref_avg)) * 100.0, 1)
            st.metric("% Stenosis:", f"{pct_stenosis}%")

    if st.button("💾 Zapisz pełne badanie biplanarne tętniaka do bazy danych", type="primary", use_container_width=True, key=f"btn_save_biplane_{pair['aha_code']}"):
        img_b64 = None
        try:
            h1, w1 = ov1.shape[:2]
            h2, w2 = ov2.shape[:2]
            target_h = 350
            w1_sc = int(w1 * (target_h / h1))
            w2_sc = int(w2 * (target_h / h2))
            ov1_sc = cv2.resize(ov1, (w1_sc, target_h))
            ov2_sc = cv2.resize(ov2, (w2_sc, target_h))
            combined_ov = np.hstack([ov1_sc, ov2_sc])
            pil_thumb = Image.fromarray(combined_ov)
            buf = io.BytesIO()
            pil_thumb.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding thumbnail: {e}")
            
        record = {
            "patient_id": active_pid,
            "vessel": vessel,
            "aha_segment": aha_segment,
            "morphology": simp["morphology"],
            "thrombus": thrombus,
            "calcification": calcification,
            "simpson_total_vol_mm3": simp["total_vol"],
            "simpson_excess_vol_mm3": simp["excess_vol"],
            "simpson_ref_vol_mm3": simp["ref_vol"],
            "simpson_angle_diff_deg": round(angle_diff, 1),
            "eccentricity": simp["eccentricity"],
            "ellipticity": simp["ellipticity"],
            "d1_max_mm": simp["d1_max"],
            "d2_max_mm": simp["d2_max"],
            "ref1_mean_mm": simp["ref1_mean"],
            "ref2_mean_mm": simp["ref2_mean"],
            "dilation_ratio_1": simp["dilation_ratio_1"],
            "dilation_ratio_2": simp["dilation_ratio_2"],
            "length_mm": simp["L"],
            "d1_slices_mm": simp.get("D1_slices", []),
            "d2_slices_mm": simp.get("D2_slices", []),
            "s_slices_mm": simp.get("s_slices", []),
            "p1_series": p1_meta["series_desc"],
            "p1_angles": f"{p1_meta['primary_angle']:+.1f}° / {p1_meta['secondary_angle']:+.1f}°",
            "p2_series": p2_meta["series_desc"],
            "p2_angles": f"{p2_meta['primary_angle']:+.1f}° / {p2_meta['secondary_angle']:+.1f}°",
            "has_concomitant_stenosis": bool(has_stenosis),
            "mld_mm": float(mld_val) if mld_val is not None else None,
            "pct_stenosis": float(pct_stenosis) if has_stenosis else 0.0,
            "thumbnail_b64": img_b64,
            "annotator": st.session_state.user.get("email", "syl.iwanczyk@gmail.com") if "user" in st.session_state else "syl.iwanczyk@gmail.com",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            from firebase_admin import firestore
            db = firestore.client()
            doc_id = f"{active_pid}_{vessel}_{aha_segment.replace(' ', '_')}_{int(time.time())}"
            db.collection("aneurysm_results").document(doc_id).set(record)
            st.success(f"✅ Zapisano pomyślnie badanie tętniaka dla pacjenta **{active_pid}** ({aha_segment}) w kolekcji 'aneurysm_results'!")
        except Exception as e:
            st.error(f"Błąd zapisu do bazy danych: {e}")
            
    return True

# ── 3. WIDOK: OBRYSOWANIE W PARZE (PAIRED DELINEATION WORKFLOW) ──────────────
def render_paired_delineation_view(active_pid, series_map):
    meta_store_key = f"caa_series_meta_{active_pid}"
    series_meta = st.session_state.get(meta_store_key, {})
    detected_pairs = find_biplane_pairs(series_map, series_meta)
    
    pair = st.session_state.get("caa_active_pair")
    if not pair and detected_pairs:
        pair = detected_pairs[0]
        st.session_state["caa_active_pair"] = pair
        
    if not pair:
        st.warning("⚠️ Brak wybranej pary biplanarnej. Oznacz co najmniej dwie projekcje tym samym segmentem AHA w galerii.")
        if st.button("👉 Przejdź do galerii", type="primary"):
            st.session_state["caa_target_view"] = "gallery"
            st.rerun()
        return

    p1_name, p1_dfp, p1_meta = pair["p1_name"], pair["p1_dfp"], pair["p1_meta"]
    p2_name, p2_dfp, p2_meta = pair["p2_name"], pair["p2_dfp"], pair["p2_meta"]
    angle_diff = pair.get("angle_diff", 0.0)

    # Top header
    c_h1, c_h2 = st.columns([3, 1])
    with c_h1:
        st.markdown(f"### 👥 Obrysowanie w parze: **{pair['aha_label']}**")
        st.caption(f"📹 **Projekcja 1:** {p1_meta['series_desc']} ({p1_meta['primary_angle']:+.1f}° / {p1_meta['secondary_angle']:+.1f}°)  |  🌐 **Projekcja 2:** {p2_meta['series_desc']} ({p2_meta['primary_angle']:+.1f}° / {p2_meta['secondary_angle']:+.1f}°)  |  📐 **Kąt 3D:** {angle_diff:.1f}°")
    with c_h2:
        if st.button("🔙 Wróć do galerii", use_container_width=True):
            st.session_state["caa_target_view"] = "gallery"
            st.rerun()

    # Step progress selector
    substep_key = f"caa_pair_substep_{active_pid}"
    if "caa_pair_target_substep" in st.session_state:
        st.session_state[substep_key] = st.session_state.pop("caa_pair_target_substep")
    if substep_key not in st.session_state:
        st.session_state[substep_key] = "1️⃣ Kalibracja cewnika (P1 i P2)"
        
    current_step = st.radio(
        "Kroki procedury w parze:",
        [
            "1️⃣ Kalibracja cewnika (P1 i P2)",
            "2️⃣ Segmentacja naczynia AI (P1 i P2)",
            "3️⃣ Obliczenie 3D Simpsona i Wyniki"
        ],
        horizontal=True,
        key=substep_key
    )
    st.markdown("---")

    # Frame and calibration states
    p1_fr = get_best_or_saved_frame(active_pid, p1_dfp, p1_meta)
    p2_fr = get_best_or_saved_frame(active_pid, p2_dfp, p2_meta)
    
    calib1 = st.session_state.get(f"caa_calib_{active_pid}_{os.path.basename(p1_dfp)}")
    calib2 = st.session_state.get(f"caa_calib_{active_pid}_{os.path.basename(p2_dfp)}")
    
    mask1 = st.session_state.get(f"caa_mask_{active_pid}_{os.path.basename(p1_dfp)}_{p1_fr}")
    mask2 = st.session_state.get(f"caa_mask_{active_pid}_{os.path.basename(p2_dfp)}_{p2_fr}")

    # ── KROK 1: KALIBRACJA CEWNIKA ──
    if current_step == "1️⃣ Kalibracja cewnika (P1 i P2)":
        c_st1, c_st2 = st.columns(2)
        with c_st1:
            if calib1:
                st.success(f"✅ Projekcja 1 ({p1_meta['series_desc']}): Skalibrowana **{calib1['mm_per_pixel']:.4f} mm/px**")
            else:
                st.info(f"⚪ Projekcja 1 ({p1_meta['series_desc']}): Oczekuje na kalibrację")
        with c_st2:
            if calib2:
                st.success(f"✅ Projekcja 2 ({p2_meta['series_desc']}): Skalibrowana **{calib2['mm_per_pixel']:.4f} mm/px**")
            else:
                st.info(f"⚪ Projekcja 2 ({p2_meta['series_desc']}): Oczekuje na kalibrację")

        cal_sel_key = f"caa_pair_cal_sel_{active_pid}"
        target_cal_key = f"caa_pair_target_cal_sel_{active_pid}"
        if target_cal_key in st.session_state:
            st.session_state[cal_sel_key] = st.session_state.pop(target_cal_key)
        elif cal_sel_key not in st.session_state and calib1 and not calib2:
            st.session_state[cal_sel_key] = f"🌐 Projekcja 2 ({p2_meta['series_desc']})"

        cal_sel = st.radio(
            "Wybierz projekcję do kalibracji:",
            [f"📹 Projekcja 1 ({p1_meta['series_desc']})", f"🌐 Projekcja 2 ({p2_meta['series_desc']})"],
            horizontal=True,
            key=cal_sel_key
        )
        
        if "Projekcja 1" in cal_sel:
            render_catheter_calibration_widget(active_pid, p1_name, p1_dfp, p1_meta, tag="P1")
            if calib1 and not calib2:
                if st.button("➡️ Przejdź do kalibracji Projekcji 2", type="primary", use_container_width=True):
                    st.session_state[target_cal_key] = f"🌐 Projekcja 2 ({p2_meta['series_desc']})"
                    st.rerun()
        else:
            render_catheter_calibration_widget(active_pid, p2_name, p2_dfp, p2_meta, tag="P2")
            if calib2 and not calib1:
                if st.button("⬅️ Przejdź do kalibracji Projekcji 1", type="primary", use_container_width=True):
                    st.session_state[target_cal_key] = f"📹 Projekcja 1 ({p1_meta['series_desc']})"
                    st.rerun()

        st.markdown("---")
        if calib1 and calib2:
            st.success("✅ Obie projekcje zostały pomyślnie skalibrowane cewnikiem!")
            if st.button("➡️ Przejdź do Kroku 2: Segmentacja obu projekcji (AI)", type="primary", use_container_width=True):
                st.session_state["caa_pair_target_substep"] = "2️⃣ Segmentacja naczynia AI (P1 i P2)"
                st.rerun()
        else:
            if st.button("⚡ Użyj skali DICOM dla nieskalibrowanych i przejdź do Kroku 2", use_container_width=True):
                st.session_state["caa_pair_target_substep"] = "2️⃣ Segmentacja naczynia AI (P1 i P2)"
                st.rerun()

    # ── KROK 2: SEGMENTACJA NACZYNIA AI ──
    elif current_step == "2️⃣ Segmentacja naczynia AI (P1 i P2)":
        c_st1, c_st2 = st.columns(2)
        with c_st1:
            if mask1 is not None:
                prof1 = st.session_state.get(f"caa_prof_{active_pid}_{os.path.basename(p1_dfp)}_{p1_fr}")
                dmax_t = f"{prof1['max_diam_mm']:.1f} mm" if prof1 else ""
                st.success(f"🟢 Projekcja 1: Obrysowana (Dmax={dmax_t}, Klatka {p1_fr+1})")
            else:
                st.info(f"⚪ Projekcja 1: Oczekuje na obrysowanie (Klatka {p1_fr+1})")
        with c_st2:
            if mask2 is not None:
                prof2 = st.session_state.get(f"caa_prof_{active_pid}_{os.path.basename(p2_dfp)}_{p2_fr}")
                dmax_t = f"{prof2['max_diam_mm']:.1f} mm" if prof2 else ""
                st.success(f"🟢 Projekcja 2: Obrysowana (Dmax={dmax_t}, Klatka {p2_fr+1})")
            else:
                st.info(f"⚪ Projekcja 2: Oczekuje na obrysowanie (Klatka {p2_fr+1})")

        seg_sel_key = f"caa_pair_seg_sel_{active_pid}"
        target_seg_key = f"caa_pair_target_seg_sel_{active_pid}"
        if target_seg_key in st.session_state:
            st.session_state[seg_sel_key] = st.session_state.pop(target_seg_key)
        elif seg_sel_key not in st.session_state and mask1 is not None and mask2 is None:
            st.session_state[seg_sel_key] = f"🌐 Projekcja 2 ({p2_meta['series_desc']})"

        seg_sel = st.radio(
            "Wybierz projekcję do obrysowania:",
            [f"📹 Projekcja 1 ({p1_meta['series_desc']})", f"🌐 Projekcja 2 ({p2_meta['series_desc']})"],
            horizontal=True,
            key=seg_sel_key
        )
        
        if "Projekcja 1" in seg_sel:
            render_artery_segmentation_widget(active_pid, p1_name, p1_dfp, p1_meta, tag="P1")
            if mask1 is not None and mask2 is None:
                if st.button("➡️ Przejdź do obrysowania Projekcji 2", type="primary", use_container_width=True):
                    st.session_state[target_seg_key] = f"🌐 Projekcja 2 ({p2_meta['series_desc']})"
                    st.rerun()
        else:
            render_artery_segmentation_widget(active_pid, p2_name, p2_dfp, p2_meta, tag="P2")
            if mask2 is not None and mask1 is None:
                if st.button("⬅️ Przejdź do obrysowania Projekcji 1", type="primary", use_container_width=True):
                    st.session_state[target_seg_key] = f"📹 Projekcja 1 ({p1_meta['series_desc']})"
                    st.rerun()

        st.markdown("---")
        if mask1 is not None and mask2 is not None:
            st.success("🎉 Obie projekcje zostały pomyślnie obrysowane!")
            if st.button("🚀 Przejdź do Kroku 3: Obliczenie objętości Simpsona 3D i ekscentryczności", type="primary", use_container_width=True):
                st.session_state["caa_pair_target_substep"] = "3️⃣ Obliczenie 3D Simpsona i Wyniki"
                st.rerun()

    # ── KROK 3: OBLICZENIE ──
    elif current_step == "3️⃣ Obliczenie 3D Simpsona i Wyniki":
        res = render_biplane_results_content(active_pid, pair, series_map)
        if not res:
            if st.button("⬅️ Wróć do Kroku 2: Segmentacja obu projekcji", type="primary", use_container_width=True):
                st.session_state["caa_pair_target_substep"] = "2️⃣ Segmentacja naczynia AI (P1 i P2)"
                st.rerun()
        else:
            st.markdown("---")
            if st.button("📋 Zakończ analizę i wróć do galerii", use_container_width=True):
                st.session_state["caa_target_view"] = "gallery"
                st.rerun()

# ── 4. WIDOK: SIMPSON 3D (WYNIKI BIPLANE Z OBU OBRYSÓW) ───────────────────────
def render_biplane_simpson_view(active_pid, series_map):
    pair = st.session_state.get("caa_active_biplane_pair", None)
    meta_store = st.session_state.get(f"caa_series_meta_{active_pid}", {})
    detected_pairs = find_biplane_pairs(series_map, meta_store)
    
    c_top1, c_top2 = st.columns([3, 1])
    with c_top1:
        st.markdown(f"### 🌐 Wyniki Simpsona 3D i Ekscentryczność z obu obrysów")
    with c_top2:
        if st.button("🔙 Wróć do galerii", use_container_width=True):
            st.session_state["caa_target_view"] = "gallery"
            st.rerun()
            
    if detected_pairs:
        pair_labels = [f"Segment: {p['aha_label']} | {p['p1_meta']['series_desc']} & {p['p2_meta']['series_desc']} (Kąt 3D: {p['angle_diff']:.1f}°)" for p in detected_pairs]
        cur_p_idx = 0
        if pair:
            for i, p in enumerate(detected_pairs):
                if p["p1_name"] == pair["p1_name"] and p["p2_name"] == pair["p2_name"]:
                    cur_p_idx = i
                    break
        chosen_pair_idx = st.selectbox("Wybierz parę biplanarną do obliczeń:", range(len(detected_pairs)), format_func=lambda i: pair_labels[i], index=cur_p_idx)
        pair = detected_pairs[chosen_pair_idx]
        st.session_state["caa_active_biplane_pair"] = pair
    elif pair is None:
        st.warning("Brak wybranej pary biplanarnej. Wróć do galerii i oznacz dwie projekcje tym samym segmentem AHA.")
        if st.button("👉 Przejdź do galerii", type="primary"):
            st.session_state["caa_target_view"] = "gallery"
            st.rerun()
        return

    render_biplane_results_content(active_pid, pair, series_map)

# ── 4. GŁÓWNY PUNKT WEJŚCIA MODUŁU TĘTNIAKÓW ─────────────────────────────────
def render_coronary_aneurysm_workspace():
    st.markdown("<h1 style='color: #38bdf8; font-family: Outfit, sans-serif;'>🩺 Coronary Artery Aneurysm (CAA) Workspace</h1>", unsafe_allow_html=True)
    st.markdown("Platforma do analizy tętniaków wieńcowych: przegląd wszystkich projekcji, oznaczanie segmentów AHA, kalibracja cewnika auto-QCA, segmentacja AI oraz wyliczanie objętości 3D Simpsona i ekscentryczności z obu obrysów.")
    
    ensure_storage_dir()
    
    # Sidebar: patient case library
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Baza pacjentów (Private Library)")
    patients = get_aneurysm_patients()
    
    if "aneurysm_patient_id" not in st.session_state:
        st.session_state.aneurysm_patient_id = patients[0] if patients else ""
        
    selected_pat = st.sidebar.selectbox(
        "Wybierz badanie pacjenta:",
        options=["-- Wybierz pacjenta --"] + patients if patients else ["(Brak badań)"],
        index=(patients.index(st.session_state.aneurysm_patient_id) + 1) if (patients and st.session_state.aneurysm_patient_id in patients) else 0,
        key="aneurysm_patient_select"
    )
    if selected_pat and selected_pat not in ["-- Wybierz pacjenta --", "(Brak badań)"]:
        st.session_state.aneurysm_patient_id = selected_pat
        
    with st.sidebar.expander("📤 Wgraj nowe badanie (Archiwum ZIP)", expanded=(len(patients) == 0)):
        st.markdown("Wgraj spakowane badanie koronarografii (ZIP). Wszystkie projekcje DICOM zostaną rozpakowane i zachowane na serwerze.")
        uploaded_zip = st.file_uploader("Wybierz plik ZIP:", type=["zip"], key="aneurysm_zip_uploader")
        custom_patient_name = st.text_input("Identyfikator pacjenta / ID badania:", value=os.path.splitext(uploaded_zip.name)[0] if uploaded_zip else "", key="aneurysm_custom_pid_input")
        
        if uploaded_zip and st.button("🚀 Wgraj i zapisz badanie", key="btn_save_aneurysm_zip", use_container_width=True):
            pid_clean = sanitize_folder_name(custom_patient_name if custom_patient_name.strip() else uploaded_zip.name)
            target_patient_dir = os.path.join(ANEURYSM_STORAGE_DIR, pid_clean)
            os.makedirs(target_patient_dir, exist_ok=True)
            
            with st.spinner(f"Rozpakowywanie badania '{pid_clean}'..."):
                try:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zf:
                        valid_members = [m for m in zf.infolist() if not m.filename.startswith('__MACOSX') and not os.path.basename(m.filename).startswith('.')]
                        for m in valid_members:
                            zf.extract(m, target_patient_dir)
                    st.success(f"✅ Pomyślnie wgrano badanie: **{pid_clean}**")
                    st.session_state.aneurysm_patient_id = pid_clean
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Błąd rozpakowywania archiwum ZIP: {e}")
                    
    active_pid = st.session_state.get("aneurysm_patient_id")
    if not active_pid or active_pid in ["-- Wybierz pacjenta --", "(Brak badań)"]:
        st.info("👋 Witamy w module analizy tętniaków! Proszę wgrać badanie ZIP lub wybrać pacjenta z lewego panelu, aby rozpocząć analizę.")
        return

    patient_dir = os.path.join(ANEURYSM_STORAGE_DIR, active_pid)
    if not os.path.exists(patient_dir):
        st.warning(f"Katalog pacjenta '{active_pid}' nie został odnaleziony.")
        return

    # Find all DICOM files recursively
    dicom_files = []
    for root, _, files in os.walk(patient_dir):
        for f in files:
            if f.startswith('.'): continue
            fp = os.path.join(root, f)
            if f.lower().endswith('.dcm') or os.path.getsize(fp) > 132:
                dicom_files.append(fp)

    if not dicom_files:
        st.warning(f"Nie odnaleziono poprawnych plików DICOM w folderze pacjenta '{active_pid}'.")
        return

    # Load all series headers
    series_map = {}
    for idx, dfp in enumerate(dicom_files):
        d_meta = load_dicom_file(dfp)
        if d_meta is not None:
            lbl = f"Series #{len(series_map)+1}: {d_meta['series_desc']} (LAO/RAO: {d_meta['primary_angle']:+.1f}°, CRA/CAU: {d_meta['secondary_angle']:+.1f}°, {d_meta['total_frames']} frames)"
            series_map[lbl] = (dfp, d_meta)

    if not series_map:
        st.error("Nie udało się odczytać nagłówków DICOM dla tego pacjenta.")
        return

    # Top-level view router
    view_key = f"caa_view_mode_{active_pid}"

    if "caa_target_view" in st.session_state:
        st.session_state[view_key] = st.session_state.pop("caa_target_view")
    if view_key not in st.session_state:
        st.session_state[view_key] = "gallery"

    # Persistent Top Navigation Bar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Widok roboczy:")
    nav_options = ["🗂️ Przegląd projekcji (Gallery)", "👥 Obrysowanie w parze (Biplane)", "🎯 Delineacja pojedyncza", "🌐 Simpson 3D (Biplane)"]
    current_idx = 0
    if st.session_state[view_key] == "paired_delineation":
        current_idx = 1
    elif st.session_state[view_key] == "single_delineation":
        current_idx = 2
    elif st.session_state[view_key] == "biplane_simpson":
        current_idx = 3

    view_choice = st.sidebar.radio(
        "Wybierz widok:",
        nav_options,
        index=current_idx,
        key=f"rad_view_{active_pid}"
    )
    
    if st.sidebar.button("🔄 Zresetuj wybory pacjenta", key=f"btn_sb_reset_{active_pid}", use_container_width=True):
        reset_patient_workspace(active_pid)
        st.rerun()
    if view_choice == "🗂️ Przegląd projekcji (Gallery)" and st.session_state[view_key] != "gallery":
        st.session_state[view_key] = "gallery"
        st.rerun()
    elif view_choice == "👥 Obrysowanie w parze (Biplane)" and st.session_state[view_key] != "paired_delineation":
        st.session_state[view_key] = "paired_delineation"
        st.rerun()
    elif view_choice == "🎯 Delineacja pojedyncza" and st.session_state[view_key] != "single_delineation":
        st.session_state[view_key] = "single_delineation"
        st.rerun()
    elif view_choice == "🌐 Simpson 3D (Biplane)" and st.session_state[view_key] != "biplane_simpson":
        st.session_state[view_key] = "biplane_simpson"
        st.rerun()

    # Route to active view
    if st.session_state[view_key] == "gallery":
        render_projections_gallery(active_pid, series_map)
    elif st.session_state[view_key] == "paired_delineation":
        render_paired_delineation_view(active_pid, series_map)
    elif st.session_state[view_key] == "single_delineation":
        render_single_delineation_view(active_pid, series_map)
    elif st.session_state[view_key] == "biplane_simpson":
        render_biplane_simpson_view(active_pid, series_map)

    # Saved records history
    st.markdown("---")
    st.markdown("#### 📋 Zapisane badania tętniaków dla pacjenta:")
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
                    "Max D1 [mm]": dt.get("d1_max_mm", dt.get("max_aneurysm_diam_mm")),
                    "Max D2 [mm]": dt.get("d2_max_mm", "—"),
                    "Ekscentryczność": dt.get("eccentricity", "—"),
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
                        st.image(img_bytes, caption=f"Zapisany obrys tętniaka: {saved_dt.get('vessel')} {saved_dt.get('aha_segment')} (Simpson: {saved_dt.get('simpson_total_vol_mm3')} mm³, Ekscentryczność: {saved_dt.get('eccentricity')})", use_column_width=True)
                    else:
                        st.info("Dla tego rekordu brak zapisanego zrzutu obrysu.")
        else:
            st.info("Brak zapisanych oznaczeń dla tego pacjenta. Wybierz i obrysuj projekcje powyżej.")
    except Exception as e:
        print(f"Error fetching saved aneurysms: {e}")
