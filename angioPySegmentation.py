import os
import os.path
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import pydicom
import glob
import mpld3
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import tifffile
from streamlit_plotly_events import plotly_events
from PIL import Image

# Monkey-patch for streamlit-drawable-canvas 0.9.3 compatibility with Streamlit >= 1.28
# Old signature: image_to_url(image, width:int, clamp, channels, output_format, image_id)
# New signature: image_to_url(image, layout_config:LayoutConfig, clamp, channels, output_format, image_id)
import streamlit.elements.image as _st_image
if not hasattr(_st_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _new_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig as _LayoutConfig
    def _image_to_url(image, width, clamp, channels, output_format, image_id):
        layout = _LayoutConfig(width=width if isinstance(width, int) else None)
        return _new_image_to_url(image, layout, clamp, channels, output_format, image_id)
    _st_image.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas
import predict
import angioPyFunctions
import scipy
import scipy.signal
import scipy.ndimage
import cv2
import io
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AngioPy Segmentation", layout="wide")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

@st.cache_data
def selectSlice(slice_ix, pixelArray, fileName):
    tifffile.imwrite(f"{outputPath}/{fileName}", pixelArray[slice_ix, :, :])
    st.session_state.btnSelectSlice = True

# ── View Routing State ─────────────────────────────────────────────────────────
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'grid'
if 'selected_dicom' not in st.session_state:
    st.session_state.selected_dicom = None
if 'best_frame_ix' not in st.session_state:
    st.session_state.best_frame_ix = 0
if 'dicom_registry' not in st.session_state:
    DicomFolder = "Dicom/"
    registry = {}
    for file in sorted(glob.glob(DicomFolder + "/*")):
        registry[os.path.basename(file)] = file
    st.session_state.dicom_registry = registry

if 'patient_id' not in st.session_state:
    st.session_state.patient_id = ""
if 'patient_cart' not in st.session_state:
    st.session_state.patient_cart = []
if 'dicom_metadata' not in st.session_state:
    st.session_state.dicom_metadata = {}

# ── AHA Vessel Segment Hierarchy ──────────────────────────────────────────────
AHA_VESSEL_SEGMENTS = {
    "RCA – Prawa tętnica wieńcowa": {
        "key": "RCA",
        "segments": [
            ("1", "Seg 1 – Proximal RCA"),
            ("2", "Seg 2 – Mid RCA"),
            ("3", "Seg 3 – Distal RCA"),
            ("4", "Seg 4 – PDA (Posterior Descending)"),
            ("14R", "Seg 14R – PLV (Posterior Left Ventricular)"),
        ]
    },
    "LM & LAD – Pień i Gałąź przednia": {
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
    "LCx – Gałąź okalająca": {
        "key": "CX",
        "segments": [
            ("11", "Seg 11 – Proximal LCx"),
            ("12", "Seg 12 – OM1 (First Obtuse Marginal)"),
            ("12a", "Seg 12a – OM2 (Second Obtuse Marginal)"),
            ("13", "Seg 13 – Distal LCx"),
            ("14L", "Seg 14L – PL (Posterolateral Branch)"),
            ("15", "Seg 15 – PDA (dominacja lewej)"),
        ]
    },
}

# Helper: map system key → segment label list
def _seg_labels(system_name):
    return [s[1] for s in AHA_VESSEL_SEGMENTS[system_name]["segments"]]

def _seg_codes(system_name):
    return [s[0] for s in AHA_VESSEL_SEGMENTS[system_name]["segments"]]

def _system_from_aha_key(key):
    """Return system name matching colourTableList key (LAD/CX/RCA)."""
    for sname, sdata in AHA_VESSEL_SEGMENTS.items():
        if sdata["key"] == key:
            return sname
    return list(AHA_VESSEL_SEGMENTS.keys())[1]  # default LAD system

def _colour_key_from_system(system_name):
    return AHA_VESSEL_SEGMENTS[system_name]["key"]

ALL_SYSTEM_NAMES = list(AHA_VESSEL_SEGMENTS.keys())

@st.cache_data
def analyze_series_flow(dicom_path, file_size=None):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pixelArray = dcm.pixel_array
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        if len(pixelArray.shape) == 2:
            pixelArray = numpy.expand_dims(pixelArray, axis=0)
        n_slices = pixelArray.shape[0]
        
        if n_slices == 1:
            return 0, 0, 0, 0, 3, "Single image, TIMI 3 assumed."
            
        scores = []
        for i in range(n_slices):
            frame = pixelArray[i]
            small = cv2.resize(frame, (256, 256))
            blurred = scipy.ndimage.gaussian_filter(small.astype(float), 2)
            grad = numpy.abs(numpy.gradient(blurred))
            scores.append(numpy.sum(grad))
            
        scores = numpy.array(scores)
        best_ix = int(numpy.argmax(scores))
        end_ix = n_slices - 1
        
        baseline = numpy.mean(scores[:3]) if len(scores) > 3 else scores[0]
        threshold = baseline + numpy.std(scores) * 1.5
        
        start_ix = 0
        for i in range(n_slices):
            if scores[i] > threshold:
                start_ix = i
                break
                
        if start_ix >= best_ix:
            start_ix = max(0, best_ix - 15)
            
        tfc = best_ix - start_ix
        
        if tfc <= 3:
            timi = 0
            justification = "No significant contrast progression detected."
        elif tfc > 45:
            timi = 1
            justification = f"Very slow penetration (TFC={tfc}). Fails to opacify."
        elif tfc > 25:
            timi = 2
            justification = f"Sluggish perfusion (TFC={tfc})."
        else:
            timi = 3
            justification = f"Brisk flow (TFC={tfc})."
            
        return best_ix, start_ix, end_ix, tfc, timi, justification
    except Exception as e:
        return 0, 0, 0, 0, 0, "Error during analysis calculation."

@st.cache_data
def detect_contrast(dicom_path, file_size=None):
    """Return True if the series likely contains angiographic contrast."""
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pa = dcm.pixel_array
        if len(pa.shape) == 4: pa = pa[:, :, :, 0]
        if len(pa.shape) == 2: pa = numpy.expand_dims(pa, axis=0)
        n = pa.shape[0]
        if n < 2:
            # Single frame — check gradient variance heuristic
            frame = pa[0].astype(numpy.float32)
            small = cv2.resize(frame, (128, 128))
            gx = numpy.diff(small, axis=1)
            gy = numpy.diff(small, axis=0)
            grad_var = float(numpy.var(gx)) + float(numpy.var(gy))
            return grad_var > 500
        # Multi-frame: compare temporal variance across frames
        sample_idxs = numpy.linspace(0, n - 1, min(n, 10), dtype=int)
        frames = [cv2.resize(pa[i].astype(numpy.float32), (64, 64)) for i in sample_idxs]
        stack = numpy.stack(frames, axis=0)
        temporal_var = float(numpy.var(stack, axis=0).mean())
        # Also check gradient of brightest frame
        best = frames[int(numpy.argmax([f.max() for f in frames]))]
        gx = numpy.diff(best, axis=1)
        gy = numpy.diff(best, axis=0)
        grad_var = float(numpy.var(gx)) + float(numpy.var(gy))
        return temporal_var > 150 or grad_var > 500
    except Exception:
        return True  # assume contrast if we can't tell



# ── GRID MODE ─────────────────────────────────────────────────────────────────
if st.session_state.current_view == 'grid':
    st.markdown("<h1 style='text-align: center;'>AngioPy Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Welcome to <b>AngioPy Segmentation</b>, an AI-driven, coronary angiography segmentation tool.</h5>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>DICOM Series Selection Grid</h2>", unsafe_allow_html=True)
    st.write("Automatically extracted optimal contrast frames from all available DICOMs.")
    
    uploadedDicoms = st.sidebar.file_uploader("Upload DICOM file(s)", key="gridDicomUploader", accept_multiple_files=True)
    if uploadedDicoms:
        import tempfile
        if "uploaded_dicom_names" not in st.session_state:
            st.session_state.uploaded_dicom_names = set()
        new_files = 0
        for uploadedDicom in uploadedDicoms:
            if uploadedDicom.name not in st.session_state.uploaded_dicom_names:
                tmpPath = os.path.join(tempfile.gettempdir(), uploadedDicom.name)
                with open(tmpPath, "wb") as f:
                    f.write(uploadedDicom.getbuffer())
                st.session_state.dicom_registry[uploadedDicom.name] = tmpPath
                st.session_state.uploaded_dicom_names.add(uploadedDicom.name)
                new_files += 1
        if new_files:
            st.sidebar.success(f"✅ Dodano {new_files} plik(ów). Łącznie: {len(st.session_state.dicom_registry)}")
    if "uploaded_dicom_names" in st.session_state and st.session_state.uploaded_dicom_names:
        st.sidebar.caption(f"📂 Załadowanych: {len(st.session_state.dicom_registry)} plików")
        if st.sidebar.button("🗑️ Usuń wgrane pliki", key="clear_uploads_btn"):
            for name in list(st.session_state.uploaded_dicom_names):
                st.session_state.dicom_registry.pop(name, None)
            st.session_state.uploaded_dicom_names.clear()
            st.rerun()

    if st.sidebar.button("🔃 Sortuj sekwencje", use_container_width=True, key="sidebar_sort_btn"):
        st.session_state._sidebar_sort_requested = True




    if not st.session_state.dicom_registry:
        st.warning("No DICOM files found in Dicom/ or uploaded.")
        st.stop()
        
    st.session_state.patient_id = st.text_input("Patient ID Number", st.session_state.patient_id)
    
    # Init metadata and sort by time
    temp_sort_list = []
    for name, path in st.session_state.dicom_registry.items():
        if name not in st.session_state.dicom_metadata:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                date_val = getattr(ds, "AcquisitionDate", getattr(ds, "SeriesDate", getattr(ds, "StudyDate", "19700101")))
                time_val = getattr(ds, "AcquisitionTime", getattr(ds, "SeriesTime", getattr(ds, "ContentTime", "999999")))
                dt_str = str(date_val) + str(time_val)
            except:
                dt_str = "99999999999999"
            st.session_state.dicom_metadata[name] = {
                "time": dt_str,
                "phase": "PRE-PCI",
                "vessel": None,
                "vessel_system": None,
                "aha": None,
                "aha_label": None,
                "ffr_registered": "N/A",
                "chosen_for_analysis": False,
                "vessel_explicitly_set": False,
            }
        temp_sort_list.append((name, path, st.session_state.dicom_metadata[name]["time"]))
        
    # ── Phase auto-tagging (chronological, first load only) ────────────────────
    temp_sort_list.sort(key=lambda x: x[2])
    if len(temp_sort_list) >= 2:
        mid = len(temp_sort_list) // 2
        for ix, (name, path, t) in enumerate(temp_sort_list):
            if "auto_tagged" not in st.session_state.dicom_metadata[name]:
                st.session_state.dicom_metadata[name]["phase"] = "PRE-PCI" if ix < mid else "POST-PCI"
                st.session_state.dicom_metadata[name]["auto_tagged"] = True

    # ── Sorting helpers (used by Sort button below) ────────────────────────────
    from collections import defaultdict

    def _build_sorted_order(base_list):
        vessel_groups = defaultdict(list)
        for name, path, t in base_list:
            m = st.session_state.dicom_metadata[name]
            if m.get("vessel_explicitly_set", False):
                seg_key = (m.get("vessel_system", ""), m.get("aha", ""))
            else:
                seg_key = ("__individual__", name)
            vessel_groups[seg_key].append((name, path, t))

        def sort_group(items):
            pre   = sorted([i for i in items if st.session_state.dicom_metadata[i[0]].get("phase") == "PRE-PCI"],  key=lambda x: x[2])
            post  = sorted([i for i in items if st.session_state.dicom_metadata[i[0]].get("phase") == "POST-PCI"], key=lambda x: x[2])
            other = sorted([i for i in items if st.session_state.dicom_metadata[i[0]].get("phase") not in ("PRE-PCI", "POST-PCI")], key=lambda x: x[2])
            return pre + post + other

        def group_earliest_time(items):
            return min(i[2] for i in items)

        def group_is_chosen(items):
            return any(st.session_state.dicom_metadata[i[0]].get("chosen_for_analysis", False) for i in items)

        sorted_groups = sorted(
            vessel_groups.values(),
            key=lambda items: (
                0 if group_is_chosen(items) else 1,
                group_earliest_time(items)
            )
        )
        result = []
        for group in sorted_groups:
            result.extend(sort_group(group))
        return result

    # ── Use persisted grid_order if available, else raw chronological ──────────
    if "grid_order" not in st.session_state:
        st.session_state.grid_order = None

    # Handle sidebar sort button request (fired before _build_sorted_order existed)
    if st.session_state.get("_sidebar_sort_requested", False):
        st.session_state.grid_order = _build_sorted_order(temp_sort_list)
        st.session_state._sidebar_sort_requested = False
        st.rerun()

    display_list = st.session_state.grid_order if st.session_state.grid_order is not None else temp_sort_list
    # keep display_list in sync with registry (handle newly added DICOMs)
    display_names = {n for n, _, _ in display_list}
    for item in temp_sort_list:
        if item[0] not in display_names:
            display_list = list(display_list) + [item]

                
    if len(st.session_state.patient_cart) > 0:
        st.markdown("---")
        st.success(f"🛒 **{len(st.session_state.patient_cart)} sequence(s) analyzed** and saved to the Patient Master Report Cart.")
        import io
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        masterPdfBuf = io.BytesIO()
        with PdfPages(masterPdfBuf) as pdf:
            for itm in st.session_state.patient_cart:
                fig_pdf = Figure(figsize=(8.5, 11))
                pid = st.session_state.patient_id if st.session_state.patient_id else "UNKNOWN"
                fig_pdf.text(0.5, 0.95, f"Master Clinical Report - Patient: {pid}", ha='center', fontsize=20, weight='bold')
                fig_pdf.text(0.5, 0.92, f"Phase: {itm['phase']}  |  {itm.get('vessel_system', itm['vessel'])} – {itm.get('aha_label', 'AHA '+itm['aha'])}  |  DICOM: {itm['dicom_name']}", ha='center', fontsize=11)
                fig_pdf.text(0.1, 0.85, f"TIMI Flow Scale: Grade {itm['metrics']['timi']}", fontsize=14, weight='bold', color='darkred')
                fig_pdf.text(0.1, 0.82, f"TFC (TIMI Frame Count): {itm['metrics']['tfc']}", fontsize=12)
                fig_pdf.text(0.1, 0.79, f"Justification: {itm['metrics']['just']}", fontsize=11, style='italic')
                if itm['phase'] == 'PRE-PCI':
                    lbl = ">50% distal to FFR"
                    ffr_txt = f"FFR Registered: {itm.get('ffr_registered', 'N/A')}  |  "
                else:
                    lbl = ">50% distal to DES/DCB"
                    ffr_txt = ""
                fig_pdf.text(0.1, 0.76, f"{ffr_txt}{lbl}: {itm.get('other_lesion_distal', 'No')}", fontsize=12, weight='bold')
                fig_pdf.text(0.1, 0.72, "QCA Metrics Summary", fontsize=14, weight='bold')
                str_dist_m = "N/A" if itm['metrics']['dist'] == "N/A" else f"{itm['metrics']['dist']:.2f} mm"
                str_ref_m  = "N/A" if itm['metrics']['ref']  == "N/A" else f"{itm['metrics']['ref']:.2f} mm"
                str_mld_m  = "N/A" if itm['metrics']['mld']  == "N/A" else f"{itm['metrics']['mld']:.2f} mm"
                str_pctD_m = "N/A" if itm['metrics']['pct_diam'] == "N/A" else f"{itm['metrics']['pct_diam']:.1f} %"
                str_pctA_m = "N/A" if itm['metrics']['pct_area'] == "N/A" else f"{itm['metrics']['pct_area']:.1f} %"
                str_len_m  = "N/A" if itm['metrics']['lesion_len'] == "N/A" else f"{itm['metrics']['lesion_len']:.2f} mm"
                _post    = itm['metrics'].get('is_post_pci', itm['phase'] == 'POST-PCI')
                prox_lbl = "Proximal Edge Diameter" if _post else "Max Proximal Reference"
                dist_lbl = "Distal Edge Diameter  " if _post else "Max Distal Reference  "
                len_lbl  = "Stent Length          " if _post else "Lesion Length         "
                m_text = (
                    f"{prox_lbl}:  {itm['metrics']['prox']:.2f} mm\n\n"
                    f"{dist_lbl}:  {str_dist_m}\n\n"
                    f"Calculated Reference:    {str_ref_m}\n\n"
                    f"Minimum Lumen Diameter:  {str_mld_m}\n\n"
                    f"% Diameter Stenosis:     {str_pctD_m}\n\n"
                    f"% Area Stenosis:         {str_pctA_m}\n\n"
                    f"{len_lbl}:  {str_len_m}"
                )
                fig_pdf.text(0.1, 0.68, m_text, fontsize=11, family='monospace', va='top')

                ax = fig_pdf.add_axes([0.1, 0.05, 0.8, 0.45])
                ax.imshow(itm['image'])
                ax.axis('off')
                pdf.savefig(fig_pdf)
        st.download_button("📄 Export Master Patient PDF", data=masterPdfBuf.getvalue(), file_name=f"Patient_{st.session_state.patient_id if st.session_state.patient_id else 'Report'}_Master.pdf", mime="application/pdf", use_container_width=True)
        st.markdown("---")

    cols = st.columns(1)
    idx = 0
    for key_idx, (name, path, _) in enumerate(display_list):
        with cols[0]:
            meta = st.session_state.dicom_metadata[name]

            # ── Frame analysis (best frame detection, no contrast filtering) ──
            best_ix, start_ix, end_ix, tfc, timi, just = analyze_series_flow(path, os.path.getsize(path))
            _card_has_contrast = True  # show all series regardless of contrast detection

            chosen_badge = "⭐ " if meta.get("chosen_for_analysis", False) else ""
            with st.expander(f"{chosen_badge}{name}", expanded=True):
                if True:
                    # ── Chosen for Analysis checkbox ──────────────────────────
                    prev_chosen = meta.get("chosen_for_analysis", False)
                    meta["chosen_for_analysis"] = st.checkbox(
                        "⭐ Chosen for Analysis",
                        value=prev_chosen,
                        key=f"chosen_{name}"
                    )
                    if prev_chosen and not meta["chosen_for_analysis"]:
                        st.session_state.grid_order = None

                    # ── Phase selection ────────────────────────────────────────
                    meta["phase"] = st.selectbox(
                        "Procedure Phase", ["PRE-PCI", "POST-PCI"],
                        index=0 if meta["phase"] == "PRE-PCI" else 1,
                        key=f"phase_{name}"
                    )

                    # ── Vessel System + Segment (no labels, inline) ───────────
                    PLACEHOLDER_SYSTEM = "— Wybierz naczynie —"
                    SYSTEM_OPTIONS = [PLACEHOLDER_SYSTEM] + ALL_SYSTEM_NAMES

                    vcol1, vcol2 = st.columns(2)
                    cur_system = meta.get("vessel_system") or PLACEHOLDER_SYSTEM
                    if cur_system not in SYSTEM_OPTIONS:
                        cur_system = PLACEHOLDER_SYSTEM
                    prev_system = cur_system

                    chosen_system = vcol1.selectbox(
                        "", SYSTEM_OPTIONS,
                        index=SYSTEM_OPTIONS.index(cur_system),
                        key=f"vessel_system_{name}",
                        label_visibility="collapsed"
                    )

                    if chosen_system == PLACEHOLDER_SYSTEM:
                        meta["vessel_system"] = None
                        meta["vessel"] = None
                        meta["aha"] = None
                        meta["aha_label"] = None
                        meta["vessel_explicitly_set"] = False
                        vcol2.warning("← Wybierz naczynie")
                    else:
                        if chosen_system != prev_system:
                            meta["vessel_explicitly_set"] = True
                        meta["vessel_system"] = chosen_system
                        meta["vessel"] = _colour_key_from_system(chosen_system)

                        seg_labels = _seg_labels(chosen_system)
                        seg_codes  = _seg_codes(chosen_system)
                        cur_aha_label = meta.get("aha_label") or seg_labels[0]
                        if cur_aha_label not in seg_labels:
                            cur_aha_label = seg_labels[0]
                        prev_aha_label = cur_aha_label
                        chosen_seg_label = vcol2.selectbox(
                            "", seg_labels,
                            index=seg_labels.index(cur_aha_label),
                            key=f"seg_{name}",
                            label_visibility="collapsed"
                        )
                        if chosen_seg_label != prev_aha_label:
                            meta["vessel_explicitly_set"] = True
                        meta["aha_label"] = chosen_seg_label
                        meta["aha"] = seg_codes[seg_labels.index(chosen_seg_label)]

                    if not meta.get("vessel_explicitly_set", False):
                        st.caption("⚠️ Nie wybrano naczynia — sekwencja nie zostanie zgrupowana")

                    # ── FFR registered ─────────────────────────────────────────
                    cur_ffr = meta.get("ffr_registered", "N/A")
                    if cur_ffr not in ["Yes", "No", "N/A"]: cur_ffr = "N/A"
                    meta["ffr_registered"] = st.radio(
                        "FFR Wire Registered?",
                        ["Yes", "No", "N/A"],
                        index=["Yes", "No", "N/A"].index(cur_ffr),
                        horizontal=True,
                        key=f"ffr_{name}"
                    )

                    st.session_state.dicom_metadata[name] = meta

            with st.spinner("Analyzing flow..."):
                # best_ix, timi etc. already computed above — reuse
                has_contrast = _card_has_contrast

                try:
                    dcm = pydicom.dcmread(path, force=True)
                    pa = dcm.pixel_array
                    if len(pa.shape) == 4: pa = pa[:, :, :, 0]
                    if len(pa.shape) == 2: pa = numpy.expand_dims(pa, axis=0)

                    frame_start = pa[start_ix]
                    frame_best  = pa[best_ix]
                    frame_end   = pa[end_ix]
                    img_start = cv2.resize(frame_start, (512, 512))
                    img_best  = cv2.resize(frame_best,  (512, 512))
                    img_end   = cv2.resize(frame_end,   (512, 512))
                    combined  = numpy.concatenate((img_start, img_best, img_end), axis=1)

                    if st.session_state.get(f"play_{name}", False):
                        import tempfile
                        tmp_gif_path = os.path.join(tempfile.gettempdir(), f"angio_{name}_{os.path.getsize(path)}.gif")
                        if not os.path.exists(tmp_gif_path):
                            pil_frames = []
                            for f_idx in range(pa.shape[0]):
                                fr = pa[f_idx].astype(numpy.float32)
                                fr = (fr - numpy.min(fr)) / (numpy.max(fr) - numpy.min(fr) + 1e-5) * 255.0
                                fr_resized = cv2.resize(fr.astype(numpy.uint8), (512, 512))
                                pil_frames.append(Image.fromarray(fr_resized))
                            if len(pil_frames) > 0:
                                framerate = getattr(dcm, "CineRate", getattr(dcm, "RecommendedDisplayFrameRate", 15))
                                try: framerate = float(framerate)
                                except: framerate = 15.0
                                if framerate <= 0: framerate = 15.0
                                pil_frames[0].save(tmp_gif_path, save_all=True, append_images=pil_frames[1:], duration=int(1000.0/framerate), loop=0)
                        if os.path.exists(tmp_gif_path):
                            c_gif_l, c_gif_c, c_gif_r = st.columns([1, 2, 1])
                            c_gif_c.image(tmp_gif_path, use_column_width=True)
                    else:
                        st.image(Image.fromarray(combined), use_column_width=True,
                                 caption="Start | Peak (QCA) | Last Frame")

                    calc_timi = str(timi)
                    current_ovr = meta.get("timi_override", "Auto")
                    if current_ovr not in ["Auto", "0", "1", "2", "3"]: current_ovr = "Auto"
                    meta["timi_override"] = st.selectbox("TIMI Grade", ["Auto", "0", "1", "2", "3"], index=["Auto", "0", "1", "2", "3"].index(current_ovr), key=f"timi_{name}")
                    st.caption(f"Calculated: Grade {timi} - {just}")
                    st.session_state.dicom_metadata[name] = meta

                    c_btn1, c_btn2 = st.columns(2)
                    vid_btn_text = "⏹️ Stop" if st.session_state.get(f"play_{name}", False) else "🎥 Odtwórz"
                    if c_btn1.button(vid_btn_text, key=f"play_btn_{name}"):
                        st.session_state[f"play_{name}"] = not st.session_state.get(f"play_{name}", False)
                        st.rerun()
                    if c_btn2.button("🔬 Analizuj (QCA)", key=f"btn_{name}"):
                        if st.session_state.selected_dicom != path:
                            for k in list(st.session_state.keys()):
                                if k not in ['current_view', 'stage', 'dicom_registry', 'dicom_metadata', 'patient_cart', 'patient_id']:
                                    del st.session_state[k]
                        st.session_state.current_view = 'analysis'
                        st.session_state.selected_dicom = path
                        st.session_state.dicomLabel = name
                        st.session_state.best_frame_ix = best_ix
                        st.rerun()

                except Exception as e:
                    st.warning(f"Error reading preview for {name}: {e}")
        idx += 1

    # ── Sort button ───────────────────────────────────────────────────────────
    st.markdown("---")
    n_chosen = sum(
        1 for n in st.session_state.dicom_metadata
        if st.session_state.dicom_metadata[n].get("chosen_for_analysis", False)
    )
    n_assigned = sum(
        1 for n in st.session_state.dicom_metadata
        if st.session_state.dicom_metadata[n].get("vessel_explicitly_set", False)
    )
    sort_col1, sort_col2, sort_col3 = st.columns([2, 1, 2])
    sort_col2.markdown(
        f"<div style='text-align:center; color:gray; font-size:13px;'>"
        f"⭐ {n_chosen} wybranych &nbsp;|&nbsp; 🩺 {n_assigned} z przypisanym naczyniem</div>",
        unsafe_allow_html=True
    )
    if sort_col2.button("🔃 Sortuj sekwencje", use_container_width=True):
        st.session_state.grid_order = _build_sorted_order(temp_sort_list)
        st.rerun()


    st.stop()

# ── ANALYSIS MODE ─────────────────────────────────────────────────────────────
selectedDicom = st.session_state.selected_dicom
dicomLabel = st.session_state.dicomLabel

if st.sidebar.button("🔙 Back to Selection Grid"):
    st.session_state.current_view = 'grid'
    st.rerun()

st.sidebar.success(f"Loaded: {dicomLabel}")



# key used for canvas reset when file changes
if "dicomDropDown" not in st.session_state:
    st.session_state["dicomDropDown"] = dicomLabel

stepOne = st.sidebar.expander("STEP ONE", True)
stepTwo = st.sidebar.expander("STEP TWO", True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Segmentation", "Analysis"])

st.markdown('''<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size:16px; }
</style>''', unsafe_allow_html=True)

# ── Load DICOM ────────────────────────────────────────────────────────────────
if selectedDicom is not None:
    try:
        dcm = pydicom.dcmread(selectedDicom, force=True)
        pixelArray = dcm.pixel_array
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        elif len(pixelArray.shape) == 3 and pixelArray.shape[2] == 3:
            pixelArray = pixelArray[numpy.newaxis, :, :, 0]
        elif len(pixelArray.shape) == 2:
            pixelArray = pixelArray[numpy.newaxis, ...]
            
        n_slices = pixelArray.shape[0]
    except Exception as e:
        st.error(f"Could not load DICOM file: {e}")
        st.stop()

    with stepOne:
        st.write("Select frame for annotation. The optimal frame has been pre-selected.")
        default_ix = st.session_state.best_frame_ix if st.session_state.best_frame_ix < n_slices else int(n_slices / 2)
        
        if n_slices > 1:
            slice_ix = st.slider('Frame', 0, n_slices - 1, default_ix, key='sliceSlider')
        else:
            slice_ix = 0
            
        _mask_key = f"predictedMask_{st.session_state.get('dicomLabel','')}_{slice_ix}"
        if _mask_key in st.session_state:
            predictedMask = st.session_state[_mask_key]
        else:
            predictedMask = numpy.zeros_like(pixelArray[slice_ix, :, :])

    with stepTwo:
        meta = st.session_state.dicom_metadata.get(st.session_state.dicomLabel, {}) if "dicomLabel" in st.session_state else {}
        # Derive the colour key from the vessel_system if available, else fall back
        _cur_sys = meta.get("vessel_system")
        if _cur_sys and _cur_sys in ALL_SYSTEM_NAMES:
            selectedArtery = _colour_key_from_system(_cur_sys)
            st.info(
                f"**Vessel:** {_cur_sys}  \n"
                f"**Segment:** {meta.get('aha_label', '—')}  |  **AHA:** {meta.get('aha', '—')}  |  "
                f"**FFR:** {meta.get('ffr_registered', 'N/A')}"
            )
        else:
            selectedArtery = "LAD"  # safe fallback for AI mask model
            st.warning("⚠️ **Nie wybrano naczynia.** Wróć do siatki i przypisz naczynie do tej sekwencji przed analizą QCA.")
        st.write("Beginning with the desired start point and finishing at the desired end point, click along the artery aiming for ~5-10 points.")



    # ── SEGMENTATION TAB ──────────────────────────────────────────────────────
    with tab1:
        objects = st.session_state.get(f"objects_{_mask_key}", pd.DataFrame())

        selectedFrame     = pixelArray[slice_ix, :, :]
        selectedFrame     = cv2.resize(selectedFrame, (512, 512))
        # Normalize to uint8 (DICOM frames are typically uint16)
        f_min, f_max = selectedFrame.min(), selectedFrame.max()
        if f_max > f_min:
            selectedFrame = ((selectedFrame.astype(numpy.float32) - f_min) / (f_max - f_min) * 255).astype(numpy.uint8)
        else:
            selectedFrame = numpy.zeros_like(selectedFrame, dtype=numpy.uint8)
        selectedFrameRGB  = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5 style='text-align:center; color:white;'>Selected frame</h5>", unsafe_allow_html=True)

            # ── Mode toggle ───────────────────────────────────────────────────
            canvasMode = st.radio(
                "Canvas mode:",
                ["📏 Calibrate catheter", "📍 Annotate artery"],
                horizontal=True,
                key="canvasMode"
            )

            isCalibMode = canvasMode.startswith("📏")

            if isCalibMode:
                CATHETER_SIZES = {"6F = 1.98 mm": 1.98, "5F = 1.67 mm": 1.67}
                catheterChoice = st.selectbox(
                    "Rozmiar cewnika:",
                    list(CATHETER_SIZES.keys()),
                    key="catheterSize"
                )
                catheterMm = CATHETER_SIZES[catheterChoice]

                c_head1, c_head2 = st.columns([3, 1])
                c_head1.caption("Zaznacz dwa punkty (początek i koniec) wzdłuż cewnika, aby wyznaczyć odcinek do kalibracji.")
                if c_head2.button("🔄 Redo Calib"):
                    for k in ["calibPoints", "mmPerPixelCalib", "calibLinePx", "canvas_calib_dots_v2"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

                # Pobieramy punkty i rysujemy na żywo NA LEWYM obrazie!
                calibBgFrame = selectedFrameRGB.copy()
                calibPointsList = st.session_state.get('calibPoints', [])
                if calibPointsList:
                    for item in calibPointsList:
                        if isinstance(item, dict):
                            ref_pt1, ref_pt2 = item['ref']
                            cv2.line(calibBgFrame, ref_pt1, ref_pt2, (0, 255, 0), 1)
                            if item['left_wall'] and item['right_wall']:
                                left_pts = numpy.array(item['left_wall'], numpy.int32).reshape((-1, 1, 2))
                                right_pts = numpy.array(item['right_wall'], numpy.int32).reshape((-1, 1, 2))
                                cv2.polylines(calibBgFrame, [left_pts], False, (255, 255, 0), 1)
                                cv2.polylines(calibBgFrame, [right_pts], False, (255, 255, 0), 1)
                        else:
                            if len(item) == 3: pt1, pt2, tube_theta = item
                            else: pt1, pt2, tube_theta = item[0], item[1], 0
                            cv2.line(calibBgFrame, pt1, pt2, (0, 255, 0), 1)
                            L_w = 25
                            dx, dy = L_w * numpy.cos(tube_theta), L_w * numpy.sin(tube_theta)
                            cv2.line(calibBgFrame, (int(pt1[0]-dx), int(pt1[1]-dy)), (int(pt1[0]+dx), int(pt1[1]+dy)), (255, 0, 0), 1)
                            cv2.line(calibBgFrame, (int(pt2[0]-dx), int(pt2[1]-dy)), (int(pt2[0]+dx), int(pt2[1]+dy)), (255, 0, 0), 1)

                calibDotCanvas = st_canvas(
                    fill_color="#00000000",
                    stroke_width=0,
                    stroke_color="#00000000",
                    background_color='black',
                    background_image=Image.fromarray(calibBgFrame),
                    update_streamlit=True,
                    height=600,
                    width=600,
                    drawing_mode="point",
                    point_display_radius=0,
                    key="canvas_calib_dots_v2",
                )

                if calibDotCanvas.json_data is not None:
                    dotObjs = pd.json_normalize(calibDotCanvas.json_data["objects"])
                    if len(dotObjs) >= 2 and "left" in dotObjs.columns:
                        detectedDiameters = []
                        calibPoints = []
                        def get_subpixel_peak(idx, array):
                            if idx == 0 or idx == len(array) - 1: return float(idx)
                            alpha, beta, gamma = array[idx-1], array[idx], array[idx+1]
                            denom = 2 * (alpha - 2*beta + gamma)
                            return float(idx) if denom == 0 else idx + (alpha - gamma) / denom

                        _cs = 512 / 600  # canvas→frame coordinate scale
                        cx1 = int((float(dotObjs.iloc[0].get("left", 0)) + 5) * _cs)
                        cy1 = int((float(dotObjs.iloc[0].get("top",  0)) + 5) * _cs)
                        cx2 = int((float(dotObjs.iloc[1].get("left", 0)) + 5) * _cs)
                        cy2 = int((float(dotObjs.iloc[1].get("top",  0)) + 5) * _cs)

                        cx1, cy1 = max(0, min(511, cx1)), max(0, min(511, cy1))
                        cx2, cy2 = max(0, min(511, cx2)), max(0, min(511, cy2))
                        
                        dx = cx2 - cx1
                        dy = cy2 - cy1
                        dist = numpy.hypot(dx, dy)
                        tube_theta = numpy.arctan2(dy, dx)
                        
                        if dist > 5:
                            # 1. Estimate base diameter at the middle to know how far out to look
                            mid_cx = (cx1 + cx2) / 2
                            mid_cy = (cy1 + cy2) / 2
                            cs_theta_init = tube_theta - numpy.pi/2
                            base_t = numpy.arange(-60, 60)
                            
                            tx = mid_cx + base_t * numpy.cos(cs_theta_init)
                            ty = mid_cy + base_t * numpy.sin(cs_theta_init)
                            coords = numpy.vstack((ty, tx))
                            profile = scipy.ndimage.map_coordinates(selectedFrame.astype(float), coords, mode='nearest')
                            smoothed = numpy.convolve(profile, numpy.ones(3)/3.0, mode='same')
                            grad = numpy.abs(numpy.gradient(smoothed))
                            
                            bestDiam = 15 # fallback diameter in pixels
                            if grad.max() > 0:
                                peaks = scipy.signal.find_peaks(grad, height=grad.max()*0.25, distance=4)[0]
                                if len(peaks) >= 2:
                                    top2 = numpy.argsort(grad[peaks])[-2:]
                                    bestDiam = abs(get_subpixel_peak(max(peaks[top2]), grad) - get_subpixel_peak(min(peaks[top2]), grad))
                            
                            # 2. Track exactly along the vector from point 1 to point 2
                            num_steps = max(2, int(dist / 5))
                            
                            lw, rw = [], []
                            caxis = tube_theta
                            
                            mid_x1, mid_y1, mid_x2, mid_y2 = None, None, None, None
                            
                            for step in range(num_steps + 1):
                                frac = step / float(num_steps)
                                ccx = cx1 + frac * dx
                                ccy = cy1 + frac * dy
                                
                                cs_theta = caxis - numpy.pi/2
                                track_L = max(15, int(bestDiam * 1.5))
                                track_t = numpy.arange(-track_L, track_L)
                                tx = ccx + track_t * numpy.cos(cs_theta)
                                ty = ccy + track_t * numpy.sin(cs_theta)
                                t_coords = numpy.vstack((ty, tx))
                                t_prof = scipy.ndimage.map_coordinates(selectedFrame.astype(float), t_coords, mode='nearest')
                                t_smooth = numpy.convolve(t_prof, numpy.ones(3)/3.0, mode='same')
                                t_grad = numpy.abs(numpy.gradient(t_smooth))
                                
                                if t_grad.max() == 0: continue
                                t_peaks = scipy.signal.find_peaks(t_grad, height=t_grad.max()*0.20, distance=max(2, int(bestDiam*0.5)))[0]
                                if len(t_peaks) >= 2:
                                    t_idx = numpy.argsort(t_grad[t_peaks])[-2:]
                                    tp0, tp1 = min(t_peaks[t_idx]), max(t_peaks[t_idx])
                                    sp0 = get_subpixel_peak(tp0, t_grad)
                                    sp1 = get_subpixel_peak(tp1, t_grad)
                                    cur_diam = sp1 - sp0
                                    detectedDiameters.append(cur_diam)
                                    
                                    woff_0 = -track_L + sp0
                                    woff_1 = -track_L + sp1
                                    wx0 = ccx + woff_0 * numpy.cos(cs_theta)
                                    wy0 = ccy + woff_0 * numpy.sin(cs_theta)
                                    wx1 = ccx + woff_1 * numpy.cos(cs_theta)
                                    wy1 = ccy + woff_1 * numpy.sin(cs_theta)
                                    lw.append((wx0, wy0))
                                    rw.append((wx1, wy1))
                                    
                                    if step == num_steps // 2:
                                        mid_x1, mid_y1, mid_x2, mid_y2 = wx0, wy0, wx1, wy1
                            
                            if lw and rw:
                                avg_diam = numpy.mean(detectedDiameters) if detectedDiameters else bestDiam
                                mid_lw = lw[len(lw)//2]
                                mid_rw = rw[len(rw)//2]
                                mid_cx = (mid_lw[0] + mid_rw[0]) / 2.0
                                mid_cy = (mid_lw[1] + mid_rw[1]) / 2.0
                                cs_theta = tube_theta - numpy.pi/2
                                gx1 = mid_cx - (avg_diam/2.0) * numpy.cos(cs_theta)
                                gy1 = mid_cy - (avg_diam/2.0) * numpy.sin(cs_theta)
                                gx2 = mid_cx + (avg_diam/2.0) * numpy.cos(cs_theta)
                                gy2 = mid_cy + (avg_diam/2.0) * numpy.sin(cs_theta)
                                
                                calibPoints.append({
                                    'ref': ((int(gx1), int(gy1)), (int(gx2), int(gy2))),
                                    'left_wall': lw,
                                    'right_wall': rw,
                                    'diam': avg_diam
                                })

                        if detectedDiameters:
                            avgDiam = numpy.mean([p['diam'] for p in calibPoints])
                            st.session_state["mmPerPixelCalib"]  = catheterMm / avgDiam
                            st.session_state["calibLinePx"]      = avgDiam
                            st.session_state["calibCatheterMm"]  = catheterMm
                            st.session_state["calibCatheterName"] = catheterChoice
                            
                            # Uruchomienie odświeżenia widoku lewego natychmiast po znalezieniu nowych ścian
                            if st.session_state.get("calibPoints") != calibPoints:
                                st.session_state["calibPoints"] = calibPoints
                                st.rerun()

                # Show calibration status
                mmPerPixelCalib = st.session_state.get("mmPerPixelCalib", None)
                if mmPerPixelCalib:
                    linePx    = st.session_state.get("calibLinePx", 0)
                    catName   = st.session_state.get("calibCatheterName", catheterChoice)
                    catMmDisp = st.session_state.get("calibCatheterMm", catheterMm)
                    st.success(f"✅ Calibration ({catName}): {linePx:.1f} px = {catMmDisp} mm → **{mmPerPixelCalib:.4f} mm/px**")
                else:
                    try:
                        dicomMmPx = float(dcm.ImagerPixelSpacing[0]) * (float(dcm.DistanceSourceToPatient) / float(dcm.DistanceSourceToDetector))
                        st.info(f"ℹ️ No catheter calibration yet — using DICOM metadata ({dicomMmPx:.4f} mm/px).")
                    except Exception:
                        st.info("ℹ️ No calibration yet.")

            else:
                a_head1, a_head2 = st.columns([3, 1])
                a_head1.caption("Click along the artery from start to end — aim for 5–10 points.")
                if a_head2.button("🔄 Redo Segment"):
                    seg_key = "canvas_seg_" + str(dicomLabel)
                    if seg_key in st.session_state:
                        del st.session_state[seg_key]
                    for k in [_mask_key, f"objects_{_mask_key}"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
                annotationCanvas = st_canvas(
                    fill_color="red",
                    stroke_width=2,
                    stroke_color="red",
                    background_color='black',
                    background_image=Image.fromarray(selectedFrameRGB),
                    update_streamlit=True,
                    height=600,
                    width=600,
                    drawing_mode="point",
                    point_display_radius=2,
                    key="canvas_seg_" + str(dicomLabel),
                )

                # Show calibration status in annotation mode too
                mmPerPixelCalib = st.session_state.get("mmPerPixelCalib", None)
                if mmPerPixelCalib:
                    linePx    = st.session_state.get("calibLinePx", 0)
                    catName   = st.session_state.get("calibCatheterName", "catheter")
                    catMmDisp = st.session_state.get("calibCatheterMm", 1.98)
                    st.success(f"✅ Calibration ({catName}): {linePx:.1f} px = {catMmDisp} mm → **{mmPerPixelCalib:.4f} mm/px**")
                else:
                    try:
                        dicomMmPx = float(dcm.ImagerPixelSpacing[0]) * (float(dcm.DistanceSourceToPatient) / float(dcm.DistanceSourceToDetector))
                        st.info(f"ℹ️ No catheter calibration yet — using DICOM metadata ({dicomMmPx:.4f} mm/px). Switch to 📏 mode to calibrate.")
                    except Exception:
                        st.info("ℹ️ No calibration yet. Switch to 📏 mode to calibrate.")

                # ── Annotation logic ──────────────────────────────────────────
                if annotationCanvas.json_data is not None:
                    objects = pd.json_normalize(annotationCanvas.json_data["objects"])

                    if len(objects) != 0:
                        for c in objects.select_dtypes(include=['object']).columns:
                            objects[c] = objects[c].astype("str")

                        _cs = 512 / 600  # canvas→frame coordinate scale
                        groundTruthPoints = numpy.vstack((
                            numpy.array(objects['top'].astype(float))   * _cs,
                            numpy.array(objects['left'].astype(float) + 3.5) * _cs
                        )).T

                        with st.spinner(f"Running segmentation on {len(objects)} points (30–60 s on CPU)…"):
                            try:
                                mask = angioPyFunctions.arterySegmentation(
                                    selectedFrame,
                                    groundTruthPoints,
                                )
                                predictedMask = predict.CoronaryDataset.mask2image(mask)
                                predictedMask = numpy.array(predictedMask)
                                st.session_state[_mask_key] = predictedMask
                                st.session_state[f"objects_{_mask_key}"] = objects
                            except Exception as segErr:
                                st.error(f"Segmentation error: {segErr}")

        with col2:
            if isCalibMode:
                st.markdown("<h5 style='text-align:center; color:white;'>Calibrated Catheter edges</h5>", unsafe_allow_html=True)
                st.markdown("<p style='text-align:center; color:white;'>Green dots show the edges detected by the app.</p>", unsafe_allow_html=True)
                
                calibShowFrame = selectedFrameRGB.copy()
                calibPointsList = st.session_state.get('calibPoints', [])
                if calibPointsList:
                    for item in calibPointsList:
                        if isinstance(item, dict):
                            ref_pt1, ref_pt2 = item['ref']
                            cv2.line(calibShowFrame, ref_pt1, ref_pt2, (0, 255, 0), 1)
                            if item['left_wall'] and item['right_wall']:
                                left_pts = numpy.array(item['left_wall'], numpy.int32).reshape((-1, 1, 2))
                                right_pts = numpy.array(item['right_wall'], numpy.int32).reshape((-1, 1, 2))
                                cv2.polylines(calibShowFrame, [left_pts], False, (255, 255, 0), 1)
                                cv2.polylines(calibShowFrame, [right_pts], False, (255, 255, 0), 1)
                        else:
                            if len(item) == 3: pt1, pt2, tube_theta = item
                            else: pt1, pt2, tube_theta = item[0], item[1], 0
                            cv2.line(calibShowFrame, pt1, pt2, (0, 255, 0), 1)
                            L_w = 25
                            dx, dy = L_w * numpy.cos(tube_theta), L_w * numpy.sin(tube_theta)
                            cv2.line(calibShowFrame, (int(pt1[0]-dx), int(pt1[1]-dy)), (int(pt1[0]+dx), int(pt1[1]+dy)), (255, 0, 0), 1)
                            cv2.line(calibShowFrame, (int(pt2[0]-dx), int(pt2[1]-dy)), (int(pt2[0]+dx), int(pt2[1]+dy)), (255, 0, 0), 1)
                st.image(calibShowFrame, width=600)
            else:
                st.markdown("<h5 style='text-align:center; color:white;'>🔍 Zoom & Mask Correction</h5>", unsafe_allow_html=True)
                
                col_z1, col_z2, col_z3 = st.columns(3)
                zoom = col_z1.slider("🔍 Zoom", 1.0, 4.0, 1.0, 0.5)
                h_z = int(512 / zoom)
                w_z = int(512 / zoom)
                
                if zoom > 1.0:
                    focus_x = col_z2.slider("🧭 Center X", w_z//2, 512 - w_z//2, 256)
                    focus_y = col_z3.slider("🧭 Center Y", h_z//2, 512 - h_z//2, 256)
                else:
                    focus_x = 256
                    focus_y = 256
                
                top_y = focus_y - h_z // 2
                left_x = focus_x - w_z // 2

                if "canvas_key_suffix" not in st.session_state:
                    st.session_state.canvas_key_suffix = 0

                canvas_key = f"maskCanvas_z_{zoom}_{focus_x}_{focus_y}_{st.session_state.canvas_key_suffix}"

                current_sig = f"{selectedDicom}_{slice_ix}_{selectedArtery}"
                if "cumulative_xor_sig" not in st.session_state or st.session_state.cumulative_xor_sig != current_sig:
                    st.session_state.cumulative_xor = numpy.zeros((512, 512), dtype=bool)
                    st.session_state.cumulative_xor_sig = current_sig

                active_canvas = st.session_state.get(canvas_key, None)
                if active_canvas is not None and getattr(active_canvas, 'json_data', None) is not None:
                    objects = active_canvas.json_data.get("objects", [])
                    if len(objects) > 0:
                        last_obj = objects[-1]
                        if last_obj["type"] == "path":
                            path_arr = last_obj["path"]
                            
                            stroke_pts = []
                            for cmd in path_arr:
                                if len(cmd) >= 3:
                                    stroke_pts.append([cmd[-2], cmd[-1]])
                            
                            if len(stroke_pts) > 2:
                                stroke_pts = numpy.array(stroke_pts, dtype=numpy.float32)
                                scaled_pts = numpy.zeros_like(stroke_pts, dtype=numpy.int32)
                                scaled_pts[:, 0] = left_x + stroke_pts[:, 0] * (w_z / 600.0)
                                scaled_pts[:, 1] = top_y + stroke_pts[:, 1] * (h_z / 600.0)
                                
                                s_rx, s_ry = scaled_pts[0]
                                e_rx, e_ry = scaled_pts[-1]
                                
                                if len(predictedMask.shape) == 3:
                                    v_mask_theo = numpy.any(predictedMask > 0, axis=-1)
                                else:
                                    v_mask_theo = predictedMask > 0
                                v_mask_theo = v_mask_theo ^ st.session_state.cumulative_xor
                                    
                                cnts, _ = cv2.findContours(v_mask_theo.astype(numpy.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                if len(cnts) > 0:
                                    main_cnt = max(cnts, key=cv2.contourArea)
                                    pts = main_cnt.reshape(-1, 2)
                                    dists_start = numpy.sum((pts - [s_rx, s_ry])**2, axis=1)
                                    dists_end = numpy.sum((pts - [e_rx, e_ry])**2, axis=1)
                                    idx_s = numpy.argmin(dists_start)
                                    idx_e = numpy.argmin(dists_end)
                                    
                                    if idx_s <= idx_e:
                                        path1 = pts[idx_s:idx_e+1]
                                        path2 = numpy.vstack((pts[idx_e:], pts[:idx_s+1]))
                                    else:
                                        path1 = pts[idx_e:idx_s+1]
                                        path2 = numpy.vstack((pts[idx_s:], pts[:idx_e+1]))
                                        
                                    c_sub = path1 if len(path1) < len(path2) else path2
                                    
                                    d1 = numpy.sum((c_sub[0] - [e_rx, e_ry])**2)
                                    d2 = numpy.sum((c_sub[-1] - [e_rx, e_ry])**2)
                                    if d2 < d1:
                                        c_sub = c_sub[::-1]
                                        
                                    poly_pts = numpy.vstack((scaled_pts, c_sub))
                                    
                                    diff_mask = numpy.zeros((512, 512), dtype=numpy.uint8)
                                    cv2.fillPoly(diff_mask, [poly_pts], 1)
                                    
                                    st.session_state.cumulative_xor = st.session_state.cumulative_xor ^ diff_mask.astype(bool)
                                
                                st.session_state.canvas_key_suffix += 1
                                st.rerun()

                # 2. Apply modifications mathematically BEFORE rendering green preview
                if len(predictedMask.shape) == 3:
                    v_mask_base = numpy.any(predictedMask > 0, axis=-1)
                else:
                    v_mask_base = predictedMask > 0
                
                v_mask_final = v_mask_base ^ st.session_state.cumulative_xor

                if len(predictedMask.shape) == 3:
                    color_bgr = angioPyFunctions.colourTableList[selectedArtery]
                    predictedMask[v_mask_final] = color_bgr
                    predictedMask[~v_mask_final] = (0, 0, 0)
                else:
                    predictedMask[v_mask_final] = 255
                    predictedMask[~v_mask_final] = 0

                st.markdown("<p style='text-align:left; color:#00ff00; font-size:14px; font-weight:bold;'>✍️ Smart Contour Snapping Active: Draw to nudge artery border</p>", unsafe_allow_html=True)
                c_mask2, _ = st.columns([1, 1])
                mask_stroke = c_mask2.slider("🖌 Stroke Target Thickness", 1, 30, 2)
                stroke_col = "rgba(0,255,0,255)"

                # 3. Create preview frame using updated predictedMask
                base_overlay = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGB)
                if len(predictedMask.shape) == 3:
                    v_mask = (numpy.any(predictedMask > 0, axis=-1) * 255).astype(numpy.uint8)
                    contour_preview = angioPyFunctions.maskOutliner(labelledArtery=v_mask, outlineThickness=1)
                    base_overlay[contour_preview, :] = [0, 255, 0]
                elif numpy.sum(predictedMask) > 0:
                    v_mask = (predictedMask > 0).astype(numpy.uint8) * 255
                    contour_preview = angioPyFunctions.maskOutliner(labelledArtery=v_mask, outlineThickness=1)
                    base_overlay[contour_preview, :] = [0, 255, 0]

                viewport_overlay = base_overlay[top_y:top_y+h_z, left_x:left_x+w_z]

                maskCanvas = st_canvas(
                    fill_color="rgba(255,255,255,0.0)",
                    stroke_width=mask_stroke,
                    stroke_color=stroke_col,
                    background_color='black',
                    background_image=Image.fromarray(viewport_overlay),
                    update_streamlit=True,
                    height=600,
                    width=600,
                    drawing_mode="freedraw",
                    point_display_radius=3,
                    key=canvas_key,
                )

    # ── ANALYSIS TAB ──────────────────────────────────────────────────────────
    if numpy.sum(predictedMask) > 0 and len(objects) > 4:
        b_channel, g_channel, r_channel = cv2.split(predictedMask)
        a_channel = numpy.full_like(predictedMask[:, :, 0], fill_value=255)
        predictedMaskRGBA = cv2.merge((predictedMask, a_channel))

        with tab2:
            # ── Compute centreline & vessel thicknesses ───────────────────────
            EDT  = scipy.ndimage.distance_transform_edt(cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2GRAY))
            skel = angioPyFunctions.skeletonise(predictedMaskRGBA)
            tck  = angioPyFunctions.skelSplinerWithThickness(skel=skel, EDT=EDT, smoothing=5)

            splinePointsY, splinePointsX, splineThicknesses = scipy.interpolate.splev(
                numpy.linspace(0.0, 1.0, 1000), tck)

            clippingLength = 20

            sp_x  = splinePointsX[clippingLength:-clippingLength]
            sp_y  = splinePointsY[clippingLength:-clippingLength]
            sp_dx = numpy.gradient(sp_x)
            sp_dy = numpy.gradient(sp_y)

            _mask_2d = numpy.any(predictedMask > 0, axis=-1)
            _h, _w   = _mask_2d.shape

            def _raycast_1d(cx, cy, nx, ny, max_r=80):
                cx, cy = float(cx), float(cy)
                for r in range(1, max_r + 1):
                    rx = int(round(cx + nx * r))
                    ry = int(round(cy + ny * r))
                    if rx < 0 or ry < 0 or rx >= _w or ry >= _h:
                        return float(r - 1)
                    if not _mask_2d[ry, rx]:
                        return float(r - 1)
                return float(max_r)

            _edt_thick = splineThicknesses[clippingLength:-clippingLength] * 2
            raycast_thicknesses = numpy.empty(len(sp_x), dtype=numpy.float32)
            for _i in range(len(sp_x)):
                _tx, _ty = sp_dx[_i], sp_dy[_i]
                _len = numpy.hypot(_tx, _ty)
                if _len > 0:
                    _nx, _ny = -_ty / _len, _tx / _len
                    _rp = _raycast_1d(sp_x[_i], sp_y[_i],  _nx,  _ny)
                    _rn = _raycast_1d(sp_x[_i], sp_y[_i], -_nx, -_ny)
                    raycast_thicknesses[_i] = _rp + _rn
                else:
                    raycast_thicknesses[_i] = _edt_thick[_i]

            _kernel = numpy.ones(7) / 7
            raycast_thicknesses = numpy.convolve(raycast_thicknesses, _kernel, mode='same')
            raycast_thicknesses[:3]  = raycast_thicknesses[3]
            raycast_thicknesses[-3:] = raycast_thicknesses[-4]
            vesselThicknesses = raycast_thicknesses

            # ── Controls row ─────────────────────────────────────────────────
            current_meta = st.session_state.dicom_metadata.get(st.session_state.dicomLabel, {}) if "dicomLabel" in st.session_state else {}
            current_phase = current_meta.get("phase", "PRE-PCI")

            def update_phase():
                if "dicomLabel" in st.session_state and st.session_state.dicomLabel in st.session_state.dicom_metadata:
                    st.session_state.dicom_metadata[st.session_state.dicomLabel]["phase"] = st.session_state.analysis_phase_toggle

            ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
            with ctrl1:
                st.radio("⏳ Procedure Phase", ["PRE-PCI", "POST-PCI"],
                         index=0 if current_phase == "PRE-PCI" else 1,
                         key="analysis_phase_toggle",
                         horizontal=True,
                         on_change=update_phase)
            swap_direction = ctrl2.checkbox("🔄 Swap Prox/Dist", value=False)
            ostial_lesion  = ctrl3.checkbox("🚨 Ostial Lesion",  value=False)

            # ── QCA indices ──────────────────────────────────────────────────
            refLen   = max(1, int(len(vesselThicknesses) * 0.20))
            prox_raw = numpy.argmax(vesselThicknesses[:refLen])
            dist_raw = len(vesselThicknesses) - refLen + numpy.argmax(vesselThicknesses[-refLen:])

            if swap_direction:
                prox_idx, dist_idx = dist_raw, prox_raw
            else:
                prox_idx, dist_idx = prox_raw, dist_raw

            idx_start = min(prox_idx, dist_idx)
            idx_end   = max(prox_idx, dist_idx)
            if ostial_lesion:
                idx_start = 0

            mld_idx = idx_start + numpy.argmin(vesselThicknesses[idx_start:idx_end+1])

            # ── Build contour image ───────────────────────────────────────────
            selectedFrameRGBA = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGBA)
            v_mask_right = (numpy.any(predictedMaskRGBA[:, :, :3] > 0, axis=-1) * 255).astype(numpy.uint8)
            contour = angioPyFunctions.maskOutliner(labelledArtery=v_mask_right, outlineThickness=1)
            selectedFrameRGBA[contour, :] = [
                angioPyFunctions.colourTableList[selectedArtery][2],
                angioPyFunctions.colourTableList[selectedArtery][1],
                angioPyFunctions.colourTableList[selectedArtery][0],
                255
            ]

            def draw_indicator_with_text(idx, label, color, is_mld=False):
                if idx < len(sp_x):
                    cx, cy = sp_x[idx], sp_y[idx]
                    tx, ty = sp_dx[idx], sp_dy[idx]
                    length = numpy.hypot(tx, ty)
                    if length > 0:
                        nx, ny = -ty / length, tx / length
                        radius = vesselThicknesses[idx] / 2.0
                        p1 = (int(cx + nx * radius), int(cy + ny * radius))
                        p2 = (int(cx - nx * radius), int(cy - ny * radius))
                        cv2.line(selectedFrameRGBA, p1, p2, color, 2)
                        cv2.putText(selectedFrameRGBA, label, (int(p1[0]+5), int(p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            _phase_here = st.session_state.dicom_metadata.get(dicomLabel, {}).get("phase", "UNKNOWN")
            if _phase_here != "POST-PCI":
                if not ostial_lesion:
                    draw_indicator_with_text(prox_idx, "Prox", (50, 255, 50, 255))
                draw_indicator_with_text(dist_idx, "Dist", (50, 255, 50, 255))
            draw_indicator_with_text(mld_idx, "MLD", (255, 50, 50, 255), is_mld=True)

            # ── Contours image — centered, large ─────────────────────────────
            st.markdown("<h5 style='text-align:center; color:white;'>Contours</h5>", unsafe_allow_html=True)
            fig2 = px.imshow(selectedFrameRGBA)
            fig2.update_xaxes(visible=False)
            fig2.update_yaxes(visible=False)
            fig2.update_layout(margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0},
                               height=700)
            fig2.update_traces(dict(showscale=False, coloraxis=None, colorscale='gray'), selector={'type': 'heatmap'})
            fig2.add_trace(go.Scatter(
                x=splinePointsX[clippingLength:-clippingLength],
                y=splinePointsY[clippingLength:-clippingLength],
                line=dict(width=1)
            ))
            _, fig2_col, _ = st.columns([1, 6, 1])
            with fig2_col:
                st.plotly_chart(fig2, use_container_width=True)

            # ── QCA METRICS (outside nested columns) ──────────────────────────
            mmPerPixelCalib = st.session_state.get("mmPerPixelCalib", None)
            if mmPerPixelCalib:
                mmPerPixel  = mmPerPixelCalib
                calibSource = st.session_state.get("calibCatheterName", "catheter")
            else:
                try:
                    mmPerPixel  = float(dcm.ImagerPixelSpacing[0]) * (float(dcm.DistanceSourceToPatient) / float(dcm.DistanceSourceToDetector))
                    calibSource = "DICOM metadata"
                except Exception:
                    mmPerPixel  = None
                    calibSource = "unknown"

            origH, origW = pixelArray[slice_ix].shape[:2]
            pxToMm = (origW / 512.0) * mmPerPixel if mmPerPixel else None

            # Resolve phase from metadata HERE (needed before QCA branching)
            meta  = st.session_state.dicom_metadata.get(dicomLabel, {})
            phase = meta.get("phase", "UNKNOWN")

            refLen = max(1, int(len(vesselThicknesses) * 0.20))

            is_post_pci = phase == "POST-PCI"

            if is_post_pci:
                # POST-PCI: measure at mask edges (proximal = start of centreline, distal = end)
                edge_avg = max(1, int(len(vesselThicknesses) * 0.05))  # avg over 5% edge
                if swap_direction:
                    proxDiamMm = numpy.mean(vesselThicknesses[-edge_avg:]) * (pxToMm or 1.0)
                    distDiamMm = numpy.mean(vesselThicknesses[:edge_avg])  * (pxToMm or 1.0)
                else:
                    proxDiamMm = numpy.mean(vesselThicknesses[:edge_avg])  * (pxToMm or 1.0)
                    distDiamMm = numpy.mean(vesselThicknesses[-edge_avg:]) * (pxToMm or 1.0)
                refDiamMm  = (proxDiamMm + distDiamMm) / 2.0
                # For post-PCI, MLD = minimum within full stented segment
                mldMm = numpy.min(vesselThicknesses) * (pxToMm or 1.0)
                prox_idx = 0
                dist_idx = len(vesselThicknesses) - 1
            else:
                prox_raw = numpy.argmax(vesselThicknesses[:refLen])
                dist_raw = len(vesselThicknesses) - refLen + numpy.argmax(vesselThicknesses[-refLen:])

                if swap_direction:
                    prox_idx, dist_idx = dist_raw, prox_raw
                else:
                    prox_idx, dist_idx = prox_raw, dist_raw

                proxDiamMm = vesselThicknesses[prox_idx] * (pxToMm or 1.0)
                distDiamMm = vesselThicknesses[dist_idx] * (pxToMm or 1.0)

                if ostial_lesion:
                    refDiamMm = distDiamMm
                else:
                    refDiamMm = (proxDiamMm + distDiamMm) / 2.0

                idx_start = min(prox_idx, dist_idx)
                idx_end   = max(prox_idx, dist_idx)
                if ostial_lesion:
                    idx_start = 0
                mldMm = numpy.min(vesselThicknesses[idx_start:idx_end+1]) * (pxToMm or 1.0)

            pctDiam = (1.0 - mldMm / refDiamMm) * 100.0 if refDiamMm > 0 else 0.0
            pctArea = (1.0 - (mldMm / refDiamMm) ** 2) * 100.0 if refDiamMm > 0 else 0.0

            # arc length along centreline (spline points are in 512-scale px → convert)
            spX    = splinePointsX[clippingLength:-clippingLength] * (origW / 512.0)
            spY    = splinePointsY[clippingLength:-clippingLength] * (origH / 512.0)
            diffs  = numpy.sqrt(numpy.diff(spX) ** 2 + numpy.diff(spY) ** 2)
            cumLen = numpy.concatenate([[0], numpy.cumsum(diffs)])  # in original pixels
            totalLenMm = cumLen[-1] * mmPerPixel if mmPerPixel else cumLen[-1]

            # stenosis length: segment where diameter < 50 % of reference
            vesselThicknessMm = vesselThicknesses * (pxToMm or 1.0)
            stenosisMask = vesselThicknessMm < (refDiamMm * 0.5)
            if numpy.any(stenosisMask):
                idxs = numpy.where(stenosisMask)[0]
                stenosisLenMm = (cumLen[min(idxs[-1], len(cumLen)-1)] - cumLen[idxs[0]]) * (mmPerPixel or 1.0)
            else:
                stenosisLenMm = 0.0

            st.markdown("---")
            st.markdown("<h5 style='color:white;'>📐 QCA Metrics</h5>", unsafe_allow_html=True)
            if mmPerPixelCalib:
                st.success(f"✅ **{calibSource} calibration** ({mmPerPixel:.4f} mm/px)")
            else:
                st.warning(f"⚠️ **DICOM metadata** ({mmPerPixel:.4f} mm/px) — switch to 📏 mode in Segmentation for accuracy")

            if mmPerPixel:
                m1, m2, m3 = st.columns(3)
                m1.metric("% Diameter Stenosis", f"{pctDiam:.1f}%")
                m2.metric("% Area Stenosis",     f"{pctArea:.1f}%")
                m3.metric("MLD",                 f"{mldMm:.2f} mm")
                m4, m5, m6 = st.columns(3)
                if is_post_pci:
                    m4.metric("Proximal Edge Diameter", f"{proxDiamMm:.2f} mm")
                    m5.metric("Distal Edge Diameter",   f"{distDiamMm:.2f} mm")
                else:
                    m4.metric("Max Proximal Reference", f"{proxDiamMm:.2f} mm")
                    m5.metric("Max Distal Reference",   f"{distDiamMm:.2f} mm")
                m6.metric("Reference Diameter",  f"{refDiamMm:.2f} mm")
                m7, m8 = st.columns(2)
                length_label = "Stent Length" if is_post_pci else "Lesion Length"
                m7.metric(length_label, f"{totalLenMm:.1f} mm")
            else:
                m1, m2 = st.columns(2)
                m1.metric("% Diameter Stenosis", f"{pctDiam:.1f}%")
                m2.metric("% Area Stenosis",     f"{pctArea:.1f}%")

            # ── EXPORT ────────────────────────────────────────────────────────
            dicomBaseName = os.path.splitext(os.path.basename(selectedDicom))[0]
            st.markdown("---")
            st.markdown("<h5 style='color:white;'>Export results</h5>", unsafe_allow_html=True)

            maskBuf = io.BytesIO()
            Image.fromarray(predictedMask).save(maskBuf, format="PNG")
            st.download_button("⬇ Download mask (PNG)", data=maskBuf.getvalue(),
                file_name=f"{dicomBaseName}_{selectedArtery}_mask_frame{slice_ix}.png", mime="image/png")

            thicknessDf = pd.DataFrame({
                "centreline_point": numpy.arange(1, len(vesselThicknesses) + 1),
                "thickness_px":     vesselThicknesses,
                "thickness_mm":     vesselThicknessMm,
                "arc_length_mm":    cumLen * (mmPerPixel if mmPerPixel else 1.0),
            })
            summaryRows = pd.DataFrame([
                {"centreline_point": "--- QCA SUMMARY ---"},
                {"centreline_point": "calibration_source",      "thickness_px": calibSource},
                {"centreline_point": "pct_diameter_stenosis_%", "thickness_px": round(pctDiam, 2)},
                {"centreline_point": "pct_area_stenosis_%",     "thickness_px": round(pctArea, 2)},
                {"centreline_point": "MLD_mm",                  "thickness_px": round(mldMm, 3)},
                {"centreline_point": "max_proximal_ref_mm" if not is_post_pci else "proximal_edge_mm",    "thickness_px": round(proxDiamMm, 3)},
                {"centreline_point": "max_distal_ref_mm"   if not is_post_pci else "distal_edge_mm",      "thickness_px": round(distDiamMm, 3)},
                {"centreline_point": "reference_diameter_mm",   "thickness_px": round(refDiamMm, 3)},
                {"centreline_point": "stent_length_mm" if is_post_pci else "lesion_length_mm",   "thickness_px": round(totalLenMm, 2)},
            ])
            csvBuf = io.StringIO()
            pd.concat([thicknessDf, summaryRows], ignore_index=True).to_csv(csvBuf, index=False)
            st.download_button("⬇ Download QCA (CSV)", data=csvBuf.getvalue(),
                file_name=f"{dicomBaseName}_{selectedArtery}_QCA_frame{slice_ix}.csv", mime="text/csv")

            overlayBuf = io.BytesIO()
            Image.fromarray(selectedFrameRGBA).save(overlayBuf, format="PNG")
            st.download_button("⬇ Download overlay (PNG)", data=overlayBuf.getvalue(),
                file_name=f"{dicomBaseName}_{selectedArtery}_overlay_frame{slice_ix}.png", mime="image/png")

            # --- SAVE TO PATIENT REPORT CART ---
            st.markdown("---")
            st.markdown("<h5 style='color:white;'>Patient Report Tagging</h5>", unsafe_allow_html=True)
            
            meta = st.session_state.dicom_metadata.get(dicomLabel, {"phase": "UNKNOWN", "vessel": "LAD", "aha": "6"})
            c1, c2 = st.columns(2)
            
            cur_system = meta.get("vessel_system", ALL_SYSTEM_NAMES[1])
            if cur_system not in ALL_SYSTEM_NAMES: cur_system = ALL_SYSTEM_NAMES[1]
            chosen_system = c1.selectbox("Vessel System", ALL_SYSTEM_NAMES, index=ALL_SYSTEM_NAMES.index(cur_system), key="cart_vessel")
            meta["vessel_system"] = chosen_system
            meta["vessel"] = _colour_key_from_system(chosen_system)

            seg_labels = _seg_labels(chosen_system)
            seg_codes  = _seg_codes(chosen_system)
            cur_aha_label = meta.get("aha_label", seg_labels[0])
            if cur_aha_label not in seg_labels: cur_aha_label = seg_labels[0]
            chosen_seg_lbl = c2.selectbox("Segment", seg_labels, index=seg_labels.index(cur_aha_label), key="cart_aha")
            meta["aha_label"] = chosen_seg_lbl
            meta["aha"] = seg_codes[seg_labels.index(chosen_seg_lbl)]
            
            c3, c4 = st.columns(2)
            cur_ffr = meta.get("ffr_registered", "N/A")
            if cur_ffr not in ["Yes", "No", "N/A"]: cur_ffr = "N/A"
            meta["ffr_registered"] = c3.radio("FFR Wire Registered?", ["Yes", "No", "N/A"], index=["Yes", "No", "N/A"].index(cur_ffr), horizontal=True, key="cart_ffr")
            if meta["phase"] == "PRE-PCI":
                meta["other_lesion_distal"] = c4.radio("Other lesion >50% distal to FFR?", ["Yes", "No", "N/A"], index=1, horizontal=True, key="cart_distal_pre")
            else:
                meta["other_lesion_distal"] = c4.radio("Other lesion >50% distal to DES/DCB?", ["Yes", "No", "N/A"], index=1, horizontal=True, key="cart_distal_post")
            
            c5, _ = st.columns(2)
            meta["known_occlude"] = c5.radio("Known occluded vessel (TIMI 0)?", ["Yes", "No"], index=1, horizontal=True)
            
            st.session_state.dicom_metadata[dicomLabel] = meta
            
            st.markdown("####")
            c_pid, _ = st.columns(2)
            pid_val = c_pid.text_input("Patient ID Number (Required)", value=st.session_state.patient_id, key="cart_pid_input")
            st.session_state.patient_id = pid_val
            
            has_pid = bool(st.session_state.patient_id and st.session_state.patient_id.strip() != "")
            if not has_pid:
                st.error("🚨 Brak Patient ID! Uzupełnij pole powyżej, aby móc zapisać analizę.")
            
            if st.button("💾 Save to Patient Report Cart", use_container_width=True, disabled=not has_pid):
                try:
                    _, _, _, tfc, timi_calc, just = analyze_series_flow(selectedDicom, os.path.getsize(selectedDicom))
                except:
                    tfc, timi_calc, just = "N/A", "N/A", "N/A"
                    
                final_timi = meta.get("timi_override", str(timi_calc))
                final_timi = str(timi_calc) if final_timi == "Auto" else final_timi
                
                if meta["known_occlude"] == "Yes":
                    final_timi = "0"
                    distDiamMm = "N/A"
                    refDiamMm = "N/A"
                    mldMm = "N/A"
                    pctDiam = "N/A"
                    pctArea = "N/A"
                    totalLenMm = "N/A"
                
                cart_item = {
                    "dicom_name": dicomLabel,
                    "phase": meta["phase"],
                    "vessel": meta["vessel"],
                    "vessel_system": meta.get("vessel_system", ""),
                    "aha": meta["aha"],
                    "aha_label": meta.get("aha_label", meta["aha"]),
                    "ffr_registered": meta.get("ffr_registered", "N/A"),
                    "other_lesion_distal": meta.get("other_lesion_distal", "N/A"),
                    "metrics": {
                        "prox": proxDiamMm,
                        "dist": distDiamMm,
                        "ref": refDiamMm,
                        "mld": mldMm,
                        "pct_diam": pctDiam,
                        "pct_area": pctArea,
                        "lesion_len": totalLenMm,
                        "is_post_pci": is_post_pci,
                        "tfc": tfc,
                        "timi": final_timi,
                        "just": just
                    },
                    "image": cv2.cvtColor(selectedFrameRGBA, cv2.COLOR_RGBA2RGB)
                }
                
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_pdf import PdfPages
                    import os
                    save_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    safe_patient_id = st.session_state.patient_id if st.session_state.patient_id else "NoID"
                    pdf_filename = f"{safe_patient_id}_{meta['vessel']}_{meta['phase']}.pdf"
                    pdf_path = os.path.join(save_dir, pdf_filename)
                    
                    with PdfPages(pdf_path) as pdf:
                        fig_pdf = plt.figure(figsize=(8.27, 11.69))
                        fig_pdf.text(0.5, 0.96, "AngioPy Single Evaluation Report", ha='center', fontsize=18, weight='bold')
                        fig_pdf.text(0.5, 0.92, f"Patient: {safe_patient_id}  |  Phase: {meta['phase']}  |  Segment: AHA {meta['aha']} ({meta['vessel']})", ha='center', fontsize=12)
                        fig_pdf.text(0.1, 0.85, f"TIMI Flow Scale: Grade {final_timi}", fontsize=14, weight='bold', color='darkred')
                        fig_pdf.text(0.1, 0.82, f"TFC (TIMI Frame Count): {tfc}", fontsize=12)
                        fig_pdf.text(0.1, 0.79, f"Justification: {just}", fontsize=11, style='italic')
                        
                        if meta['phase'] == 'PRE-PCI':
                            lbl = ">50% distal to FFR"
                            ffr_txt = f"FFR Registered: {meta['ffr_registered']}  |  "
                        else:
                            lbl = ">50% distal to DES/DCB"
                            ffr_txt = ""
                            
                        fig_pdf.text(0.1, 0.76, f"{ffr_txt}{lbl}: {meta['other_lesion_distal']}", fontsize=12, weight='bold')
                        fig_pdf.text(0.1, 0.72, "QCA Metrics Summary", fontsize=14, weight='bold')
                        
                        str_dist = "N/A" if distDiamMm == "N/A" else f"{distDiamMm:.2f} mm"
                        str_ref = "N/A" if refDiamMm == "N/A" else f"{refDiamMm:.2f} mm"
                        str_mld = "N/A" if mldMm == "N/A" else f"{mldMm:.2f} mm"
                        str_pctD = "N/A" if pctDiam == "N/A" else f"{pctDiam:.1f} %"
                        str_pctA = "N/A" if pctArea == "N/A" else f"{pctArea:.1f} %"
                        str_len = "N/A" if totalLenMm == "N/A" else f"{totalLenMm:.2f} mm"
                        
                        m_text = (
                            f"Max Proximal Reference:      {proxDiamMm:.2f} mm\n\n"
                            f"Max Distal Reference:        {str_dist}\n\n"
                            f"Calculated Reference:        {str_ref}\n\n"
                            f"Minimum Lumen Diameter:      {str_mld}\n\n"
                            f"% Diameter Stenosis:         {str_pctD}\n\n"
                            f"% Area Stenosis:             {str_pctA}\n\n"
                            f"Lesion Length:               {str_len}"
                        )
                        fig_pdf.text(0.1, 0.68, m_text, fontsize=11, family='monospace', va='top')
                        ax = fig_pdf.add_axes([0.1, 0.05, 0.8, 0.45])
                        ax.imshow(cv2.cvtColor(selectedFrameRGBA, cv2.COLOR_RGBA2RGB))
                        ax.axis('off')
                        pdf.savefig(fig_pdf)
                        plt.close(fig_pdf)
                except Exception as e:
                    pass
                
                try:
                    import pandas as pd
                    import os
                    
                    target_xlsx = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/AngioPy.xlsx")
                    
                    row = {
                        "Patient ID": safe_patient_id, 
                        "DICOM Name": dicomLabel, 
                        "Phase": meta["phase"],
                        "Vessel": meta["vessel"],
                        "AHA Segment": meta["aha"],
                        "FFR position registered": meta["ffr_registered"],
                        "Other lesion >50% distal": meta["other_lesion_distal"],
                        "Known Occluded Vessel": meta["known_occlude"],
                        "Max Prox [mm]": round(proxDiamMm, 2),
                        "Max Dist [mm]": "N/A" if distDiamMm == "N/A" else round(distDiamMm, 2),
                        "Reference [mm]": "N/A" if refDiamMm == "N/A" else round(refDiamMm, 2),
                        "MLD [mm]": "N/A" if mldMm == "N/A" else round(mldMm, 2),
                        "% Diameter Stenosis": "N/A" if pctDiam == "N/A" else round(pctDiam, 1),
                        "% Area Stenosis": "N/A" if pctArea == "N/A" else round(pctArea, 1),
                        "Lesion Length [mm]": "N/A" if totalLenMm == "N/A" else round(totalLenMm, 2),
                        "TIMI Grade": final_timi,
                        "TFC": tfc
                    }
                    
                    df_new = pd.DataFrame([row])
                    
                    if os.path.exists(target_xlsx):
                        try:
                            df_existing = pd.read_excel(target_xlsx)
                            df_final = pd.concat([df_existing, df_new], ignore_index=True)
                        except Exception:
                            df_final = df_new
                    else:
                        os.makedirs(os.path.dirname(target_xlsx), exist_ok=True)
                        df_final = df_new
                        
                    df_final.to_excel(target_xlsx, index=False)
                except Exception as e:
                    pass
                
                st.session_state.patient_cart.append(cart_item)
                st.session_state.current_view = 'grid'
                st.rerun()
