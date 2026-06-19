import os
import os.path
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import streamlit as st
st.set_page_config(page_title="AngioPy Segmentation", layout="wide")
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
import angioPyFunctions
import scipy
import scipy.signal
import scipy.ndimage
import cv2
import io
import ssl
import threading
import time
import json
import re
from ecrf_extractor import ECRFExtractor, build_clean_json

import firebase_admin
from firebase_admin import credentials, firestore, storage
import hashlib
import urllib.parse
import datetime

# Initialize Firebase states in session state
if "firebase_init" not in st.session_state:
    st.session_state.firebase_init = False
if "firebase_error" not in st.session_state:
    st.session_state.firebase_error = None
if "firestore_db" not in st.session_state:
    st.session_state.firestore_db = None
if "firebase_bucket" not in st.session_state:
    st.session_state.firebase_bucket = None
if "user" not in st.session_state:
    st.session_state.user = None

def hash_password(username, password):
    salt = f"angiopy_{username.lower()}_salt"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

class DicomMetadataMock:
    def __init__(self, spacing, dist_p, dist_d, cine_rate, rec_rate):
        self.ImagerPixelSpacing = spacing
        self.DistanceSourceToPatient = dist_p
        self.DistanceSourceToDetector = dist_d
        self.CineRate = cine_rate
        self.RecommendedDisplayFrameRate = rec_rate

@st.cache_data(max_entries=3, ttl=300, show_spinner=False)
def load_dicom_data(dicom_path):
    dcm = pydicom.dcmread(dicom_path, force=True)
    pixelArray = dcm.pixel_array
    if len(pixelArray.shape) == 4:
        pixelArray = pixelArray[:, :, :, 0]
    elif len(pixelArray.shape) == 3 and pixelArray.shape[2] == 3:
        pixelArray = pixelArray[numpy.newaxis, :, :, 0]
    elif len(pixelArray.shape) == 2:
        pixelArray = pixelArray[numpy.newaxis, ...]
    
    dist_p = getattr(dcm, "DistanceSourceToPatient", None)
    dist_d = getattr(dcm, "DistanceSourceToDetector", None)
    spacing = getattr(dcm, "ImagerPixelSpacing", None)
    cine_rate = getattr(dcm, "CineRate", None)
    rec_rate = getattr(dcm, "RecommendedDisplayFrameRate", None)

    if dist_p is not None:
        try: dist_p = float(dist_p)
        except Exception: dist_p = None
    if dist_d is not None:
        try: dist_d = float(dist_d)
        except Exception: dist_d = None
    if spacing is not None:
        try: spacing = [float(x) for x in spacing]
        except Exception: spacing = None
    if cine_rate is not None:
        try: cine_rate = float(cine_rate)
        except Exception: cine_rate = None
    if rec_rate is not None:
        try: rec_rate = float(rec_rate)
        except Exception: rec_rate = None

    return pixelArray, dist_p, dist_d, spacing, cine_rate, rec_rate


def init_firebase():
    if st.session_state.firebase_init:
        return True
    
    cred_file = "google_credentials.json"
    if not os.path.exists(cred_file):
        st.session_state.firebase_error = "The google_credentials.json file does not exist on the server."
        st.session_state.firebase_init = False
        return False
        
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_file)
            with open(cred_file, "r") as f:
                cred_data = json.load(f)
            project_id = cred_data.get("project_id")
            
            # Resolve bucket name dynamically
            bucket_name = f"{project_id}.firebasestorage.app"
            api_key = "AIzaSyAfowkyjJjKDZ6mNyvZJqBk2FoiDMI-iGY"
            try:
                from google.cloud import storage as gcs
                client = gcs.Client.from_service_account_json(cred_file)
                buckets = [b.name for b in client.list_buckets()]
                if buckets:
                    for b in buckets:
                        if b.startswith(project_id):
                            bucket_name = b
                            break
                    else:
                        bucket_name = buckets[0]
            except Exception:
                pass
                
            try:
                from google.auth.transport.requests import AuthorizedSession
                from google.oauth2 import service_account
                gcred = service_account.Credentials.from_service_account_file(cred_file, scopes=["https://www.googleapis.com/auth/cloud-platform"])
                session = AuthorizedSession(gcred)
                res = session.get(f"https://firebase.googleapis.com/v1beta1/projects/{project_id}/webApps")
                apps = res.json().get("apps", [])
                if apps:
                    app_name = apps[0]["name"]
                    res_config = session.get(f"https://firebase.googleapis.com/v1beta1/{app_name}/config")
                    api_key = res_config.json().get("apiKey", api_key)
            except Exception:
                pass
                
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            st.session_state.firebase_api_key = api_key
            
        st.session_state.firestore_db = firestore.client()
        st.session_state.firebase_bucket = storage.bucket()
        st.session_state.firebase_init = True
        st.session_state.firebase_error = None
        return True
    except Exception as e:
        st.session_state.firebase_error = str(e)
        st.session_state.firebase_init = False
        return False

def verify_firebase_auth(email_or_username, password):
    import requests
    if not st.session_state.firebase_init:
        return None
    
    email = email_or_username
    if "@" not in email_or_username:
        try:
            from firebase_admin import auth
            page = auth.list_users()
            resolved = False
            while page:
                for user in page.users:
                    if user.email and user.email.lower().split("@")[0] == email_or_username.lower():
                        email = user.email
                        resolved = True
                        break
                if resolved:
                    break
                page = page.get_next_page()
        except Exception as e:
            print(f"Error resolving email: {e}")
            
    try:
        api_key = st.session_state.get("firebase_api_key", "AIzaSyAfowkyjJjKDZ6mNyvZJqBk2FoiDMI-iGY")
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            data = res.json()
            from firebase_admin import auth
            user_info = auth.get_user(data["localId"])
            name = user_info.display_name or user_info.email.split("@")[0]
            is_admin = email.lower() == "tomaszroleder@gmail.com" or email.lower().startswith("tomaszroleder@")
            user_data = {
                "uid": data["localId"],
                "username": email.lower().split("@")[0],
                "email": email,
                "name": name,
                "role": "admin" if is_admin else "analyst"
            }
            print(f"[DEBUG] verify_firebase_auth user_data: {user_data}")
            return user_data
        else:
            return None
    except Exception as e:
        print(f"Error in verify_firebase_auth: {e}")
        return None

def render_login_page():
    st.markdown(
        """
        <style>
        .login-card {
            background-color: #1e1e24;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            max-width: 450px;
            margin: 4rem auto;
            border: 1px solid #333;
        }
        .login-title {
            text-align: center;
            color: #00ff00;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .login-subtitle {
            text-align: center;
            color: #aaa;
            font-size: 14px;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<h1 class="login-title">AngioPy</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Coronary Angiography Analysis System</p>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Email or Username (login)", key="login_username").strip()
            password = st.text_input("Password", type="password", key="login_password")
            
            submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("Please enter email/username and password.")
                else:
                    user_data = verify_firebase_auth(username, password)
                    if user_data:
                        st.session_state.user = user_data
                        st.success("Logged in successfully!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                            
        st.markdown('</div>', unsafe_allow_html=True)

def render_analyst_management():
    if not st.session_state.firebase_init:
        return
    
    st.markdown("##### Add new analyst")
    with st.form("add_analyst_form", clear_on_submit=True):
        new_email = st.text_input("Email Address (e.g. john@gmail.com)").strip()
        new_name = st.text_input("First and Last Name")
        new_pass = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Add user", use_container_width=True)
        if submitted:
            if not new_email or not new_name or not new_pass:
                st.error("Please fill in all fields.")
            elif "@" not in new_email:
                st.error("Please enter a valid email address.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                from firebase_admin import auth
                try:
                    user = auth.create_user(
                        email=new_email,
                        password=new_pass,
                        display_name=new_name
                    )
                    get_all_analysts.clear()
                    st.success(f"Successfully added user {new_email}!")
                except Exception as e:
                    st.error(f"Error adding user: {e}")

def upload_pdf_to_firebase(pdf_bytes, pdf_filename):
    if not st.session_state.firebase_init or st.session_state.firebase_bucket is None:
        return None
    try:
        bucket = st.session_state.firebase_bucket
        blob_path = f"reports/{pdf_filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        
        # 1. Try public access
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            pass
            
        # 2. Try signed URL
        try:
            url = blob.generate_signed_url(expiration=datetime.timedelta(days=365))
            return url
        except Exception:
            pass
            
        # 3. Fallback direct link
        escaped_path = urllib.parse.quote(blob_path, safe='')
        return f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{escaped_path}?alt=media"
    except Exception as e:
        print(f"Error in upload_pdf_to_firebase: {e}")
        return None

def save_analysis_to_firestore(row, pdf_url):
    if not st.session_state.firebase_init or st.session_state.firestore_db is None:
        return False
    try:
        db = st.session_state.firestore_db
        ref_diam_val = None
        for k, v in row.items():
            if k.endswith("Reference [mm]"):
                ref_diam_val = v
                break
                
        doc_data = {
            "patient_id": row.get("Patient ID"),
            "dicom_name": row.get("DICOM Name"),
            "phase": row.get("Phase"),
            "vessel": row.get("Vessel"),
            "aha": row.get("AHA Segment"),
            "ffr_registered": row.get("FFR position registered"),
            "other_lesion_distal": row.get("Other lesion >50% distal"),
            "known_occlude": row.get("Known Occluded Vessel"),
            "pdf_url": pdf_url,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "analyst": st.session_state.user.get("username") if st.session_state.user else "unknown",
            "metrics": {
                "prox_diam_mm": row.get("Max Prox [mm]"),
                "dist_diam_mm": row.get("Max Dist [mm]"),
                "ref_diam_mm": ref_diam_val,
                "mld_mm": row.get("MLD [mm]"),
                "pct_diameter_stenosis": row.get("% Diameter Stenosis"),
                "pct_area_stenosis": row.get("% Area Stenosis"),
                "lesion_length_mm": row.get("Lesion Length [mm]"),
                "timi_grade": row.get("TIMI Grade"),
                "tfc": row.get("TFC")
            }
        }
        db.collection("analysis_results").add(doc_data)
        clear_db_caches()
        return True
    except Exception as e:
        print(f"Error saving analysis to Firestore: {e}")
        return False

def save_chosen_dicoms_to_firestore(patient_id, chosen_dicoms):
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return
    try:
        db = st.session_state.firestore_db
        doc_ref = db.collection("patient_selections").document(patient_id)
        doc_ref.set({
            "patient_id": patient_id,
            "chosen_dicoms": chosen_dicoms,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "updated_by": st.session_state.user.get("username") if st.session_state.user else "unknown"
        })
    except Exception as e:
        print(f"Error saving patient selections: {e}")

def get_chosen_dicoms_from_firestore(patient_id):
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return []
    try:
        db = st.session_state.firestore_db
        doc_ref = db.collection("patient_selections").document(patient_id)
        doc = doc_ref.get()
        chosen = []
        if doc.exists:
            chosen = doc.to_dict().get("chosen_dicoms", [])
        
        reports_ref = db.collection("analysis_results").where("patient_id", "==", patient_id).stream()
        for r in reports_ref:
            d = r.to_dict()
            dicom_name = d.get("dicom_name")
            if dicom_name and dicom_name not in chosen:
                chosen.append(dicom_name)
        return chosen
    except Exception as e:
        print(f"Error fetching chosen dicoms: {e}")
        return []

def parse_qca_pdf(pdf_file_or_bytes):
    try:
        from pypdf import PdfReader
        import io
        
        if isinstance(pdf_file_or_bytes, bytes):
            stream = io.BytesIO(pdf_file_or_bytes)
        else:
            stream = pdf_file_or_bytes
            
        reader = PdfReader(stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
        m_patient = re.search(r"Patient:\s*(.*?)(?=\s*\|)", text)
        m_phase = re.search(r"Phase:\s*(.*?)(?=\s*\|)", text)
        m_segment = re.search(r"Segment:\s*AHA\s*([^\s(]+)\s*\((.*?)\)", text)
        
        m_timi = re.search(r"TIMI Flow Scale:\s*Grade\s*([^\s\n]+)", text)
        m_tfc = re.search(r"TFC \(TIMI Frame Count\):\s*([^\s\n]+)", text)
        m_just = re.search(r"Justification:\s*(.*)", text)
        
        m_ffr = re.search(r"FFR Registered:\s*([^\s|]+)", text)
        m_distal_ffr = re.search(r">50% distal to FFR:\s*([^\s\n]+)", text)
        m_distal_des = re.search(r">50% distal to DES/DCB:\s*([^\s\n]+)", text)
        
        m_prox = re.search(r"Max Proximal Reference:\s*([^\s]+)\s*mm", text)
        m_dist = re.search(r"Max Distal Reference:\s*([^\s]+)\s*mm", text)
        m_ref = re.search(r"(?:Interpolated|Averaged|Custom)\s*Reference:\s*([^\s]+)\s*mm", text)
        m_mld = re.search(r"Minimum Lumen Diameter:\s*([^\s]+)\s*mm", text)
        m_pctD = re.search(r"% Diameter Stenosis:\s*([^\s]+)\s*%", text)
        m_pctA = re.search(r"% Area Stenosis:\s*([^\s]+)\s*%", text)
        m_len = re.search(r"Lesion Length:\s*([^\s]+)\s*mm", text)
        
        if not m_patient:
            return None
            
        def to_float(val):
            if val is None or val == "N/A" or val == "—":
                return "N/A"
            try:
                return float(val)
            except:
                return val
                
        def to_int(val):
            if val is None or val == "N/A" or val == "—":
                return "N/A"
            try:
                return int(val)
            except:
                return val
                
        patient_id = m_patient.group(1).strip()
        phase = m_phase.group(1).strip() if m_phase else "PRE-PCI"
        aha = m_segment.group(1).strip() if m_segment else "N/A"
        vessel = m_segment.group(2).strip() if m_segment else "N/A"
        
        timi = m_timi.group(1).strip() if m_timi else "N/A"
        tfc = m_tfc.group(1).strip() if m_tfc else "N/A"
        just = m_just.group(1).strip() if m_just else "N/A"
        
        ffr = m_ffr.group(1).strip() if m_ffr else "N/A"
        distal_ffr = m_distal_ffr.group(1).strip() if m_distal_ffr else "N/A"
        distal_des = m_distal_des.group(1).strip() if m_distal_des else "N/A"
        
        if phase == "PRE-PCI":
            other_lesion_distal = distal_ffr
        else:
            other_lesion_distal = distal_des
            
        prox = to_float(m_prox.group(1).strip()) if m_prox else "N/A"
        dist = to_float(m_dist.group(1).strip()) if m_dist else "N/A"
        ref = to_float(m_ref.group(1).strip()) if m_ref else "N/A"
        mld = to_float(m_mld.group(1).strip()) if m_mld else "N/A"
        pctD = to_float(m_pctD.group(1).strip()) if m_pctD else "N/A"
        pctA = to_float(m_pctA.group(1).strip()) if m_pctA else "N/A"
        lesion_len = to_float(m_len.group(1).strip()) if m_len else "N/A"
        
        known_occlude = "No"
        if ref == "N/A" or timi == "0":
            known_occlude = "Yes"
            
        return {
            "patient_id": patient_id,
            "phase": phase,
            "aha": aha,
            "vessel": vessel,
            "timi": timi,
            "tfc": tfc,
            "just": just,
            "ffr_registered": ffr,
            "other_lesion_distal": other_lesion_distal,
            "known_occlude": known_occlude,
            "metrics": {
                "prox_diam_mm": prox,
                "dist_diam_mm": dist,
                "ref_diam_mm": ref,
                "mld_mm": mld,
                "pct_diameter_stenosis": pctD,
                "pct_area_stenosis": pctA,
                "lesion_length_mm": lesion_len,
                "timi_grade": timi,
                "tfc": tfc
            }
        }
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None

def get_clean_patient_id(dir_name):
    # Matches patterns like 5010-0052 or XXXX-YYYY
    match = re.match(r"^(\d{4}-\d{4})", dir_name)
    if match:
        return match.group(1)
    return dir_name

def get_raw_folders_for_clean_path(nav_path, base_dir="/mnt/dane_dicom/"):
    """
    Given a nav_path (e.g. "5010/5010-0053" or "5010"), find all matching raw directory paths relative to base_dir.
    Supports arbitrarily deep subdirectories.
    """
    parts = [p for p in nav_path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return [nav_path]
        
    site, patient_id = parts[0], parts[1]
    site_abs = os.path.join(base_dir, site)
    if not os.path.exists(site_abs) or not os.path.isdir(site_abs):
        return [nav_path]
        
    matching_raw = []
    for entry in os.scandir(site_abs):
        if entry.is_dir() and not entry.name.startswith("."):
            if get_clean_patient_id(entry.name) == patient_id:
                subdirs = parts[2:]
                matching_raw.append(os.path.join(site, entry.name, *subdirs))
                
    if not matching_raw:
        return [nav_path]
    return matching_raw

def resolve_clean_path_to_raw(nav_path, base_dir="/mnt/dane_dicom/"):
    """
    Resolves a clean nav_path (e.g. "5010/5010-0053/DICOM") to a real existing raw directory path on disk.
    If multiple exist, returns the first one. If none exist, returns the original path.
    """
    parts = [p for p in nav_path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return nav_path
        
    site, patient_id = parts[0], parts[1]
    site_abs = os.path.join(base_dir, site)
    if not os.path.exists(site_abs) or not os.path.isdir(site_abs):
        return nav_path
        
    resolved_patient_dir = None
    for entry in os.scandir(site_abs):
        if entry.is_dir() and not entry.name.startswith("."):
            if get_clean_patient_id(entry.name) == patient_id:
                resolved_patient_dir = entry.name
                break
                
    if resolved_patient_dir is None:
        return nav_path
        
    subdirs = parts[2:]
    return os.path.join(site, resolved_patient_dir, *subdirs)

def get_cache_file_path(src_fp, base_dir="/mnt/dane_dicom/"):
    """
    Given an absolute source file path on the Tailscale mount (e.g. '/mnt/dane_dicom/2200/2200-0001-raw/ANGIO/I0'),
    resolves the corresponding local cache destination path (e.g. './tailscale_cache/2200/2200-0001/ANGIO/I0').
    """
    base_abs = os.path.abspath(base_dir)
    src_abs = os.path.abspath(src_fp)
    try:
        rel = os.path.relpath(src_abs, base_abs)
    except Exception:
        rel = os.path.basename(src_abs)
        
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        parts[1] = get_clean_patient_id(parts[1])
    clean_rel = "/".join(parts)
    return os.path.abspath(os.path.join("./tailscale_cache", clean_rel))

def check_cache_status_and_heal(loc, base_dir="/mnt/dane_dicom/"):
    """
    Checks if the cache for clean path `loc` (e.g. '2200/2200-0001' or '2200/2200-0001/ANGIO') is complete.
    If the sentinel file `.cache_complete` exists, returns True.
    If it doesn't exist but the cache folder exists:
      Runs a comparison check against the source files.
      If complete, writes `.cache_complete` and returns True.
      Otherwise, returns False.
    If the cache folder doesn't exist, returns False.
    """
    dst_abs = os.path.abspath(os.path.join("./tailscale_cache", loc))
    sentinel_path = os.path.join(dst_abs, ".cache_complete")
    
    if os.path.exists(sentinel_path):
        return True
        
    if os.path.exists(dst_abs) and os.path.isdir(dst_abs):
        if not os.path.exists(base_dir):
            # If Tailscale is offline/unmounted, check if we have any files inside
            for root, dirs, files in os.walk(dst_abs):
                if any(not f.startswith(".") for f in files):
                    return True
            return False
            
        r_locs = get_raw_folders_for_clean_path(loc, base_dir)
        source_files = []
        for r_loc in r_locs:
            src_abs = os.path.abspath(os.path.join(base_dir, r_loc))
            if os.path.exists(src_abs) and os.path.isdir(src_abs):
                for root, dirs, files in os.walk(src_abs):
                    for file in files:
                        if not file.startswith("."):
                            source_files.append(os.path.join(root, file))
                            
        if not source_files:
            return False
            
        all_match = True
        for src_fp in source_files:
            dst_fp = get_cache_file_path(src_fp, base_dir)
            if not os.path.exists(dst_fp):
                all_match = False
                break
            try:
                if os.path.getsize(src_fp) != os.path.getsize(dst_fp):
                    all_match = False
                    break
            except Exception:
                all_match = False
                break
                
        if all_match:
            try:
                os.makedirs(dst_abs, exist_ok=True)
                with open(sentinel_path, "w") as f:
                    f.write("completed")
                return True
            except Exception:
                pass
                
    return False

def robust_copy(src, dst):
    import shutil
    try:
        shutil.copy2(src, dst)
    except OSError as e:
        # Fallback to manual chunk-based copy if sendfile/copy_file_range fails (e.g. CIFS/SMB share bug)
        try:
            with open(src, 'rb') as fsrc:
                with open(dst, 'wb') as fdst:
                    while True:
                        buf = fsrc.read(1024 * 1024)
                        if not buf:
                            break
                        fdst.write(buf)
            try:
                shutil.copystat(src, dst)
            except Exception:
                pass
        except Exception:
            raise e

def auto_detect_angio_subfolder(nav_path, base_dir="/mnt/dane_dicom/"):
    """
    If the given nav_path has an ANGIO subfolder (case-insensitive) on disk,
    returns the nav_path with the actual case of the ANGIO folder name appended.
    Otherwise returns the original nav_path.
    """
    if not nav_path:
        return nav_path
    resolved = resolve_clean_path_to_raw(nav_path, base_dir)
    abs_path = os.path.abspath(os.path.join(base_dir, resolved))
    if os.path.exists(abs_path) and os.path.isdir(abs_path):
        try:
            for entry in os.scandir(abs_path):
                if entry.is_dir() and entry.name.upper() == "ANGIO":
                    return os.path.join(nav_path, entry.name)
        except Exception as e:
            print(f"[DEBUG] Error checking for ANGIO subfolder in {abs_path}: {e}")
    return nav_path

@st.cache_data(ttl=3600, show_spinner=False)
def scan_all_tailscale_patients():
    base_dir = "/mnt/dane_dicom/"
    patients = []
    if not os.path.exists(base_dir):
        return patients
    try:
        seen = set()
        for site in os.listdir(base_dir):
            site_path = os.path.join(base_dir, site)
            if os.path.isdir(site_path) and not site.startswith("."):
                for patient in os.listdir(site_path):
                    patient_path = os.path.join(site_path, patient)
                    if os.path.isdir(patient_path) and not patient.startswith("."):
                        clean_pid = get_clean_patient_id(patient)
                        # We only count directories matching XXXX-YYYY as valid patients
                        if re.match(r"^\d{4}-\d{4}$", clean_pid):
                            key = (site, clean_pid)
                            if key not in seen:
                                seen.add(key)
                                patients.append({
                                    "site": site,
                                    "patient_id": clean_pid
                                })
    except Exception as e:
        print(f"Error scanning tailscale patients: {e}")
    return patients

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_analysts():
    analysts = []
    if not st.session_state.firebase_init:
        return analysts
    try:
        from firebase_admin import auth
        page = auth.list_users()
        while page:
            for user in page.users:
                if user.email:
                    analysts.append({
                        "uid": user.uid,
                        "email": user.email,
                        "name": user.display_name or user.email.split("@")[0],
                        "username": user.email.split("@")[0]
                    })
            page = page.get_next_page()
    except Exception as e:
        print(f"Error listing analysts: {e}")
    return analysts

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_completed_pids():
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return set()
    try:
        db = st.session_state.firestore_db
        reports_ref = db.collection("analysis_results").where("phase", "==", "COMPLETED").stream()
        return {r.to_dict().get("patient_id") for r in reports_ref if r.to_dict().get("patient_id")}
    except Exception as e:
        print(f"Error fetching completed pids: {e}")
        return set()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_analyst_assignments(username):
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return set(), set(), set()
    try:
        db = st.session_state.firestore_db
        # 1. Fetch site assignments for this analyst
        site_assign_ref = db.collection("site_assignments").where("assigned_to", "==", username).stream()
        assigned_sites = {doc.id for doc in site_assign_ref}
        
        # 2. Fetch individual case assignments and unassigned cases
        assign_ref = db.collection("assignments").stream()
        assigned_cases = set()
        unassigned_cases = set()
        for doc in assign_ref:
            data = doc.to_dict()
            pid = data.get("patient_id")
            site = data.get("site")
            status = data.get("status")
            assigned_to = data.get("assigned_to")
            
            if status == "unassigned" or assigned_to == "unassigned":
                if pid:
                    unassigned_cases.add(pid)
            elif assigned_to == username:
                if pid:
                    assigned_cases.add(pid)
                if site:
                    assigned_sites.add(site)
        return assigned_sites, assigned_cases, unassigned_cases
    except Exception as e:
        print(f"Error fetching analyst assignments: {e}")
        return set(), set(), set()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_completed_reports():
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return []
    try:
        db = st.session_state.firestore_db
        reports_ref = db.collection("analysis_results").stream()
        return [r.to_dict() for r in reports_ref]
    except Exception as e:
        print(f"Error fetching completed reports: {e}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_analyst_assignments_detail(username):
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return [], []
    try:
        db = st.session_state.firestore_db
        # Fetch my individual assignments
        assign_ref = db.collection("assignments").where("assigned_to", "==", username).stream()
        assignments = [a.to_dict() for a in assign_ref]
        
        # Fetch my site assignments
        site_assign_ref = db.collection("site_assignments").where("assigned_to", "==", username).stream()
        site_assignments = []
        for doc in site_assign_ref:
            data = doc.to_dict()
            data["site"] = doc.id
            site_assignments.append(data)
        return assignments, site_assignments
    except Exception as e:
        print(f"Error fetching analyst assignments detail: {e}")
        return [], []

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_admin_data():
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return [], [], []
    try:
        db = st.session_state.firestore_db
        # Fetch completed reports
        reports_ref = db.collection("analysis_results").stream()
        completed_reports = [r.to_dict() for r in reports_ref]
        
        # Fetch individual assignments
        assign_ref = db.collection("assignments").stream()
        assignments = [a.to_dict() for a in assign_ref]
        
        # Fetch site assignments
        site_assign_ref = db.collection("site_assignments").stream()
        site_assignments = []
        for doc in site_assign_ref:
            data = doc.to_dict()
            data["site"] = doc.id
            site_assignments.append(data)
            
        return completed_reports, assignments, site_assignments
    except Exception as e:
        print(f"Error fetching admin data: {e}")
        return [], [], []

def get_completed_pids_dict(completed_reports):
    completed_pids = {}
    for r in completed_reports:
        pid = r.get("patient_id")
        if not pid:
            continue
            
        phase = r.get("phase")
        if phase == "COMPLETED" or r.get("dicom_name") == "marked_completed":
            if pid not in completed_pids or phase == "COMPLETED" or r.get("dicom_name") == "marked_completed":
                completed_pids[pid] = r
                
    return completed_pids

@st.cache_data(ttl=600, show_spinner=False)
def get_patient_statuses_map():
    if not st.session_state.get("firebase_init"):
        return {}
    completed_reports = get_cached_completed_reports()
    completed_pids = get_completed_pids_dict(completed_reports)
    
    patient_statuses = {}
    for pid in completed_pids:
        patient_statuses[pid] = "🟢"
        
    for r in completed_reports:
        pid = r.get("patient_id")
        if not pid:
            continue
        if pid not in patient_statuses:
            patient_statuses[pid] = "🟡"
            
    return patient_statuses

def clear_db_caches():
    get_cached_completed_pids.clear()
    get_cached_analyst_assignments.clear()
    get_cached_completed_reports.clear()
    get_cached_analyst_assignments_detail.clear()
    get_cached_admin_data.clear()
    get_patient_statuses_map.clear()

@st.cache_data(ttl=86400, show_spinner=False)
def estimate_patient_download_size(site, patient_id, base_dir="/mnt/dane_dicom/"):
    import re
    total_bytes = 0
    clean_path = f"{site}/{patient_id}"
    r_locs = get_raw_folders_for_clean_path(clean_path, base_dir)
    for r_loc in r_locs:
        r_path = os.path.join(base_dir, r_loc)
        if os.path.exists(r_path):
            has_angio = False
            try:
                for entry in os.scandir(r_path):
                    if entry.is_dir() and entry.name.upper() == "ANGIO":
                        has_angio = True
                        break
            except:
                pass
            for root, dirs, files in os.walk(r_path):
                for file in files:
                    if not file.startswith('.'):
                        src_fp = os.path.join(root, file)
                        if has_angio:
                            rel_to_patient = os.path.relpath(src_fp, r_path)
                            rel_parts_upper = [part.upper() for part in rel_to_patient.replace("\\", "/").split("/")]
                            if "ANGIO" not in rel_parts_upper:
                                continue
                        try:
                            total_bytes += os.path.getsize(src_fp)
                        except:
                            pass
    return total_bytes

def format_patient_folder_option(option):
    if option == "-- Select Subfolder --":
        return option
    clean_pid = get_clean_patient_id(option)
    if re.match(r"^\d{4}-\d{4}$", clean_pid):
        status_map = get_patient_statuses_map()
        status_emoji = status_map.get(clean_pid, "⚪")
        if status_emoji == "🟢":
            return f"🟢 {option} (complete)"
        elif status_emoji == "🟡":
            return f"🟡 {option} (in progress)"
        else:
            return f"⚪ {option} (to do)"
    return option

def is_patient_cached(site, patient_id):
    return check_cache_status_and_heal(f"{site}/{patient_id}")

def get_master_pdf_bytes(patient_id):
    if not st.session_state.get("firebase_init") or st.session_state.get("firestore_db") is None:
        return None
    import pypdf
    import io
    import requests
    
    try:
        db = st.session_state.firestore_db
        reports_ref = db.collection("analysis_results")\
                        .where("patient_id", "==", patient_id)\
                        .stream()
                        
        reports = [r.to_dict() for r in reports_ref]
        reports = [r for r in reports if r.get("dicom_name") != "marked_completed" and r.get("phase") != "COMPLETED"]
        
        if not reports:
            return None
            
        reports.sort(key=lambda x: x.get("timestamp") or 0)
        
        merger = pypdf.PdfWriter()
        has_pages = False
        
        for r in reports:
            pdf_url = r.get("pdf_url")
            if not pdf_url:
                continue
                
            pdf_bytes = None
            import urllib.parse
            pdf_filename = pdf_url.split("/")[-1].split("?")[0]
            pdf_filename = urllib.parse.unquote(pdf_filename)
            if "reports/" in pdf_filename:
                pdf_filename = pdf_filename.split("reports/")[-1]
                
            local_dir = st.session_state.get("report_save_dir", os.path.abspath("./reports/"))
            local_path = os.path.join(local_dir, pdf_filename)
            
            if os.path.exists(local_path):
                try:
                    with open(local_path, "rb") as f:
                        pdf_bytes = f.read()
                except Exception:
                    pass
                    
            if not pdf_bytes:
                try:
                    resp = requests.get(pdf_url, timeout=10)
                    if resp.status_code == 200:
                        pdf_bytes = resp.content
                except Exception as e:
                    print(f"Error downloading PDF {pdf_url}: {e}")
                    
            if pdf_bytes:
                try:
                    merger.append(io.BytesIO(pdf_bytes))
                    has_pages = True
                except Exception as e:
                    print(f"Error appending PDF page: {e}")
                    
        if has_pages:
            out = io.BytesIO()
            merger.write(out)
            merger.close()
            return out.getvalue()
    except Exception as e:
        print(f"Error generating master PDF: {e}")
    return None

def _prefetch_cases_worker(cases, task_id):
    import time
    import shutil
    import os
    import re
    import threading
    
    # Trigger eCRF extraction for all cases in background threads
    try:
        login_val = "roledert"
        pass_val = "Troleder79!"
        for c in cases:
            pid = c.get("patient_id", "").strip()
            if pid and re.match(r"^\d{4}-\d{4}$", pid):
                ecrf_data_dir = os.path.abspath("./ecrf_data")
                os.makedirs(ecrf_data_dir, exist_ok=True)
                json_path = os.path.join(ecrf_data_dir, f"result_{pid}.json")
                if not os.path.exists(json_path):
                    rc = {"data": None, "error": None}
                    threading.Thread(
                        target=_run_extraction,
                        args=(login_val, pass_val, pid, rc),
                        daemon=True
                    ).start()
    except Exception:
        pass

    base_dir = "/mnt/dane_dicom/"
    
    try:
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "running",
                "copied_files": 0,
                "total_files": len(cases),
                "copied_bytes": 0,
                "total_bytes": 0,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": "Scanning files on Tailscale..."
            }
            
        case_folders = []
        for c in cases:
            site = c["site"]
            pid = c["patient_id"]
            clean_path = f"{site}/{pid}"
            r_locs = get_raw_folders_for_clean_path(clean_path, base_dir)
            src_abs_list = [os.path.abspath(os.path.join(base_dir, r)) for r in r_locs]
            dst_abs = os.path.abspath(os.path.join("./tailscale_cache", clean_path))
            case_folders.append((src_abs_list, dst_abs))
            
        files_to_copy = []
        total_size_bytes = 0
        
        for src_abs_list, dst_abs in case_folders:
            for s_path in src_abs_list:
                if not os.path.exists(s_path):
                    continue
                s_name = os.path.basename(s_path)
                
                # Check if an ANGIO subfolder (case-insensitive) exists under this patient folder
                has_angio = False
                try:
                    for entry in os.scandir(s_path):
                        if entry.is_dir() and entry.name.upper() == "ANGIO":
                            has_angio = True
                            break
                except Exception:
                    pass

                for root, dirs, files in os.walk(s_path):
                    for file in files:
                        if file.startswith('.'):
                            continue
                        src_fp = os.path.join(root, file)
                        
                        # Restrict to files under the ANGIO subfolder if it exists
                        if has_angio:
                            rel_to_patient = os.path.relpath(src_fp, s_path)
                            rel_parts_upper = [p.upper() for p in rel_to_patient.replace("\\", "/").split("/")]
                            if "ANGIO" not in rel_parts_upper:
                                continue

                        try:
                            sz = os.path.getsize(src_fp)
                        except Exception:
                            sz = 0
                        dst_fp = get_cache_file_path(src_fp, base_dir)
                        files_to_copy.append((src_fp, dst_fp, sz))
                        total_size_bytes += sz
                        
        total_files = len(files_to_copy)
        if total_files == 0:
            with _copy_tasks_lock:
                _active_copy_tasks[task_id] = {
                    "status": "success",
                    "total_files": 0,
                    "total_bytes": 0
                }
            return
            
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "running",
                "copied_files": 0,
                "total_files": total_files,
                "copied_bytes": 0,
                "total_bytes": total_size_bytes,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": ""
            }
            
        copied_bytes = 0
        start_time = time.time()
        
        for idx, (src_fp, dst_fp, sz) in enumerate(files_to_copy):
            os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
            robust_copy(src_fp, dst_fp)
            copied_bytes += sz
            
            elapsed = time.time() - start_time
            speed_mb = 0.0
            if elapsed > 0:
                speed_mb = (copied_bytes / (1024 * 1024)) / elapsed
                
            est_left = 0.0
            if copied_bytes > 0:
                bytes_left = total_size_bytes - copied_bytes
                est_left = bytes_left / (copied_bytes / elapsed)
                
            with _copy_tasks_lock:
                _active_copy_tasks[task_id] = {
                    "status": "running",
                    "copied_files": idx + 1,
                    "total_files": total_files,
                    "copied_bytes": copied_bytes,
                    "total_bytes": total_size_bytes,
                    "speed": speed_mb,
                    "est_left": est_left,
                    "start_time": start_time
                }
                
        for _, dst_abs in case_folders:
            try:
                now = time.time()
                os.utime(dst_abs, (now, now))
                sentinel_path = os.path.join(dst_abs, ".cache_complete")
                with open(sentinel_path, "w") as f:
                    f.write("completed")
            except Exception:
                pass
                
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "success",
                "copied_files": total_files,
                "total_files": total_files,
                "copied_bytes": total_size_bytes,
                "total_bytes": total_size_bytes,
                "speed": 0.0,
                "est_left": 0.0
            }
    except Exception as e:
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "error",
                "error_msg": str(e)
            }

@st.cache_data(ttl=15, show_spinner=False)
def get_recursive_preview_files_cached(nav_path, base_dir):
    r_locs = get_raw_folders_for_clean_path(nav_path, base_dir)
    files_found = []
    max_preview_files = 100
    total_files_count = 0
    
    for r_loc in r_locs:
        r_path = os.path.abspath(os.path.join(base_dir, r_loc))
        if os.path.exists(r_path):
            # Recursive scan using os.walk to find nested DICOM files
            for root, dirs, files in os.walk(r_path):
                for file in files:
                    if not file.startswith("."):
                        total_files_count += 1
                        if len(files_found) < max_preview_files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, r_path)
                            try:
                                sz_bytes = os.path.getsize(full_path)
                                sz_str = f"({sz_bytes / (1024*1024):.2f} MB)"
                            except Exception:
                                sz_str = ""
                            files_found.append((rel_path, sz_str, full_path))
    return files_found, total_files_count

@st.dialog("Finish Patient Analysis")
def confirm_finish_dialog():
    patient_id = st.session_state.get("patient_id", "").strip()
    st.write(f"Are you sure you want to finish the analysis for patient **{patient_id}** and mark them as completed?")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Finish", use_container_width=True, type="primary"):
            # Mark as completed in Firebase if we have database connection
            if st.session_state.firebase_init and st.session_state.firestore_db:
                try:
                    db = st.session_state.firestore_db
                    username = st.session_state.user.get("username") if st.session_state.user else "unknown"
                    
                    # 1. Compile consolidated Master PDF
                    pdf_bytes = get_master_pdf_bytes(patient_id)
                    if not pdf_bytes and len(st.session_state.patient_cart) > 0:
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
                        pdf_bytes = masterPdfBuf.getvalue()

                    pdf_url = ""
                    if pdf_bytes:
                        pdf_filename = f"Patient_{patient_id}_Master.pdf"
                        pdf_url = upload_pdf_to_firebase(pdf_bytes, pdf_filename)

                    # Check if already completed
                    check_ref = db.collection("analysis_results")\
                                  .where("patient_id", "==", patient_id)\
                                  .where("phase", "==", "COMPLETED")\
                                  .limit(1).stream()
                    already_done = len(list(check_ref)) > 0
                    
                    if not already_done:
                        # Prepare the completed marker document
                        doc_data = {
                            "patient_id": patient_id,
                            "dicom_name": "marked_completed",
                            "phase": "COMPLETED",
                            "vessel": "NONE",
                            "aha": "NONE",
                            "ffr_registered": "N/A",
                            "other_lesion_distal": "N/A",
                            "known_occlude": "N/A",
                            "pdf_url": pdf_url,
                            "timestamp": firestore.SERVER_TIMESTAMP,
                            "analyst": username,
                            "metrics": {}
                        }
                        db.collection("analysis_results").add(doc_data)
                    st.success("Patient marked as completed in database.")
                    clear_db_caches()
                except Exception as e:
                    st.error(f"Error updating database: {e}")
                    time.sleep(1.5)
            
            # Reset patient analysis state
            keys_to_preserve = [
                'firebase_init', 'firebase_error', 'firestore_db', 'firebase_bucket', 
                'firebase_api_key', 'user', 'app_mode', 'app_mode_select'
            ]
            for key in list(st.session_state.keys()):
                if key not in keys_to_preserve:
                    del st.session_state[key]
            
            st.rerun()
            
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


def filter_subdirs_by_assignments(nav_path, raw_dirs):
    if not st.session_state.get("firebase_init") or not st.session_state.user:
        return sorted(raw_dirs)
        
    username = st.session_state.user.get("username")
    role = st.session_state.user.get("role")
    
    print(f"[DEBUG] filter_subdirs_by_assignments: nav_path='{nav_path}', username='{username}', role='{role}'")
    
    try:
        parts = [p for p in nav_path.replace("\\", "/").split("/") if p]
        
        assigned_sites, assigned_cases, unassigned_cases = get_cached_analyst_assignments(username)
        print(f"[DEBUG] User assignments - assigned_sites: {assigned_sites}, assigned_cases: {assigned_cases}, unassigned: {unassigned_cases}")
                 
        if len(parts) == 0: # Root level, subfolders are sites
            return sorted([d for d in raw_dirs if d in assigned_sites])
        elif len(parts) == 1: # Site level, subfolders are patient folders
            site = parts[0]
            site_is_assigned_to_user = site in assigned_sites
            
            unique_clean_pids = set()
            for d in raw_dirs:
                clean_pid = get_clean_patient_id(d)
                if re.match(r"^\d{4}-\d{4}$", clean_pid):
                    if clean_pid in unassigned_cases:
                        continue
                    if site_is_assigned_to_user or (clean_pid in assigned_cases):
                        unique_clean_pids.add(clean_pid)
            print(f"[DEBUG] User unique_clean_pids count for site {site}: {len(unique_clean_pids)}")
            return sorted(list(unique_clean_pids))
            
    except Exception as e:
        print(f"Error filtering subdirs: {e}")
        
    return sorted(raw_dirs)

def generate_instructions_pdf():
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            canvas.Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_elements(num_pages)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def draw_page_elements(self, page_count):
            self.saveState()
            
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#00ff00"))
            self.drawString(54, 750, "AngioPy Segmentation Application")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#777777"))
            self.drawRightString(558, 750, "Official User & Technical Manual")
            
            self.setStrokeColor(colors.HexColor("#00ff00"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
            # Footer
            self.line(54, 50, 558, 50)
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#777777"))
            self.drawString(54, 38, "CONFIDENTIAL - CLINICAL & RESEARCH USE ONLY")
            
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 38, page_text)
            
            self.restoreState()

    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles (dark mode / clinical accents)
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#00ff00"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    story = []
    
    story.append(Spacer(1, 40))
    story.append(Paragraph("AngioPy Segmentation Manual", title_style))
    story.append(Paragraph("Consolidated Operations, Diagnostics, Caching, and QCA Analysis Instructions", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Welcome to the official manual for <b>AngioPy Segmentation</b>, a premium quantitative coronary angiography (QCA) platform. This guide explains all key modules, workflows, and procedures.", body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1. Analyst Authentication & Session Bootstrapping", h1_style))
    story.append(Paragraph("The platform is secured by Firebase Authentication. Analysts must log in to view assignments and synchronize records. When first accessing the app, administrators can configure default analysts via the Admin Panel.", body_style))
    story.append(Paragraph("• <b>Default Credentials:</b> If offline or first initialization, the system uses safe hashed authentication checks.", bullet_style))
    story.append(Paragraph("• <b>User Security:</b> Passwords are encrypted server-side using SHA-256 with specific analyst salts.", bullet_style))
    story.append(Paragraph("• <b>Session Persistence:</b> Session tokens are persisted browser-side until an explicit logout is triggered.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. DICOM Folder Browser & Cache Importer", h1_style))
    story.append(Paragraph("To optimize load speeds, DICOM cases are cached on the VPS disk. The <b>Local Cache Importer</b> sidebar panel offers multiple ways to buffer patient files:", body_style))
    story.append(Paragraph("• <b>🌐 Tailscale Shared Folder:</b> Navigate the remote Tailscale mount directly. Emojis represent case progress: ⚪ (To Do), 🟡 (In Progress), 🟢 (Complete). Click <i>Cache Current Folder</i> to copy to VPS local cache in a thread-safe background process.", bullet_style))
    story.append(Paragraph("• <b>🚀 Mass Prefetch Cases:</b> Multiselect assigned patient cases on Tailscale. The system estimates size, checks against a 100 GB limit, and runs copies sequentially in a background thread with live speed tracking.", bullet_style))
    story.append(Paragraph("• <b>💻 Upload Local Files:</b> Upload individual DICOM files or drag whole directories (via browser JS folder injection). The folder name is automatically detected from DICOM headers.", bullet_style))
    story.append(Paragraph("• <b>📦 Upload ZIP Archive:</b> Upload ZIP compressed cases. The server extracts files on the fly showing an extraction progress bar.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("3. DICOM Series Selection Grid", h1_style))
    story.append(Paragraph("Once a patient folder is loaded into cache, the workspace switches to the Selection Grid mode:", body_style))
    story.append(Paragraph("• <b>Frame Auto-Detection:</b> The app scans frames and automatically identifies the optimal contrast frame containing peak opacification.", bullet_style))
    story.append(Paragraph("• <b>Sequence Configuration:</b> Check <i>Chosen for Analysis</i> to select sequences. Tag the Procedure Phase (PRE-PCI or POST-PCI), Vessel System (LAD, LCx, RCA) and Segment (AHA classification code).", bullet_style))
    story.append(Paragraph("• <b>Animation Control:</b> Click <i>🎥 Play</i> to run the angiographic cine-loop. Click <i>⏹️ Stop</i> to freeze.", bullet_style))
    story.append(Paragraph("• <b>Direct Download:</b> Use <i>📥 Download DICOM</i> to download the raw DICOM file directly from the VPS cache to support workstation analysis.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. QCA Interactive Analysis Workspace", h1_style))
    story.append(Paragraph("Clicking <i>Analyze (QCA)</i> loads the interactive workstation containing advanced algorithmic and manual tools:", body_style))
    story.append(Paragraph("• <b>Centerline and Contour Detection:</b> The system runs automatic edge detection. Manual sliders allow tweaking landmarks: Proximal Reference, Distal Reference, and Minimum Lumen Diameter (MLD) positions.", bullet_style))
    story.append(Paragraph("• <b>Edge Reference Modes:</b> Analysts can choose between Interpolated Reference (centerline arc-length cumLen) or Mean, Max, and Manual reference computation.", bullet_style))
    story.append(Paragraph("• <b>PCI Options:</b> Adjust FFR wire indicators, other distal lesions (>50% distal to FFR/DES), and procedure phase offsets.", bullet_style))
    story.append(Paragraph("• <b>Save Sequence:</b> Generate the sequence report. The PDF is saved locally, and numerical outputs are synchronized to the Firestore <i>analysis_results</i> collection.", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("5. Saved Stenoses Checklist & Master PDF Persistence", h1_style))
    story.append(Paragraph("• <b>Saved Checklist:</b> A widget reads all reports from Firestore under the current patient ID, ensuring that all necessary vessel segments have been analyzed prior to completion.", bullet_style))
    story.append(Paragraph("• <b>Persistent Master PDF:</b> Clicking <i>Export Master Patient PDF</i> consolidates all saved individual sequence PDFs on-the-fly, allowing session restoration after refreshing.", bullet_style))
    story.append(Paragraph("• <b>Finish Patient Analysis:</b> Confirms session closure. The consolidated PDF is uploaded to Firebase Storage and the patient ID is marked as Completed (🟢).", bullet_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("6. Administrator Capabilities (👑 Admin Panel)", h1_style))
    story.append(Paragraph("Administrators can navigate to the Admin Panel to oversee operations:", body_style))
    story.append(Paragraph("• <b>Analyst Assignments:</b> Assign cases or entire sites (hospitals/centers) to specific analysts.", bullet_style))
    story.append(Paragraph("• <b>Progress Metrics:</b> Access database statistics, disk file metrics, and individual analyst completion rates.", bullet_style))
    story.append(Paragraph("• <b>Historical Import:</b> Bulk import historical PDF reports. The app parses text via regex to backfill Firestore values and uploads PDFs to Storage.", bullet_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    return pdf_buffer.getvalue()

def render_instructions_panel():
    st.markdown("<h1 style='color: #00ff00; font-family: Outfit, sans-serif;'>📖 User & Technical Instructions</h1>", unsafe_allow_html=True)
    st.markdown("Detailed guide and operation instructions for all components of the AngioPy Segmentation application.")
    
    # PDF Manual download button
    try:
        pdf_bytes = generate_instructions_pdf()
        st.download_button(
            "📥 Download PDF Manual",
            data=pdf_bytes,
            file_name="AngioPy_User_Manual.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            key="btn_download_instructions_pdf"
        )
    except Exception as e:
        st.error(f"Error compiling PDF Manual: {e}")
        
    st.markdown("---")
    
    with st.expander("🔐 1. Analyst Authentication & User Management", expanded=True):
        st.markdown("""
        ### Authentication Workflow
        The application is secured by **Firebase Authentication** to restrict access to authorized clinical analysts.
        - **Login Screen**: Enter your email/username and password. If the database is online, authentication is verified via Firebase cloud servers.
        - **Administrator Privileges**: Only users designated with the `admin` role can access the **👑 Admin Panel** and utilize **Analyst Management** in the sidebar.
        - **Bootstrap Registration**: Admins can register new analyst accounts directly from the sidebar user manager.
        - **Default/Offline Session**: If Firebase is offline, a local fallback mode is initialized with warning prompts.
        """)
        
    with st.expander("📥 2. DICOM Local Cache Importer & Pre-fetching", expanded=False):
        st.markdown("""
        ### Caching and Pre-fetching
        AngioPy utilizes local server disk storage (`./tailscale_cache`) to store patient DICOMs. This isolates analysts from weak internet connections or remote drive speed bottlenecks.
        
        #### Import Modes:
        1. **🌐 Tailscale Shared Folder**: 
           Navigate the raw Tailscale mount directly. Folders are decorated with status coloring emojis:
           * `⚪` **To Do**: No database records exist for this patient ID.
           * `🟡` **In Progress**: QCA report records exist, but analysis has not been finalized.
           * `🟢` **Complete**: Patient analysis has been explicitly marked as completed.
           * *Caching*: Click **Cache Current Folder** to transfer the directory to local VPS disk cache.
        2. **🚀 Mass Prefetch Cases**:
           Lists all assigned cases not yet cached. Select cases, view consolidated size (100 GB limit enforced), and click **Start Pre-fetching** to run copies in a thread-safe background task.
        3. **💻 Upload Local Files**:
           Select or drag a patient folder from your computer. HTML5 folder upload injection is supported. The destination directory name is auto-detected from DICOM headers.
        4. **📦 Upload ZIP Archive**:
           Upload a zipped patient case. The server handles extraction on the fly.
        """)
        
    with st.expander("🖥️ 3. Series Selection Grid & Sequence Setup", expanded=False):
        st.markdown("""
        ### Workspace Preparation
        Once a patient is selected, all sequences are scanned:
        - **Contrast Auto-detection**: The system parses the pixel frames, runs peak opacification algorithms, and selects the best frame containing maximal contrast filling.
        - **Sequence Card Settings**:
          * **Chosen for Analysis**: Check this to assign the sequence to the active cart.
          * **Procedure Phase**: Tag as `PRE-PCI` or `POST-PCI`.
          * **Vessel & AHA Segments**: Inline selectboxes allow mapping to specific vessels (LAD, LCx, RCA) and AHA segment codes.
          * **FFR Status**: Record if an FFR wire is registered in the frame.
          * **TIMI Flow Grade**: Displays calculated TIMI grade and justification, with manual override capability.
          * **Cine Animation**: Click **Play** to animate the movie clip frame loop, and **Stop** to freeze.
          * **Raw Download**: Click **Download DICOM** to retrieve the original file from the VPS cache.
        """)

    with st.expander("🔬 4. QCA Workstation & Landmark Corrections", expanded=False):
        st.markdown("""
        ### Interactive QCA Workstation
        Clicking **Analyze (QCA)** loads the main segmentation screen:
        - **Centerline & Contours**: Centerline, vessel diameter profile, and contours are plotted in real time.
        - **Visual Adjustments**: Correct contrast limits and pixel depth on the fly.
        - **Manual Slide Corrections**:
          * Use sliders to move the **Max Proximal Reference**, **Max Distal Reference**, and **Minimum Lumen Diameter (MLD)** landmarks along the vessel centerline.
          * Use the **Swap direction** check to flip the analysis beginning/end nodes.
        - **Reference Estimation Methods**:
          * Choose between *Interpolated* (centerline length-based), *Mean*, *Max*, or *Manual* reference mode.
        - **Save Sequence**: Click **Save PDF Report** to commit the sequence. Numerical values are saved to Firestore, and a PDF is generated.
        """)

    with st.expander("📋 5. Saved Checklist, Session Persistence & Closing Case", expanded=False):
        st.markdown("""
        ### Patient Closure Workflow
        - **📋 Saved Checklist**: Displayed under patient ID. Shows all reports in Firestore for the active patient, ensuring no vessel segments (e.g. PRE vs POST) are missed.
        - **Persistent Master PDF**: Rebuilds session state on the fly. Merges saved individual sequence report PDFs using Firestore data, even after refreshes or logouts.
        - **🏁 Finish Patient Analysis**:
          * Merges saved sequence reports.
          * Uploads the final master report to Firebase Storage.
          * Marks the patient status as completed (`🟢`) in the database.
          * Resets session state to prepare the workspace for the next case without needing to log out.
        """)

    with st.expander("👑 6. Administrator & Progress Controls", expanded=False):
        st.markdown("""
        ### Admin Panel Capabilities
        Admin accounts can navigate to **Application Mode -> Admin Panel**:
        - **Case Assignments**: Assign individual cases or whole sites (e.g., center codes) to specific analysts.
        - **Analytics Metrics**: Track case completion rates, disk space stats, and individual analyst logs.
        - **Bulk Imports**: Upload legacy consolidated QCA report PDFs to parse metrics and upload documents.
        """)

@st.fragment(run_every="4s")
def render_server_caching_progress():
    import time
    st.empty()  # Always render an element to ensure the fragment mounts in the frontend
    with _copy_tasks_lock:
        active_tasks = list(_active_copy_tasks.items())
        
    # Print status to stdout so it appears in the systemd logs (helpful for tracking/debugging)
    if active_tasks:
        print(f"[PROGRESS UI] Active copy tasks in UI render loop: {[(tid, t.get('status'), t.get('copied_files'), t.get('total_files')) for tid, t in active_tasks]}")
        
    if not active_tasks:
        return
        
    st.markdown("#### ⏳ Server Caching Progress")
    
    for tid, task in active_tasks:
        display_id = tid
        if tid.startswith("prefetch_admin_site_"):
            site_id = tid.split("_")[3]
            display_id = f"Caching Site {site_id}"
        elif tid.startswith("prefetch_admin_patient_"):
            parts = tid.split("_")
            if len(parts) >= 6:
                patient_id = f"{parts[3]}-{parts[4]}"
            else:
                patient_id = parts[3]
            display_id = f"Caching Patient {patient_id}"
        elif tid.startswith("prefetch_"):
            display_id = "Mass Prefetch Cases"
        elif "tailscale_cache" in tid:
            parts = tid.replace("\\", "/").split("/")
            if len(parts) == 1:
                parts = tid.split("_")
            try:
                if "tailscale" in parts and "cache" in parts:
                    t_idx = parts.index("tailscale")
                    if t_idx + 1 < len(parts) and parts[t_idx+1] == "cache":
                        rem = parts[t_idx+2:]
                        if len(rem) >= 2:
                            display_id = f"Importing Case {rem[0]}/{rem[1]}"
                        elif len(rem) == 1:
                            display_id = f"Importing Folder {rem[0]}"
                elif "tailscale_cache" in parts:
                    idx = parts.index("tailscale_cache")
                    if idx + 2 < len(parts):
                        display_id = f"Importing Case {parts[idx+1]}/{parts[idx+2]}"
                    elif idx + 1 < len(parts):
                        display_id = f"Importing Folder {parts[idx+1]}"
            except Exception:
                pass
            
        if task.get("status") == "running":
            copied_files = task.get("copied_files", 0)
            total_files = task.get("total_files", 0)
            copied_bytes = task.get("copied_bytes", 0)
            total_bytes = task.get("total_bytes", 0)
            speed = task.get("speed", 0.0)
            est_left = task.get("est_left", 0.0)
            detail = task.get("detail", "")
            
            percent = int(copied_files / total_files * 100) if total_files > 0 else 0
            copied_mb = copied_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            
            st.markdown(f"⏳ **Active Task:** {display_id}")
            
            if detail == "Scanning files on Tailscale...":
                st.info("🔍 Scanning remote directory on Tailscale... (This can take up to 30 seconds)")
                st.progress(0.0)
            else:
                time_str = "Calculating..."
                if est_left > 60:
                    time_str = f"{int(est_left // 60)}m {int(est_left % 60)}s"
                elif est_left > 0:
                    time_str = f"{int(est_left)}s"
                elif copied_files > 0:
                    time_str = "0s"
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Progress (Percentage)", f"{percent}%")
                col_m2.metric("Files Copied", f"{copied_files} / {total_files} ({copied_mb:.1f} / {total_mb:.1f} MB)")
                col_m3.metric("Transfer Speed", f"{speed:.1f} MB/s")
                col_m4.metric("Time Remaining", time_str)
                
                st.progress(max(0.0, min(1.0, percent / 100.0)))
                
        elif task.get("status") == "success":
            check_and_cleanup_cache("./tailscale_cache")
            st.success(f"✅ {display_id} successfully cached on the server!")
            if st.button("Dismiss notification", key=f"dismiss_success_{tid}"):
                with _copy_tasks_lock:
                    _active_copy_tasks.pop(tid, None)
                active_id = st.session_state.get("active_copy_task_id")
                if active_id and _active_copy_tasks._safe_filename(active_id) == tid:
                    st.session_state.active_copy_task_id = None
                st.rerun()
                
        elif task.get("status") == "error":
            st.error(f"⚠️ {display_id} copy failed: {task.get('error_msg')}")
            if st.button("Dismiss error", key=f"dismiss_error_{tid}"):
                with _copy_tasks_lock:
                    _active_copy_tasks.pop(tid, None)
                active_id = st.session_state.get("active_copy_task_id")
                if active_id and _active_copy_tasks._safe_filename(active_id) == tid:
                    st.session_state.active_copy_task_id = None
                st.rerun()

def render_admin_panel():
    st.markdown("<h1 style='color: #00ff00; font-family: Outfit, sans-serif;'>👑 Admin Panel</h1>", unsafe_allow_html=True)
    col_hdr, col_ref = st.columns([3, 1])
    col_hdr.markdown("Work progress statistics and assignment of patients and entire sites to analysts.")
    if col_ref.button("🔄 Refresh Data", key="admin_refresh_data_btn", use_container_width=True):
        clear_db_caches()
        scan_all_tailscale_patients.clear()
        get_all_analysts.clear()
        estimate_patient_download_size.clear()
        st.toast("🔄 All cache invalidated!")
        st.rerun()
    
    if not st.session_state.firebase_init:
        st.error("Firebase database is offline. Admin Panel requires a database connection.")
        return
        
    db = st.session_state.firestore_db
    
    # 1. Fetch all assignments, site assignments and completed reports
    try:
        completed_reports, assignments, site_assignments = get_cached_admin_data()
        completed_pids = get_completed_pids_dict(completed_reports)
        assigned_pids = {a.get("patient_id"): a for a in assignments if a.get("patient_id")}
        assigned_sites_dict = {sa.get("site"): sa for sa in site_assignments if sa.get("site")}
    except Exception as e:
        st.error(f"Failed to fetch data from Firebase: {e}")
        return
        
    # Scan all patients from VPS Tailscale Shared Folder
    all_patients = scan_all_tailscale_patients()
    all_sites = sorted(list({p["site"] for p in all_patients}))
    
    analysts = get_all_analysts()
    
    # Calculate assigned patients for each analyst dynamically matching against disk scan
    analyst_assigned_pids = {a["username"]: set() for a in analysts}
    global_assigned_pids = set()
    
    # Populate individual assignments
    for asg in assignments:
        pid = asg.get("patient_id")
        to_user = asg.get("assigned_to")
        if pid and to_user in analyst_assigned_pids:
            analyst_assigned_pids[to_user].add(pid)
            global_assigned_pids.add(pid)
            
    # Populate site assignments
    for sa in site_assignments:
        site = sa.get("site")
        to_user = sa.get("assigned_to")
        if site and to_user in analyst_assigned_pids:
            for p in all_patients:
                if p["site"] == site:
                    analyst_assigned_pids[to_user].add(p["patient_id"])
                    global_assigned_pids.add(p["patient_id"])
                    
    # 2. Render general metrics
    total_completed = len(completed_reports)
    unique_completed_patients = len(completed_pids)
    total_assigned = len(global_assigned_pids)
    
    completed_assigned_count = sum(1 for pid in global_assigned_pids if pid in completed_pids)
    pct_progress = (completed_assigned_count / total_assigned * 100.0) if total_assigned > 0 else 100.0
    
    total_on_disk = len(all_patients)
    completed_on_disk = sum(1 for p in all_patients if p["patient_id"] in completed_pids)
    remaining_on_disk = max(0, total_on_disk - completed_on_disk)
    
    st.markdown("##### 📈 General Database and Assignment Statistics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Completed analyses in database", f"{total_completed}")
    m2.metric("Unique patients examined", f"{unique_completed_patients}")
    m3.metric("Progress of assigned tasks", f"{completed_assigned_count} / {total_assigned} ({pct_progress:.1f}%)")
    
    st.markdown("##### 📁 File Statistics on Disk")
    m4, m5, m6 = st.columns(3)
    m4.metric("All patients on disk", f"{total_on_disk}")
    m5.metric("Analyzed on disk", f"{completed_on_disk}")
    m6.metric("Remaining to analyze", f"{remaining_on_disk}")
    
    st.markdown("---")
    
    # 2.5 Show site assignment overview
    st.markdown("### 🏢 Site Assignment Status")
    with st.expander("🏢 Show site assignment status", expanded=False):
        col_hdr1, col_hdr2, col_hdr3 = st.columns([2, 3, 2])
        col_hdr1.markdown("**Site**")
        col_hdr2.markdown("**Assignment Status**")
        col_hdr3.markdown("**Patient Progress on Disk**")
        st.markdown("---")
        
        for site in all_sites:
            col_s, col_stat, col_prog = st.columns([2, 3, 2])
            col_s.write(f"🏢 **Site {site}**")
            
            # Check if assigned as a whole
            if site in assigned_sites_dict:
                assigned_user = assigned_sites_dict[site].get("assigned_to", "unknown")
                analyst_name = next((a["name"] for a in analysts if a["username"] == assigned_user), assigned_user)
                col_stat.markdown(f"🟢 **Assigned in full** to `{analyst_name}`")
            else:
                # Check individual cases in assignments
                site_inds = [asg for asg in assignments if asg.get("site") == site]
                if site_inds:
                    user_sets = {asg.get("assigned_to") for asg in site_inds if asg.get("assigned_to")}
                    user_names = []
                    for u in user_sets:
                        name = next((a["name"] for a in analysts if a["username"] == u), u)
                        user_names.append(f"`{name}`")
                    col_stat.markdown(f"🟡 **Partially assigned** to {', '.join(user_names)}")
                else:
                    col_stat.markdown("🔴 **Unassigned**")
            
            # Count patients in this site on disk and how many are completed
            site_p_disk = [p for p in all_patients if p["site"] == site]
            total_site_p = len(site_p_disk)
            comp_site_p = sum(1 for p in site_p_disk if p["patient_id"] in completed_pids)
            col_prog.write(f"{comp_site_p} / {total_site_p} completed")
            
    st.markdown("---")
    
    # 3. Form: Assign case or site
    st.markdown("### ➕ Assign Patient or Entire Site for Analysis")
    
    assign_type = st.radio("Assignment Type", ["Patient", "Entire Site"], horizontal=True)
    
    if not analysts:
        st.warning("No registered analysts in the system. Add analysts in the sidebar.")
    else:
        if assign_type == "Patient":
            # Find available patients (not individually assigned, site not assigned to anyone)
            available_patients = []
            for p in all_patients:
                pid = p["patient_id"]
                site = p["site"]
                if pid not in assigned_pids and site not in assigned_sites_dict:
                    available_patients.append(p)
                    
            if not available_patients:
                st.info("ℹ️ No patients meet the criteria (unassigned).")
            else:
                with st.form("assign_patient_form"):
                    col_p, col_a, col_btn = st.columns([2, 2, 1])
                    p_options = []
                    for p in available_patients:
                        pid = p["patient_id"]
                        is_comp = pid in completed_pids
                        comp_prefix = "🟢 " if is_comp else ""
                        comp_suffix = " (complete)" if is_comp else ""
                        p_options.append(f"{comp_prefix}{pid} (Site {p['site']}){comp_suffix}")
                    selected_p_str = col_p.selectbox("Select patient", p_options)
                    
                    a_options = [f"{a['name']} ({a['email']})" for a in analysts]
                    selected_a_str = col_a.selectbox("Select analyst", a_options)
                    
                    submitted = col_btn.form_submit_button("➕ Assign patient", use_container_width=True)
                    if submitted:
                        idx_p = p_options.index(selected_p_str)
                        patient_data = available_patients[idx_p]
                        pid = patient_data["patient_id"]
                        site = patient_data["site"]
                        
                        idx_a = a_options.index(selected_a_str)
                        analyst_data = analysts[idx_a]
                        username = analyst_data["username"]
                        
                        try:
                            doc_check = db.collection("assignments").document(pid).get()
                            site_check = db.collection("site_assignments").document(site).get()
                            comp_check = db.collection("analysis_results")\
                                           .where("patient_id", "==", pid)\
                                           .where("phase", "==", "COMPLETED")\
                                           .limit(1).stream()
                            is_comp = len(list(comp_check)) > 0
                            
                            if doc_check.exists:
                                st.error(f"Error: Patient {pid} is already assigned!")
                            elif site_check.exists:
                                st.error(f"Error: Site {site} is assigned to {site_check.to_dict().get('assigned_to')}!")
                            else:
                                db.collection("assignments").document(pid).set({
                                    "patient_id": pid,
                                    "site": site,
                                    "assigned_to": username,
                                    "assigned_by": st.session_state.user.get("username", "admin"),
                                    "assigned_at": firestore.SERVER_TIMESTAMP
                                })
                                clear_db_caches()
                                comp_note = " (completed case)" if is_comp else ""
                                st.success(f"Successfully assigned patient {pid} to {username}{comp_note}!")
                                time.sleep(1.0)
                                st.rerun()
                        except Exception as ex:
                            st.error(f"Error during assignment: {ex}")
        else: # Entire Site
            # Find available sites (not already assigned as a whole)
            available_sites = [s for s in all_sites if s not in assigned_sites_dict]
            
            if not available_sites:
                st.info("ℹ️ All sites are already assigned.")
            else:
                with st.form("assign_site_form"):
                    col_s, col_a, col_btn = st.columns([2, 2, 1])
                    selected_site = col_s.selectbox("Select site", available_sites)
                    
                    a_options = [f"{a['name']} ({a['email']})" for a in analysts]
                    selected_a_str = col_a.selectbox("Select analyst", a_options)
                    
                    submitted = col_btn.form_submit_button("➕ Assign site", use_container_width=True)
                    if submitted:
                        idx_a = a_options.index(selected_a_str)
                        analyst_data = analysts[idx_a]
                        username = analyst_data["username"]
                        
                        try:
                            site_check = db.collection("site_assignments").document(selected_site).get()
                            if site_check.exists:
                                st.error(f"Error: Site {selected_site} is already assigned!")
                            else:
                                # Check if any patient in this site is individually assigned to another analyst
                                conflicting_cases_ref = db.collection("assignments").where("site", "==", selected_site).stream()
                                conflicting_cases = [doc for doc in conflicting_cases_ref]
                                conflict_found = False
                                for doc in conflicting_cases:
                                    doc_data = doc.to_dict()
                                    if doc_data.get("assigned_to") != username:
                                        st.error(f"Error: Patient {doc.id} from this site is already assigned to another analyst ({doc_data.get('assigned_to')})! Unassign them first.")
                                        conflict_found = True
                                        break
                                
                                if not conflict_found:
                                    # Perform the write for site assignment
                                    db.collection("site_assignments").document(selected_site).set({
                                        "assigned_to": username,
                                        "assigned_by": st.session_state.user.get("username", "admin"),
                                        "assigned_at": firestore.SERVER_TIMESTAMP
                                    })
                                    
                                    # Clean up any individual assignments of this site's patients to the same analyst
                                    for doc in conflicting_cases:
                                        if doc.to_dict().get("assigned_to") == username:
                                            db.collection("assignments").document(doc.id).delete()
                                            
                                    clear_db_caches()
                                    st.success(f"Successfully assigned site {selected_site} to {username}!")
                                    time.sleep(1.0)
                                    st.rerun()
                        except Exception as ex:
                            st.error(f"Error during site assignment: {ex}")
                            
    st.markdown("---")
    st.markdown("### 📥 Cache Assigned Data to VPS (ANGIO only)")
    
    # Render active server prefetching/caching progress bar
    render_server_caching_progress()
    
    cache_type = st.radio("Cache Scope", ["Site", "Patient"], key="admin_cache_scope", horizontal=True)
    
    import threading
    if cache_type == "Site":
        # Get assigned sites
        assigned_sites = sorted(list({sa.get("site") for sa in site_assignments if sa.get("site")}))
        if not assigned_sites:
            st.info("ℹ️ No sites are currently assigned.")
        else:
            col_cs, col_empty = st.columns([3, 2])
            selected_cache_site = col_cs.selectbox("Select assigned site to cache", assigned_sites, key="admin_cache_site_select")
            
            # Find patients under this site
            all_site_pats = [p for p in all_patients if p["site"] == selected_cache_site]
            completed_pids = get_cached_completed_pids()
            
            completed_site_pats = [p for p in all_site_pats if p["patient_id"] in completed_pids]
            active_site_pats = [p for p in all_site_pats if p["patient_id"] not in completed_pids]
            
            cached_active_pats = [p for p in active_site_pats if is_patient_cached(p["site"], p["patient_id"])]
            uncached_active_pats = [p for p in active_site_pats if not is_patient_cached(p["site"], p["patient_id"])]
            
            if not all_site_pats:
                st.warning(f"No patients found on disk for Site {selected_cache_site}.")
            else:
                st.write(f"📊 **Site Statistics:** Total patients on disk: **{len(all_site_pats)}**")
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown(f"🟢 **Completed Cases (will be skipped):** `{len(completed_site_pats)}` patients")
                    with st.expander("Show skipped cases", expanded=False):
                        if completed_site_pats:
                            st.write(", ".join([p["patient_id"] for p in completed_site_pats]))
                        else:
                            st.write("None")
                            
                with c_col2:
                    st.markdown(f"🟡 **Active / In-Progress Cases:** `{len(active_site_pats)}` patients")
                    st.write(f"• Already Cached: **{len(cached_active_pats)}**")
                    st.write(f"• Remaining to Cache: **{len(uncached_active_pats)}**")
                
                if len(uncached_active_pats) == 0:
                    st.success("🎉 All active patients for this site are already cached!")
                else:
                    st.markdown("---")
                    st.markdown("##### 📁 Cases scheduled for download:")
                    st.write(", ".join([p["patient_id"] for p in uncached_active_pats]))
                    
                    # Calculate estimated size of files
                    with st.spinner("Obliczanie szacunkowej wielkości plików do pobrania..."):
                        total_bytes = 0
                        base_dir = "/mnt/dane_dicom/"
                        for p in uncached_active_pats:
                            total_bytes += estimate_patient_download_size(p['site'], p['patient_id'], base_dir)
                        size_gb = total_bytes / (1024**3)
                    
                    st.info(f"📦 **Estimated download size:** `{size_gb:.2f} GB`")
                    
                    # Ask for acceptance
                    accept = st.checkbox("✅ Potwierdzam chęć wgrania powyższych aktywnych przypadków do pamięci VPS", key="confirm_site_cache_upload")
                    
                    if st.button("🚀 Cache Site (ANGIO folders only)", key="admin_cache_site_btn", use_container_width=True, disabled=not accept):
                        task_id = f"prefetch_admin_site_{selected_cache_site}_{int(time.time())}"
                        
                        with _copy_tasks_lock:
                            _active_copy_tasks[task_id] = {
                                "status": "running",
                                "copied_files": 0,
                                "total_files": len(uncached_active_pats),
                                "copied_bytes": 0,
                                "total_bytes": total_bytes,
                                "speed": 0.0,
                                "est_left": 0.0,
                                "start_time": time.time()
                            }
                        thread = threading.Thread(
                            target=_prefetch_cases_worker,
                            args=(uncached_active_pats, task_id),
                            daemon=True
                        )
                        thread.start()
                        st.session_state.active_copy_task_id = task_id
                        st.toast(f"Starting cache copy task for Site {selected_cache_site}...")
                        st.rerun()
    elif cache_type == "Patient":
        assigned_pats = []
        completed_pids = get_cached_completed_pids()
        for pid in global_assigned_pids:
            if pid in completed_pids:
                continue
            site = next((p["site"] for p in all_patients if p["patient_id"] == pid), None)
            if site:
                assigned_pats.append({"patient_id": pid, "site": site})
        
        if not assigned_pats:
            st.info("ℹ️ No patients are currently assigned.")
        else:
            col_cp, col_empty = st.columns([3, 2])
            p_options = []
            p_mapping = {}
            for p in assigned_pats:
                site = p["site"]
                pid = p["patient_id"]
                cached_status = "Cached" if is_patient_cached(site, pid) else "Not Cached"
                opt_str = f"📁 {pid} (Site {site}) - [{cached_status}]"
                p_options.append(opt_str)
                p_mapping[opt_str] = {"patient_id": pid, "site": site}
                
            selected_opt = col_cp.selectbox("Select assigned patient to cache", p_options, key="admin_cache_patient_select")
            target_patient = p_mapping[selected_opt]
            
            is_cached = is_patient_cached(target_patient["site"], target_patient["patient_id"])
            if is_cached:
                st.success(f"🎉 Patient {target_patient['patient_id']} is already cached on the server!")
            else:
                # Calculate estimated size of files
                with st.spinner("Obliczanie wielkości plików pacjenta..."):
                    total_bytes = estimate_patient_download_size(target_patient['site'], target_patient['patient_id'], base_dir)
                    size_gb = total_bytes / (1024**3)
                
                st.info(f"📦 **Estimated download size:** `{size_gb:.2f} GB`")
                
                # Ask for acceptance
                accept_pat = st.checkbox(f"✅ Potwierdzam chęć wgrania pacjenta {target_patient['patient_id']} do pamięci VPS", key="confirm_patient_cache_upload")
                
                if st.button("🚀 Cache Patient (ANGIO folders only)", key="admin_cache_patient_btn", use_container_width=True, disabled=not accept_pat):
                    task_id = f"prefetch_admin_patient_{target_patient['patient_id']}_{int(time.time())}"
                    
                    with _copy_tasks_lock:
                        _active_copy_tasks[task_id] = {
                            "status": "running",
                            "copied_files": 0,
                            "total_files": 1,
                            "copied_bytes": 0,
                            "total_bytes": total_bytes,
                            "speed": 0.0,
                            "est_left": 0.0,
                            "start_time": time.time()
                        }
                    
                    thread = threading.Thread(
                        target=_prefetch_cases_worker,
                        args=([target_patient], task_id),
                        daemon=True
                    )
                    thread.start()
                    st.session_state.active_copy_task_id = task_id
                    st.toast(f"Starting cache copy task for Patient {target_patient['patient_id']}...")
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Bulk Import of Previous Analyses (PDF)")
    with st.expander("📥 Expand to upload previous PDF reports", expanded=False):
        st.write("Select the analyst to whom the analyses should be assigned, then upload the PDF files from disk.")
        
        a_import_options = [f"{a['name']} ({a['email']})" for a in analysts]
        if not a_import_options:
            st.warning("No registered analysts.")
        else:
            selected_import_a_str = st.selectbox("Assign imported reports to:", a_import_options, key="bulk_import_analyst_select")
            
            uploaded_pdfs = st.file_uploader(
                "Select PDF report files (you can select multiple files at once)", 
                type=["pdf"], 
                accept_multiple_files=True,
                key="bulk_pdf_uploader"
            )
            
            if uploaded_pdfs:
                st.info(f"Selected {len(uploaded_pdfs)} PDF files for import.")
                if st.button("🚀 Start import", key="start_bulk_import_btn", use_container_width=True):
                    idx_a = a_import_options.index(selected_import_a_str)
                    analyst_data = analysts[idx_a]
                    import_username = analyst_data["username"]
                    
                    success_count = 0
                    error_count = 0
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, pdf_file in enumerate(uploaded_pdfs):
                        try:
                            status_text.write(f"Processing file {pdf_file.name} ({i+1}/{len(uploaded_pdfs)})...")
                            pdf_bytes = pdf_file.getvalue()
                            
                            # Parse PDF QCA parameters
                            parsed = parse_qca_pdf(pdf_bytes)
                            if parsed is None:
                                st.error(f"❌ Failed to parse parameters from file: {pdf_file.name}. Make sure it is a valid AngioPy report.")
                                error_count += 1
                                continue
                                
                            # Upload PDF to Firebase Storage
                            pdf_url = upload_pdf_to_firebase(pdf_bytes, pdf_file.name)
                            if not pdf_url:
                                st.error(f"❌ Failed to upload file {pdf_file.name} to Storage.")
                                error_count += 1
                                continue
                                
                            # Insert record into Firestore
                            ref_val = parsed["metrics"]["ref_diam_mm"]
                            
                            doc_data = {
                                "patient_id": parsed["patient_id"],
                                "dicom_name": f"imported_{pdf_file.name[:-4]}",
                                "phase": parsed["phase"],
                                "vessel": parsed["vessel"],
                                "aha": parsed["aha"],
                                "ffr_registered": parsed["ffr_registered"],
                                "other_lesion_distal": parsed["other_lesion_distal"],
                                "known_occlude": parsed["known_occlude"],
                                "pdf_url": pdf_url,
                                "timestamp": firestore.SERVER_TIMESTAMP,
                                "analyst": import_username,
                                "metrics": {
                                    "prox_diam_mm": parsed["metrics"]["prox_diam_mm"],
                                    "dist_diam_mm": parsed["metrics"]["dist_diam_mm"],
                                    "ref_diam_mm": ref_val,
                                    "mld_mm": parsed["metrics"]["mld_mm"],
                                    "pct_diameter_stenosis": parsed["metrics"]["pct_diameter_stenosis"],
                                    "pct_area_stenosis": parsed["metrics"]["pct_area_stenosis"],
                                    "lesion_length_mm": parsed["metrics"]["lesion_length_mm"],
                                    "timi_grade": parsed["metrics"]["timi_grade"],
                                    "tfc": parsed["metrics"]["tfc"]
                                }
                            }
                            
                            db.collection("analysis_results").add(doc_data)
                            st.success(f"🟢 Successfully imported: {pdf_file.name}")
                            success_count += 1
                            
                        except Exception as import_err:
                            st.error(f"❌ Error during import of file {pdf_file.name}: {import_err}")
                            error_count += 1
                            
                        progress_bar.progress((i + 1) / len(uploaded_pdfs))
                        
                    status_text.empty()
                    st.write(f"**Import summary:** Successfully imported: **{success_count}**, Errors: **{error_count}**.")
                    clear_db_caches()
                    time.sleep(2.0)
                    st.rerun()

    st.markdown("---")
    
    # 3.8. Show unassigned cases
    st.markdown("### 🚨 Unassigned Cases (Odpisane przypadki)")
    unassigned_list = [asg for asg in assignments if asg.get("status") == "unassigned" or asg.get("assigned_to") == "unassigned"]
    
    if not unassigned_list:
        st.info("ℹ️ No unassigned cases.")
    else:
        for u_asg in unassigned_list:
            u_pid = u_asg.get("patient_id")
            u_site = u_asg.get("site", "unknown")
            u_by = u_asg.get("unassigned_by", "unknown")
            u_reason = u_asg.get("unassigned_reason", "No reason provided")
            u_at = u_asg.get("unassigned_at")
            
            # Format timestamp
            u_date_str = ""
            if u_at:
                try:
                    u_date_str = u_at.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    u_date_str = str(u_at)
                    
            with st.container(border=True):
                col_u1, col_u2, col_u3 = st.columns([3, 2, 2])
                col_u1.markdown(f"📁 **Patient:** `{u_pid}` (Site {u_site})")
                col_u1.markdown(f"👤 **Unassigned by:** `{u_by}` | 🕒 `{u_date_str}`")
                col_u1.markdown(f"💬 **Reason:** *{u_reason}*")
                
                # Reassign controls
                re_a_options = [f"{a['name']} ({a['email']})" for a in analysts]
                selected_re_a = col_u2.selectbox("Reassign to:", re_a_options, key=f"re_select_{u_pid}")
                
                if col_u3.button("🔄 Reassign Case", key=f"re_btn_{u_pid}", use_container_width=True):
                    idx_re = re_a_options.index(selected_re_a)
                    re_analyst = analysts[idx_re]["username"]
                    try:
                        db.collection("assignments").document(u_pid).set({
                            "patient_id": u_pid,
                            "site": u_site,
                            "assigned_to": re_analyst,
                            "assigned_by": st.session_state.user.get("username", "admin"),
                            "assigned_at": firestore.SERVER_TIMESTAMP,
                            "status": "assigned"
                        })
                        clear_db_caches()
                        st.success(f"Successfully reassigned patient {u_pid} to {re_analyst}!")
                        time.sleep(1.0)
                        st.rerun()
                    except Exception as re_err:
                        st.error(f"Error reassigning: {re_err}")

    st.markdown("---")
    
    # 4. Detail list per analyst
    st.markdown("### 📊 Analysts' Progress")
    
    for a in analysts:
        username = a["username"]
        assigned_pids_set = analyst_assigned_pids.get(username, set())
        total_a = len(assigned_pids_set)
        
        completed_a_set = {pid for pid in assigned_pids_set if pid in completed_pids}
        completed_a = len(completed_a_set)
        
        pct_a = (completed_a / total_a * 100.0) if total_a > 0 else 100.0
        
        # Calculate total completed reports and unique patients in entire database by this analyst
        analyst_reports = [r for r in completed_reports if r.get("analyst") == username]
        total_reports_by_analyst = len(analyst_reports)
        unique_patients_by_analyst = len({r.get("patient_id") for r in analyst_reports if r.get("patient_id")})
        
        remaining_assigned = max(0, total_a - completed_a)
        
        st.markdown(f"##### 👤 {a['name']} ({a['email']})")
        st.write(
            f"📋 **From assigned:** analyzed **{completed_a}** out of **{total_a}** patients "
            f"(remaining to do: **{remaining_assigned}**)"
        )
        st.write(
            f"🏆 **Overall performance:** analyzed a total of **{unique_patients_by_analyst}** patients "
            f"(saved **{total_reports_by_analyst}** reports)"
        )
        st.progress(pct_a / 100.0)
        
        # Get assigned sites for this analyst
        a_sites = [sa for sa in site_assignments if sa.get("assigned_to") == username]
        # Get individual assignments for this analyst
        a_individual = [asg for asg in assignments if asg.get("assigned_to") == username]
        
        total_items = len(a_sites) + len(a_individual)
        
        if total_items > 0:
            with st.expander(f"Show assigned tasks for {a['name']} ({total_items})", expanded=False):
                # 1. Show site assignments
                if a_sites:
                    st.markdown("**Sites (assigned in full):**")
                    for sa in a_sites:
                        site = sa.get("site")
                        c_col1, c_col2, c_col3 = st.columns([2, 2, 1])
                        c_col1.write(f"🏢 **Site:** `{site}`")
                        
                        # Count patients in this site: on disk vs completed
                        site_pids_on_disk = [p["patient_id"] for p in all_patients if p["site"] == site]
                        site_completed = [pid for pid in site_pids_on_disk if pid in completed_pids]
                        
                        c_col2.write(f"Progress: {len(site_completed)} / {len(site_pids_on_disk)} completed")
                        
                        unassign_btn_key = f"unassign_site_{site}_{username}"
                        if c_col3.button("🗑️ Unassign", key=unassign_btn_key, use_container_width=True):
                            try:
                                db.collection("site_assignments").document(site).delete()
                                clear_db_caches()
                                st.success(f"Successfully unassigned site {site}!")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"Error unassigning: {ex}")
                                
                # 2. Show individual patient assignments
                if a_individual:
                    st.markdown("**Individual patients:**")
                    for asg in a_individual:
                        pid = asg.get("patient_id")
                        site = asg.get("site")
                        
                        is_done = pid in completed_pids
                        status_icon = "🟢 Completed" if is_done else "🟡 In progress"
                        
                        c_col1, c_col2, c_col3 = st.columns([2, 2, 1])
                        c_col1.write(f"📁 **Patient:** `{pid}` (Site {site})")
                        c_col2.write(f"Status: **{status_icon}**")
                        
                        if is_done:
                            rep_data = completed_pids[pid]
                            pdf_url = rep_data.get("pdf_url")
                            if pdf_url:
                                c_col3.markdown(f"[📄 View Report]({pdf_url})", unsafe_allow_html=True)
                            else:
                                c_col3.write("No PDF link")
                        else:
                            unassign_btn_key = f"unassign_{pid}_{username}"
                            if c_col3.button("🗑️ Unassign", key=unassign_btn_key, use_container_width=True):
                                try:
                                    db.collection("assignments").document(pid).delete()
                                    clear_db_caches()
                                    st.success(f"Successfully unassigned patient {pid}!")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as ex:
                                    st.error(f"Error unassigning: {ex}")
        else:
            st.caption("No assigned patients or sites.")
        st.markdown("####")

def render_analyst_stats_panel():
    st.markdown("<h1 style='color: #00ff00; font-family: Outfit, sans-serif;'>📊 My Statistics</h1>", unsafe_allow_html=True)
    st.markdown("Statistics of your analyses and the list of assigned tasks.")
    
    if not st.session_state.firebase_init:
        st.error("Firebase database is offline.")
        return
        
    db = st.session_state.firestore_db
    username = st.session_state.user.get("username")
    
    try:
        completed_reports = get_cached_completed_reports()
        completed_pids = get_completed_pids_dict(completed_reports)
        assignments, site_assignments = get_cached_analyst_assignments_detail(username)
    except Exception as e:
        st.error(f"Failed to fetch data from Firebase: {e}")
        return
        
    # Scan all patients from VPS Tailscale Shared Folder
    all_patients = scan_all_tailscale_patients()
    
    # Calculate assigned patients for this analyst dynamically matching against disk scan
    assigned_pids_set = set()
    
    # Populate individual assignments
    for asg in assignments:
        pid = asg.get("patient_id")
        if pid:
            assigned_pids_set.add(pid)
            
    # Populate site assignments
    for sa in site_assignments:
        site = sa.get("site")
        if site:
            for p in all_patients:
                if p["site"] == site:
                    assigned_pids_set.add(p["patient_id"])
                    
    # Calculate statistics
    total_assigned = len(assigned_pids_set)
    completed_assigned_set = {pid for pid in assigned_pids_set if pid in completed_pids}
    completed_assigned = len(completed_assigned_set)
    remaining_assigned = max(0, total_assigned - completed_assigned)
    
    # Calculate total completed reports and unique patients in entire database by this analyst
    my_reports = [r for r in completed_reports if r.get("analyst") == username]
    total_reports_by_me = len(my_reports)
    unique_patients_by_me = len({r.get("patient_id") for r in my_reports if r.get("patient_id")})
    
    pct = (completed_assigned / total_assigned * 100.0) if total_assigned > 0 else 100.0
    
    # Render metrics
    st.markdown("##### 📈 My Work Progress")
    m1, m2, m3 = st.columns(3)
    m1.metric("To analyze (from assigned)", f"{remaining_assigned}")
    m2.metric("Completed (from assigned)", f"{completed_assigned} / {total_assigned}")
    m3.metric("Completed analyses (all)", f"{total_reports_by_me} (patients: {unique_patients_by_me})")
    
    st.progress(pct / 100.0)
    st.write(f"You have completed **{pct:.1f}%** of assigned tasks.")
    
    st.markdown("---")
    st.markdown("### 📋 Assigned Tasks")
    
    total_items = len(site_assignments) + len(assignments)
    if total_items > 0:
        # 1. Show site assignments
        if site_assignments:
            st.markdown("##### 🏢 Assigned sites (in full):")
            for sa in site_assignments:
                site = sa.get("site")
                c_col1, c_col2 = st.columns([3, 2])
                c_col1.write(f"🏢 **Site {site}**")
                
                # Count patients in this site on disk vs completed
                site_pids_on_disk = [p["patient_id"] for p in all_patients if p["site"] == site]
                site_completed = [pid for pid in site_pids_on_disk if pid in completed_pids]
                c_col2.write(f"Progress: **{len(site_completed)} / {len(site_pids_on_disk)}** completed patients")
                
        # 2. Show individual patient assignments
        if assignments:
            st.markdown("##### 📁 Assigned individual cases:")
            for asg in assignments:
                pid = asg.get("patient_id")
                site = asg.get("site")
                
                is_done = pid in completed_pids
                status_icon = "🟢 Completed" if is_done else "🟡 In progress"
                
                c_col1, c_col2, c_col3 = st.columns([3, 2, 2])
                c_col1.write(f"📁 **Patient:** `{pid}` (Site {site})")
                c_col2.write(f"Status: **{status_icon}**")
                
                if is_done:
                    rep_data = completed_pids[pid]
                    pdf_url = rep_data.get("pdf_url")
                    if pdf_url:
                        c_col3.markdown(f"[📄 View Report]({pdf_url})", unsafe_allow_html=True)
                    else:
                        c_col3.write("No PDF link")
                else:
                    c_col3.write("—")
    else:
        st.info("ℹ️ You do not have any tasks assigned to analyze at the moment.")

def timi_label(val):
    if val is None:
        return "—"
    return ["0", "I", "II", "III"][val] if 0 <= val <= 3 else str(val)

CELL_STYLE = (
    "display:inline-block;font-size:18px;padding:4px 14px 4px 0;"
    "min-width:140px;vertical-align:top;"
)
LABEL_STYLE = "font-size:12px;color:#999;display:block;margin-bottom:1px;"

def _cell(label: str, value: str, color: str = "") -> str:
    val_style = (
        f"font-size:18px;font-weight:600;color:{color};"
        if color else "font-size:18px;font-weight:600;"
    )
    return (
        f'<span style="{CELL_STYLE}">'
        f'<span style="{LABEL_STYLE}">{label}</span>'
        f'<span style="{val_style}">{value}</span>'
        f'</span>'
    )

def _bool_val(val, true_color: str = "", false_color: str = "") -> str:
    if val is True:
        return (
            f'<span style="color:{true_color};font-size:18px;font-weight:600;">YES</span>'
            if true_color else "YES"
        )
    if val is False:
        return (
            f'<span style="color:{false_color};font-size:18px;font-weight:600;">NO</span>'
            if false_color else "NO"
        )
    return "—"

def _colored(val, color):
    return f'<span style="color:{color};font-size:18px;font-weight:600;">{val}</span>'

def _run_extraction(login, password, patient_val, result_container):
    try:
        print(f"[Thread] Starting eCRF extraction for patient {patient_val}...")
        with ECRFExtractor(headless=True) as ex:
            raw = ex.extract(login, password, patient_val)
        print(f"[Thread] Extraction finished. Cleaning JSON...")
        clean = build_clean_json(raw)
        result_container["data"] = clean
        
        # Auto-save file to local cache
        try:
            ecrf_data_dir = os.path.abspath("./ecrf_data")
            os.makedirs(ecrf_data_dir, exist_ok=True)
            out_path = os.path.join(ecrf_data_dir, f"result_{patient_val}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(clean, f, ensure_ascii=False, indent=2)
            print(f"[Thread] Saved results to {out_path}")
        except Exception as e:
            print(f"[Thread] Error saving cached file: {e}")
    except RuntimeError as exc:
        import traceback
        traceback.print_exc()
        result_container["error"] = str(exc)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result_container["error"] = f"Extraction error: {exc}"

def _wait_thread(thread, start_time, steps, status_box):
    step_i = 0
    while thread.is_alive():
        elapsed = time.time() - start_time
        step_i = min(int(elapsed / 12), len(steps) - 1)
        status_box.info(f"{steps[step_i]}  ({int(elapsed)}s)")
        time.sleep(0.5)
    thread.join()

def render_patient(data):
    p = data.get("patient", {})
    vessels = data.get("vessels", [])
    rand = p.get("randomization_number", "?")
    arm  = p.get("arm") or "—"

    st.markdown(
        f"### Patient {rand}<br>"
        f"<span style='font-size:24px;font-weight:600;color:#e6a817'>{arm}</span>",
        unsafe_allow_html=True,
    )

    if not vessels:
        st.info("No vessel data available.")
        return

    st.markdown("**Vessels**")
    for i, v in enumerate(vessels):
        is_culprit = v.get("culprit")
        culprit_html = (
            ' <span style="color:#c0392b;font-weight:700;">★ CULPRIT</span>'
            if is_culprit else ""
        )

        with st.container(border=True):
            st.markdown(f"**Vessel {i+1}: {v.get('segment', '?')}**")
            ffr_val = v.get("ffr_adenosine")
            ffr_str = _colored(ffr_val, "#e6a817") if ffr_val is not None else "—"
            pci_str = _bool_val(v.get("pci_performed"), true_color="#1a7f47")
            culprit_str = (
                '<span style="color:#c0392b;font-size:18px;font-weight:600;">YES ★</span>'
                if is_culprit else "NO"
            )
            oct_pre_str = (
                _colored("YES", "#e67e22") if v.get("oct_pre") else "NO"
            )

            html = f'<div style="line-height:2.2;">{culprit_html}<br>'

            # ── Wiersz 1: angiografia ───────────────────────────────────────
            html += _cell("Stenosis",   f"{v['stenosis_pct']}%" if v.get("stenosis_pct") is not None else "—")
            html += _cell("TIMI pre",  timi_label(v.get("timi_pre")))
            html += _cell("TIMI post", timi_label(v.get("timi_post")))
            html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">FFR</span>{ffr_str}</span>'

            # ── Wiersz 2: PCI / stent ───────────────────────────────────────
            html += "<br>"
            html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">OCT pre</span>{oct_pre_str}</span>'
            html += _cell("Bifurcation",  _bool_val(v.get("bifurcation")))
            html += _cell("Predilatation", _bool_val(v.get("predilatation")))
            html += _cell("Stent",        _bool_val(v.get("stent_placed")))

            # ── Wiersz 3: PCI / fizjologia ─────────────────────────────────
            html += "<br>"
            html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">PCI</span>{pci_str}</span>'
            html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">Culprit</span>{culprit_str}</span>'
            if v.get("pci_successful") is not None:
                html += _cell("PCI success", _bool_val(v.get("pci_successful"), true_color="#1a7f47", false_color="#c0392b"))

            # Pd/Pa + RFR
            if v.get("pd_pa") is not None or v.get("rfr") is not None:
                html += "<br>"
                pd_pa_val = v.get("pd_pa")
                rfr_val2  = v.get("rfr")
                html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">Pd/Pa</span>{_colored(pd_pa_val, "#e6a817") if pd_pa_val is not None else "—"}</span>'
                html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">RFR</span>{_colored(rfr_val2, "#e6a817") if rfr_val2 is not None else "—"}</span>'

            # ── Wiersz OCT (pola 6.2–6.12) — tylko gdy OCT wykonane ───────
            if v.get("oct_pre"):
                html += "<br><hr style='border:none;border-top:1px solid #eee;margin:6px 0'>"
                html += f'<span style="font-size:12px;color:#e67e22;font-weight:700;text-transform:uppercase;letter-spacing:.05em">OCT Details</span><br>'
                html += _cell("Preparation", _bool_val(v.get("oct_lesion_prep")))
                html += _cell("Catheter",        v.get("oct_catheter") or "—")
                html += _cell("Pullback",       v.get("oct_pullback") or "—")
                html += "<br>"
                tcfa_str = _colored("YES", "#c0392b") if v.get("oct_tcfa") else ("NO" if v.get("oct_tcfa") is False else "—")
                rup_str  = _colored("YES", "#c0392b") if v.get("oct_plaque_rupture") else ("NO" if v.get("oct_plaque_rupture") is False else "—")
                ero_str  = _colored("YES", "#c0392b") if v.get("oct_plaque_erosion") else ("NO" if v.get("oct_plaque_erosion") is False else "—")
                html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">TCFA</span>{tcfa_str}</span>'
                html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">Plaque rupture</span>{rup_str}</span>'
                html += f'<span style="{CELL_STYLE}"><span style="{LABEL_STYLE}">Plaque erosion</span>{ero_str}</span>'
                html += "<br>"
                html += _cell("MLA (mm²)",       str(v["oct_mla_mm2"])      if v.get("oct_mla_mm2")      is not None else "—")
                html += _cell("OCT Stenosis (%)", str(v["oct_pct_lumen_stenosis"]) if v.get("oct_pct_lumen_stenosis") is not None else "—")
                html += _cell("Lesion length (mm)", str(v["oct_lesion_length_mm"]) if v.get("oct_lesion_length_mm") is not None else "—")
                html += "<br>"
                html += _cell("Prox. diam. (mm)", str(v["oct_proximal_diam_mm"]) if v.get("oct_proximal_diam_mm") is not None else "—")
                html += _cell("Dist. diam. (mm)", str(v["oct_distal_diam_mm"])   if v.get("oct_distal_diam_mm")   is not None else "—")

            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

            if v.get("stents"):
                parts = [
                    f"{s.get('type') or 'stent'} {s.get('diameter_mm')}×{s.get('length_mm')}mm"
                    if s.get("diameter_mm") and s.get("length_mm")
                    else s.get("type") or "stent"
                    for s in v["stents"]
                ]
                st.info("Stents: " + " | ".join(parts))

def render_ecrf_sidebar():
    active_patient_id = st.session_state.get("patient_id", "").strip()
    is_valid_pid = bool(re.match(r"^\d{4}-\d{4}$", active_patient_id))
    
    ecrf_data_dir = os.path.abspath("./ecrf_data")
    os.makedirs(ecrf_data_dir, exist_ok=True)
    
    # Check if a background thread is active
    thread_key = f"ecrf_thread_{active_patient_id}"
    thread_info = st.session_state.get(thread_key)
    
    # 1. Render progress bar in the sidebar if thread is running
    if thread_info and is_valid_pid:
        thread = thread_info["thread"]
        rc = thread_info["rc"]
        start_time = thread_info["start_time"]
        
        if thread.is_alive():
            # Show non-blocking progress status and bar directly in the sidebar
            elapsed = time.time() - start_time
            steps = ["Logging into eCRF...", "Searching for patient...", "Loading CRF sections...", "Extracting data..."]
            step_i = min(int(elapsed / 12), len(steps) - 1)
            
            st.sidebar.info(f"⏳ Fetching eCRF for {active_patient_id}...\nStatus: {steps[step_i]} ({int(elapsed)}s)")
            progress_val = min(elapsed / 45.0, 0.99)
            st.sidebar.progress(progress_val)
            
            # Autorefresh: wait 1.5 seconds and rerun to update the UI
            time.sleep(1.5)
            st.rerun()
        else:
            # Thread finished
            if rc["error"]:
                st.sidebar.error(f"❌ eCRF extraction failed: {rc['error']}")
                st.session_state.pop(thread_key, None)
            elif rc["data"]:
                st.sidebar.success(f"✅ eCRF data fetched for {active_patient_id}!")
                st.session_state.pop(thread_key, None)
                st.rerun()
 
    # 2. Render the collapsible eCRF expander
    with st.sidebar.expander("🏥 eCRF Diagram", expanded=False):
        json_path = None
        if is_valid_pid:
            json_path = os.path.join(ecrf_data_dir, f"result_{active_patient_id}.json")
            
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    patient_data = json.load(f)
                
                # Render clinical cards inside the sidebar!
                render_patient(patient_data)
                
                # Download and delete buttons
                st.markdown("---")
                
                if st.button("🔄 Force Refresh from eCRF", key=f"sidebar_ref_{active_patient_id}", use_container_width=True):
                    status_box = st.empty()
                    rc = {"data": None, "error": None}
                    start_time = time.time()
                    steps = ["Logging in...", "Searching...", "Loading...", "Extracting..."]
                    
                    thread = threading.Thread(
                        target=_run_extraction,
                        args=("roledert", "Troleder79!", active_patient_id, rc),
                        daemon=True
                    )
                    st.session_state[f"ecrf_thread_{active_patient_id}"] = {
                        "thread": thread,
                        "rc": rc,
                        "start_time": start_time,
                        "patient_id": active_patient_id
                    }
                    thread.start()
                    
                    with st.spinner("Fetching updated eCRF data..."):
                        _wait_thread(thread, start_time, steps, status_box)
                    status_box.empty()
                    
                    if rc["error"]:
                        st.sidebar.error(f"❌ eCRF extraction failed: {rc['error']}")
                    elif rc["data"]:
                        st.sidebar.success(f"✅ eCRF data updated!")
                        st.session_state.pop(f"ecrf_thread_{active_patient_id}", None)
                        st.rerun()

                col_dl, col_del = st.columns(2)
                with col_dl:
                    st.download_button(
                        "⬇️ JSON",
                        data=json.dumps(patient_data, ensure_ascii=False, indent=2),
                        file_name=f"result_{active_patient_id}.json",
                        mime="application/json",
                        key=f"sidebar_dl_{active_patient_id}",
                        use_container_width=True
                    )
                with col_del:
                    if st.button("🗑️ Delete", key=f"sidebar_del_{active_patient_id}", use_container_width=True):
                        try:
                            os.remove(json_path)
                            st.toast(f"Removed cache for {active_patient_id}")
                            st.rerun()
                        except Exception as del_err:
                            st.error(f"Error: {del_err}")
            except Exception as read_err:
                st.error(f"File read error: {read_err}")
        else:
            # File not found
            if active_patient_id:
                if is_valid_pid:
                    st.info(f"No eCRF data found for Patient: {active_patient_id}")
                else:
                    st.warning(f"Invalid Patient ID format: {active_patient_id}")
            else:
                st.info("No active Patient ID.")
                
            st.markdown("**Manual Fetch / Login**")
            with st.form("sidebar_ecrf_scrape_form"):
                sc_patient = st.text_input("Patient ID", value=active_patient_id if is_valid_pid else "", placeholder="e.g. 4001-0002")
                sc_login = st.text_input("Login", value="roledert")
                sc_pass = st.text_input("Password", type="password", value="Troleder79!")
                sc_submit = st.form_submit_button("Fetch from eCRF", use_container_width=True)
                
            if sc_submit:
                sc_patient_val = sc_patient.strip()
                if not re.match(r"^\d{4}-\d{4}$", sc_patient_val):
                    st.error("Invalid format!")
                elif not sc_login or not sc_pass:
                    st.error("Please provide login and password!")
                else:
                    status_box = st.empty()
                    rc = {"data": None, "error": None}
                    start_time = time.time()
                    steps = ["Logging in...", "Searching...", "Loading...", "Extracting..."]
                    
                    thread = threading.Thread(target=_run_extraction, args=(sc_login, sc_pass, sc_patient_val, rc), daemon=True)
                    st.session_state[f"ecrf_thread_{sc_patient_val}"] = {
                        "thread": thread,
                        "rc": rc,
                        "start_time": start_time,
                        "patient_id": sc_patient_val
                    }
                    thread.start()
                    
                    with st.spinner("Fetching..."):
                        _wait_thread(thread, start_time, steps, status_box)
                    status_box.empty()
                    st.rerun()
                    
            st.markdown("---")
            uploaded_crf_file = st.file_uploader("Upload JSON file manually", type=["json"], key="sidebar_uploaded_crf")
            if uploaded_crf_file:
                try:
                    uploaded_data = json.load(uploaded_crf_file)
                    if "patient" not in uploaded_data or "vessels" not in uploaded_data:
                        st.error("Invalid JSON file format.")
                    else:
                        upl_pid = uploaded_data.get("patient", {}).get("randomization_number")
                        if not upl_pid:
                            upl_pid = active_patient_id
                        if not re.match(r"^\d{4}-\d{4}$", upl_pid):
                            st.error("Failed to read patient ID.")
                        else:
                            out_path = os.path.join(ecrf_data_dir, f"result_{upl_pid}.json")
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(uploaded_data, f, ensure_ascii=False, indent=2)
                            st.success("Uploaded successfully!")
                            st.rerun()
                except Exception as up_err:
                    st.error(f"Load error: {up_err}")

ssl._create_default_https_context = ssl._create_unverified_context

class DiskSyncedActiveTasks:
    def __init__(self, directory=None):
        import os
        if directory is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            directory = os.path.join(script_dir, "local_cache", "tasks")
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.memory_tasks = {}
        
    def _safe_filename(self, key):
        import re
        # Convert absolute or relative paths to a safe filename string
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', key)
        return safe

    def _save_to_disk(self, key, value):
        import os
        import json
        try:
            safe_key = self._safe_filename(key)
            temp_path = os.path.join(self.directory, f"{safe_key}.tmp")
            final_path = os.path.join(self.directory, f"{safe_key}.json")
            with open(temp_path, "w") as f:
                json.dump(value, f)
            os.replace(temp_path, final_path)
        except Exception as e:
            print(f"[DISK TASKS] Error saving task {key} to disk: {e}")
            
    def _read_from_disk(self, key):
        import os
        import json
        safe_key = self._safe_filename(key)
        path = os.path.join(self.directory, f"{safe_key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DISK TASKS] Error reading task {key} from disk: {e}")
        return None
        
    def _get_all_tasks(self):
        import os
        import json
        import time
        tasks = {}
        if not os.path.exists(self.directory):
            return tasks
        try:
            for fn in os.listdir(self.directory):
                if fn.endswith(".json"):
                    tid = fn[:-5]
                    path = os.path.join(self.directory, fn)
                    try:
                        mtime = os.path.getmtime(path)
                        with open(path, "r") as f:
                            data = json.load(f)
                        
                        is_running = data.get("status") == "running"
                        recent = (time.time() - mtime) < 3600  # 1 hour limit for completed/failed tasks
                        if is_running:
                            # 5-minute timeout check for running tasks to recover from crashes/restarts
                            if (time.time() - mtime) < 300:
                                tasks[tid] = data
                            else:
                                data["status"] = "error"
                                data["error_msg"] = "Task aborted (timeout / server restart)"
                                tasks[tid] = data
                                try:
                                    temp_path = os.path.join(self.directory, f"{fn}.tmp")
                                    with open(temp_path, "w") as f_out:
                                        json.dump(data, f_out)
                                    os.replace(temp_path, path)
                                except:
                                    pass
                        elif recent:
                            tasks[tid] = data
                    except:
                        pass
        except Exception as e:
            print(f"[DISK TASKS] Error listing tasks from disk: {e}")
            
        for k, v in self.memory_tasks.items():
            safe_k = self._safe_filename(k)
            if safe_k not in tasks:
                tasks[safe_k] = v
        return tasks

    def __setitem__(self, key, value):
        self.memory_tasks[key] = value
        self._save_to_disk(key, value)

    def __getitem__(self, key):
        val = self._read_from_disk(key)
        if val is not None:
            return val
        return self.memory_tasks[key]

    def __contains__(self, key):
        import os
        safe_key = self._safe_filename(key)
        path = os.path.join(self.directory, f"{safe_key}.json")
        return os.path.exists(path) or (key in self.memory_tasks)

    def pop(self, key, default=None):
        import os
        import json
        self.memory_tasks.pop(key, None)
        safe_key = self._safe_filename(key)
        path = os.path.join(self.directory, f"{safe_key}.json")
        val = None
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    val = json.load(f)
                os.remove(path)
            except Exception as e:
                print(f"[DISK TASKS] Error deleting task {key}: {e}")
        return val if val is not None else default

    def items(self):
        return self._get_all_tasks().items()

    def keys(self):
        return self._get_all_tasks().keys()

    def values(self):
        return self._get_all_tasks().values()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        import os
        self.memory_tasks.clear()
        if os.path.exists(self.directory):
            for fn in os.listdir(self.directory):
                if fn.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.directory, fn))
                    except:
                        pass

@st.cache_resource
def get_global_copy_tasks():
    import threading
    return DiskSyncedActiveTasks(), threading.Lock()

_active_copy_tasks, _copy_tasks_lock = get_global_copy_tasks()

def _background_copy_worker(src, dst, task_id):
    import shutil
    import time
    import re
    import threading
    
    # Trigger eCRF extraction in background thread
    try:
        src_list = src if isinstance(src, list) else [src]
        for s in src_list:
            match = re.search(r'\b\d{4}-\d{4}\b', s)
            if match:
                pid = match.group(0)
                ecrf_data_dir = os.path.abspath("./ecrf_data")
                os.makedirs(ecrf_data_dir, exist_ok=True)
                json_path = os.path.join(ecrf_data_dir, f"result_{pid}.json")
                if not os.path.exists(json_path):
                    login_val = "roledert"
                    pass_val = "Troleder79!"
                    rc = {"data": None, "error": None}
                    threading.Thread(
                        target=_run_extraction,
                        args=(login_val, pass_val, pid, rc),
                        daemon=True
                    ).start()
                    break
    except Exception:
        pass

    try:
        src_list = src if isinstance(src, list) else [src]
        valid_srcs = [s for s in src_list if os.path.exists(s)]
        if not valid_srcs:
            with _copy_tasks_lock:
                _active_copy_tasks[task_id] = {
                    "status": "error",
                    "error_msg": "Source paths do not exist."
                }
            return
            
        os.makedirs(dst, exist_ok=True)
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "running",
                "copied_files": 0,
                "total_files": len(valid_srcs),
                "copied_bytes": 0,
                "total_bytes": 0,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": "Scanning files on Tailscale..."
            }
        files_to_copy = []
        total_size_bytes = 0
        
        for s_path in valid_srcs:
            s_name = os.path.basename(s_path)
            
            # Check if an ANGIO subfolder (case-insensitive) exists under this patient folder
            has_angio = False
            try:
                for entry in os.scandir(s_path):
                    if entry.is_dir() and entry.name.upper() == "ANGIO":
                        has_angio = True
                        break
            except Exception:
                pass

            for root, dirs, files in os.walk(s_path):
                for file in files:
                    if file.startswith('.'):
                        continue
                    src_fp = os.path.join(root, file)
                    
                    # Restrict to files under the ANGIO subfolder if it exists
                    if has_angio:
                        rel_to_patient = os.path.relpath(src_fp, s_path)
                        rel_parts_upper = [p.upper() for p in rel_to_patient.replace("\\", "/").split("/")]
                        if "ANGIO" not in rel_parts_upper:
                            continue

                    try:
                        sz = os.path.getsize(src_fp)
                    except Exception:
                        sz = 0
                    dst_fp = get_cache_file_path(src_fp)
                    files_to_copy.append((src_fp, dst_fp, sz))
                    total_size_bytes += sz
                    
        total_files = len(files_to_copy)
        if total_files == 0:
            with _copy_tasks_lock:
                _active_copy_tasks[task_id] = {
                    "status": "success",
                    "total_files": 0,
                    "total_bytes": 0
                }
            return
            
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "running",
                "copied_files": 0,
                "total_files": total_files,
                "copied_bytes": 0,
                "total_bytes": total_size_bytes,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": ""
            }
            
        copied_bytes = 0
        start_time = time.time()
        
        for idx, (src_fp, dst_fp, sz) in enumerate(files_to_copy):
            os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
            robust_copy(src_fp, dst_fp)
            copied_bytes += sz
            
            elapsed = time.time() - start_time
            speed_mb = 0.0
            if elapsed > 0:
                speed_mb = (copied_bytes / (1024 * 1024)) / elapsed
                
            est_left = 0.0
            if copied_bytes > 0:
                bytes_left = total_size_bytes - copied_bytes
                est_left = bytes_left / (copied_bytes / elapsed)
                
            with _copy_tasks_lock:
                _active_copy_tasks[task_id] = {
                    "status": "running",
                    "copied_files": idx + 1,
                    "total_files": total_files,
                    "copied_bytes": copied_bytes,
                    "total_bytes": total_size_bytes,
                    "speed": speed_mb,
                    "est_left": est_left,
                    "start_time": start_time
                }
                
        try:
            now = time.time()
            os.utime(dst, (now, now))
            sentinel_path = os.path.join(dst, ".cache_complete")
            with open(sentinel_path, "w") as f:
                f.write("completed")
        except Exception:
            pass
            
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "success",
                "copied_files": total_files,
                "total_files": total_files,
                "copied_bytes": total_size_bytes,
                "total_bytes": total_size_bytes,
                "speed": 0.0,
                "est_left": 0.0
            }
            
    except Exception as e:
        with _copy_tasks_lock:
            _active_copy_tasks[task_id] = {
                "status": "error",
                "error_msg": str(e)
            }

def render_loaded_files_sidebar():
    if st.session_state.dicom_registry:
        with st.sidebar.expander("📄 List of loaded files", expanded=False):
            st.markdown(f"**Number of files:** {len(st.session_state.dicom_registry)}")
            file_names = sorted(list(st.session_state.dicom_registry.keys()))
            for fn in file_names[:100]:
                path = st.session_state.dicom_registry[fn]
                try:
                    sz_bytes = os.path.getsize(path)
                    sz_str = f"({sz_bytes / (1024*1024):.2f} MB)"
                except Exception:
                    sz_str = ""
                st.markdown(f"- `{fn}` {sz_str}", help=f"Path: {path}")
            if len(file_names) > 100:
                st.markdown(f"*...and {len(file_names) - 100} other files*")

@st.fragment(run_every="4s")
def render_sidebar_copy_progress(active_task_id):
    st.empty()  # Always render an element to ensure the fragment mounts in the frontend
    with _copy_tasks_lock:
        task = _active_copy_tasks.get(active_task_id)
    if task:
        if task["status"] == "running":
            copied_files = task.get("copied_files", 0)
            total_files = task.get("total_files", 0)
            copied_bytes = task.get("copied_bytes", 0)
            total_bytes = task.get("total_bytes", 0)
            speed = task.get("speed", 0.0)
            est_left = task.get("est_left", 0.0)
            detail = task.get("detail", "")
            
            if detail == "Scanning files on Tailscale...":
                st.info("🔍 Scanning remote directory on Tailscale... (This can take up to 30 seconds)")
                st.progress(0.0)
            else:
                percent = int(copied_files / total_files * 100) if total_files > 0 else 0
                copied_mb = copied_bytes / (1024 * 1024)
                total_mb = total_bytes / (1024 * 1024)
                
                time_str = ""
                if est_left > 60:
                    time_str = f" (~{int(est_left // 60)}m {int(est_left % 60)}s left)"
                elif est_left > 0:
                    time_str = f" (~{int(est_left)}s left)"
                    
                st.markdown(f"📥 **Caching Background Task:** {copied_files} of {total_files} files ({copied_mb:.1f}/{total_mb:.1f} MB, {speed:.1f} MB/s)")
                st.progress(percent / 100.0, text=f"{percent}% complete{time_str}")
        elif task["status"] == "success":
            check_and_cleanup_cache("./tailscale_cache")
            st.success("✅ Folder successfully cached in background!")
            if st.button("Dismiss success", key=f"dismiss_sidebar_success_{active_task_id}"):
                with _copy_tasks_lock:
                    _active_copy_tasks.pop(active_task_id, None)
                st.session_state.active_copy_task_id = None
                st.rerun()
        elif task["status"] == "error":
            st.error(f"⚠️ Copy failed: {task['error_msg']}")
            if st.button("Dismiss error", key=f"dismiss_sidebar_error_{active_task_id}"):
                with _copy_tasks_lock:
                    _active_copy_tasks.pop(active_task_id, None)
                st.session_state.active_copy_task_id = None
                st.rerun()

def render_cache_importer_sidebar():
    if 'importer_expanded' not in st.session_state:
        st.session_state.importer_expanded = False
        
    if 'importer_tailscale_nav_path' not in st.session_state:
        st.session_state.importer_tailscale_nav_path = ""
        
    def keep_importer_open():
        st.session_state.importer_expanded = True
        
    def detect_site_and_patient(uploaded_files, default_folder="uploaded_patient"):
        import pydicom
        import re
        for uploaded_file in uploaded_files[:5]:
            try:
                uploaded_file.seek(0)
                dcm = pydicom.dcmread(uploaded_file, force=True)
                uploaded_file.seek(0)
                if "PatientID" in dcm and dcm.PatientID:
                    pid = dcm.PatientID.strip()
                    pid = re.sub(r'[\\/*?:"<>|]', '', pid)
                    match = re.match(r"^(\d{4})[-_\s]?\d*", pid)
                    if match:
                        site = match.group(1)
                        return f"{site}/{pid}"
                    return pid
            except Exception:
                pass
        patient_pattern = re.compile(r"(\d{4})[-_\s](\d+)")
        for uploaded_file in uploaded_files:
            match = patient_pattern.search(uploaded_file.name)
            if match:
                site = match.group(1)
                pid = f"{site}-{match.group(2)}"
                return f"{site}/{pid}"
        return default_folder
        
    st.sidebar.markdown("---")
    with st.sidebar.expander("📥 Local Cache Importer", expanded=st.session_state.importer_expanded):
        st.markdown("<p style='font-size:0.85rem; color:#a1a1aa;'>Import folders from Tailscale or upload files directly into the local server cache.</p>", unsafe_allow_html=True)
        
        # Check active background copy task
        active_task_id = st.session_state.get("active_copy_task_id")
        if not active_task_id:
            with _copy_tasks_lock:
                running_tasks = [tid for tid, t in _active_copy_tasks.items() if t["status"] == "running" and not tid.startswith("prefetch_admin_")]
                if running_tasks:
                    active_task_id = running_tasks[0]
                    st.session_state.active_copy_task_id = active_task_id
                    
        if active_task_id:
            render_sidebar_copy_progress(active_task_id)
            return
        
        mode = st.radio(
            "Select Import Source:", 
            ["🌐 Tailscale Shared Folder", "🚀 Mass Prefetch Cases", "💻 Upload Local Files", "📦 Upload ZIP Archive"], 
            key="cache_import_mode",
            on_change=keep_importer_open
        )
        
        if mode == "🌐 Tailscale Shared Folder":
            base_dir = "/mnt/dane_dicom/"
            curr_nav = st.session_state.importer_tailscale_nav_path
            resolved_nav = resolve_clean_path_to_raw(curr_nav, base_dir)
            current_abs_path = os.path.abspath(os.path.join(base_dir, resolved_nav))
            if not current_abs_path.startswith(os.path.abspath(base_dir)):
                current_abs_path = os.path.abspath(base_dir)
                st.session_state.importer_tailscale_nav_path = ""
                
            rel_display = os.path.relpath(current_abs_path, base_dir)
            if rel_display == ".":
                rel_display = "Root"
                
            # If the resolved path has a raw name with vessel suffix, let's display only the clean version to the user
            display_path = curr_nav if curr_nav != "" else "Root"
            st.markdown(f"📁 **Tailscale Path:** `{display_path}`")
            
            subdirs = []
            try:
                if os.path.exists(current_abs_path):
                    raw_dirs = []
                    for entry in os.scandir(current_abs_path):
                        if entry.is_dir() and not entry.name.startswith('.'):
                            raw_dirs.append(entry.name)
                    subdirs = filter_subdirs_by_assignments(st.session_state.importer_tailscale_nav_path, raw_dirs)
            except Exception as e:
                st.error(f"Error reading folder: {e}")
            
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                if st.session_state.importer_tailscale_nav_path != "":
                    if st.button("⬅️ Parent Dir", key="imp_parent_dir_btn", use_container_width=True):
                        parent = os.path.dirname(st.session_state.importer_tailscale_nav_path)
                        st.session_state.importer_tailscale_nav_path = parent
                        st.session_state.importer_expanded = True
                        st.rerun()
            with col_nav2:
                if st.session_state.importer_tailscale_nav_path != "":
                    if st.button("🏠 Root", key="imp_root_dir_btn", use_container_width=True):
                        st.session_state.importer_tailscale_nav_path = ""
                        st.session_state.importer_expanded = True
                        st.rerun()
                        
            if subdirs:
                selected_subdir = st.selectbox(
                    "Go into subfolder:", 
                    ["-- Select Subfolder --"] + subdirs, 
                    key="imp_go_into_subfolder_select",
                    on_change=keep_importer_open,
                    format_func=format_patient_folder_option
                )
                if selected_subdir != "-- Select Subfolder --":
                    new_path = os.path.join(st.session_state.importer_tailscale_nav_path, selected_subdir)
                    st.session_state.importer_tailscale_nav_path = auto_detect_angio_subfolder(new_path, base_dir)
                    st.session_state.importer_expanded = True
                    st.rerun()
                    
            if st.button("🚀 Cache Current Folder", key="imp_cache_folder_btn", use_container_width=True):
                st.session_state.importer_expanded = True
                if display_path == "Root":
                    st.error("⚠️ Cannot cache the root directory. Please navigate into a site/patient subfolder.")
                else:
                    # Resolve all matching raw source paths to copy
                    r_locs = get_raw_folders_for_clean_path(st.session_state.importer_tailscale_nav_path, base_dir)
                    src_abs_list = [os.path.abspath(os.path.join(base_dir, r)) for r in r_locs]
                    dst_abs = os.path.abspath(os.path.join("./tailscale_cache", st.session_state.importer_tailscale_nav_path))
                    
                    total_size = sum(get_dir_size(s) for s in src_abs_list)
                    max_allowed_copy_bytes = 100 * 1024 * 1024 * 1024
                    if total_size > max_allowed_copy_bytes:
                        st.error(f"⚠️ Selected data size ({total_size / (1024**3):.2f} GB) exceeds the 100 GB limit.")
                    else:
                        task_id = dst_abs
                        with _copy_tasks_lock:
                            _active_copy_tasks[task_id] = {
                                "status": "running",
                                "copied_files": 0,
                                "total_files": 1,
                                "copied_bytes": 0,
                                "total_bytes": total_size,
                                "speed": 0.0,
                                "est_left": 0.0,
                                "start_time": time.time()
                            }
                        thread = threading.Thread(
                            target=_background_copy_worker,
                            args=(src_abs_list, dst_abs, task_id),
                            daemon=True
                        )
                        thread.start()
                        st.session_state.active_copy_task_id = task_id
                        st.rerun()
                        
        elif mode == "🚀 Mass Prefetch Cases":
            if not st.session_state.get("firebase_init") or not st.session_state.user:
                st.warning("⚠️ Accessing assigned cases requires being logged in with Firebase online.")
            else:
                username = st.session_state.user.get("username")
                assigned_sites, assigned_cases, unassigned_cases = get_cached_analyst_assignments(username)
                all_patients = scan_all_tailscale_patients()
                
                assigned_patients = []
                for p in all_patients:
                    site = p["site"]
                    pid = p["patient_id"]
                    if pid in unassigned_cases:
                        continue
                    if site in assigned_sites or pid in assigned_cases:
                        assigned_patients.append(p)
                
                # Filter out already cached cases and completed cases
                uncached_assigned = []
                completed_pids = get_cached_completed_pids()
                for p in assigned_patients:
                    if p["patient_id"] not in completed_pids and not is_patient_cached(p["site"], p["patient_id"]):
                        uncached_assigned.append(p)
                        
                if not uncached_assigned:
                    st.info("ℹ️ All assigned cases are already cached locally!")
                else:
                    options = [f"{p['site']}/{p['patient_id']}" for p in uncached_assigned]
                    selected_options = st.multiselect(
                        "Select cases to pre-fetch:", 
                        options, 
                        key="prefetch_multiselect",
                        on_change=keep_importer_open
                    )
                    
                    if selected_options:
                        total_prefetch_size = 0
                        base_dir = "/mnt/dane_dicom/"
                        for opt in selected_options:
                            site, pid = opt.split("/")
                            clean_path = f"{site}/{pid}"
                            r_locs = get_raw_folders_for_clean_path(clean_path, base_dir)
                            for r_loc in r_locs:
                                r_path = os.path.join(base_dir, r_loc)
                                if os.path.exists(r_path):
                                    for root, dirs, files in os.walk(r_path):
                                        for file in files:
                                            if not file.startswith("."):
                                                try:
                                                    total_prefetch_size += os.path.getsize(os.path.join(root, file))
                                                except:
                                                    pass
                                                    
                        size_gb = total_prefetch_size / (1024**3)
                        st.markdown(f"📦 **Estimated download size:** `{size_gb:.2f} GB`")
                        
                        max_allowed_bytes = 100 * 1024 * 1024 * 1024
                        if total_prefetch_size > max_allowed_bytes:
                            st.error(f"⚠️ Prefetch size ({size_gb:.2f} GB) exceeds 100 GB limit. Please select fewer cases.")
                        else:
                            if st.button("🚀 Start Pre-fetching Cases", key="prefetch_start_btn", use_container_width=True):
                                st.session_state.importer_expanded = True
                                cases_to_prefetch = []
                                for opt in selected_options:
                                    site, pid = opt.split("/")
                                    cases_to_prefetch.append({"site": site, "patient_id": pid})
                                    
                                task_id = "prefetch_" + "_".join([opt.replace("/", "_") for opt in selected_options])[:100]
                                
                                with _copy_tasks_lock:
                                    _active_copy_tasks[task_id] = {
                                        "status": "running",
                                        "copied_files": 0,
                                        "total_files": 1,
                                        "copied_bytes": 0,
                                        "total_bytes": total_prefetch_size,
                                        "speed": 0.0,
                                        "est_left": 0.0,
                                        "start_time": time.time()
                                    }
                                
                                thread = threading.Thread(
                                    target=_prefetch_cases_worker,
                                    args=(cases_to_prefetch, task_id),
                                    daemon=True
                                )
                                thread.start()
                                st.session_state.active_copy_task_id = task_id
                                st.rerun()

        elif mode == "💻 Upload Local Files":
            # JS component injection to make st.file_uploader select directories
            import streamlit.components.v1 as components
            components.html("""
            <script>
            const doc = window.parent.document;
            const expanders = doc.querySelectorAll('div[data-testid="stExpander"]');
            expanders.forEach(exp => {
                const header = exp.querySelector('summary');
                if (header && header.textContent.includes('Local Cache Importer')) {
                    const fileInput = exp.querySelector('input[type="file"]');
                    if (fileInput) {
                        fileInput.setAttribute('webkitdirectory', '');
                        fileInput.setAttribute('directory', '');
                        const dropLabel = exp.querySelector('div[data-testid="stFileUploaderDropzone"] span');
                        if (dropLabel && dropLabel.textContent.includes('files')) {
                            dropLabel.textContent = dropLabel.textContent.replace('files', 'folder');
                        }
                    }
                }
            });
            </script>
            """, height=0)
            
            uploaded_files = st.file_uploader(
                "Select folder to upload:", 
                accept_multiple_files=True, 
                key="imp_local_uploader",
                on_change=keep_importer_open
            )
            
            active_pid = st.session_state.get("patient_id", "")
            default_folder_name = active_pid if active_pid else "uploaded_patient"
            if uploaded_files:
                default_folder_name = detect_site_and_patient(uploaded_files, default_folder=default_folder_name)
                
            folder_name = st.text_input(
                "Destination Folder Name:", 
                value=default_folder_name, 
                key="imp_local_folder_name",
                on_change=keep_importer_open
            )
            
            if uploaded_files:
                if st.button("🚀 Upload & Cache Folder", key="imp_local_upload_btn", use_container_width=True):
                    st.session_state.importer_expanded = True
                    if not folder_name.strip():
                        st.error("Please enter a destination folder name.")
                    else:
                        clean_folder_name = folder_name.replace("..", "").strip().strip("/")
                        dst_dir = os.path.abspath(os.path.join("./tailscale_cache", clean_folder_name))
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        total_size_bytes = sum(f.size for f in uploaded_files)
                        max_allowed_copy_bytes = 100 * 1024 * 1024 * 1024
                        
                        if total_size_bytes > max_allowed_copy_bytes:
                            st.error(f"⚠️ Selected data size ({total_size_bytes / (1024**3):.2f} GB) exceeds the 100 GB limit.")
                        else:
                            status_meta = st.empty()
                            progress_meta = st.progress(0, text="Uploading files...")
                            
                            total_files = len(uploaded_files)
                            start_time = time.time()
                            uploaded_bytes = 0
                            
                            for idx, uploaded_file in enumerate(uploaded_files):
                                dst_fp = os.path.join(dst_dir, uploaded_file.name)
                                os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
                                with open(dst_fp, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                uploaded_bytes += uploaded_file.size
                                
                                percent = int((idx + 1) / total_files * 100)
                                elapsed = time.time() - start_time
                                
                                speed_mb = 0.0
                                if elapsed > 0:
                                    speed_mb = (uploaded_bytes / (1024 * 1024)) / elapsed
                                    
                                est_left = 0
                                if uploaded_bytes > 0:
                                    bytes_left = total_size_bytes - uploaded_bytes
                                    est_left = bytes_left / (uploaded_bytes / elapsed)
                                    
                                time_str = ""
                                if est_left > 60:
                                    time_str = f" (~{int(est_left // 60)}m {int(est_left % 60)}s left)"
                                elif est_left > 0:
                                    time_str = f" (~{int(est_left)}s left)"
                                    
                                copied_mb = uploaded_bytes / (1024 * 1024)
                                total_mb = total_size_bytes / (1024 * 1024)
                                
                                status_meta.markdown(f"📥 **Uploading:** {idx + 1} of {total_files} files ({copied_mb:.1f}/{total_mb:.1f} MB, {speed_mb:.1f} MB/s)")
                                progress_meta.progress(percent / 100.0, text=f"{percent}% complete{time_str}")
                                
                            status_meta.empty()
                            progress_meta.empty()
                            
                            touch_directory(dst_dir)
                            try:
                                sentinel_path = os.path.join(dst_dir, ".cache_complete")
                                with open(sentinel_path, "w") as f:
                                    f.write("completed")
                            except Exception:
                                pass
                            # Trigger eCRF extraction for newly uploaded directory
                            try:
                                match = re.search(r'\b\d{4}-\d{4}\b', dst_dir)
                                if match:
                                    pid = match.group(0)
                                    ecrf_data_dir = os.path.abspath("./ecrf_data")
                                    os.makedirs(ecrf_data_dir, exist_ok=True)
                                    json_path = os.path.join(ecrf_data_dir, f"result_{pid}.json")
                                    if not os.path.exists(json_path):
                                        login_val = "roledert"
                                        pass_val = "Troleder79!"
                                        rc = {"data": None, "error": None}
                                        threading.Thread(
                                            target=_run_extraction,
                                            args=(login_val, pass_val, pid, rc),
                                            daemon=True
                                        ).start()
                            except Exception:
                                pass

                            check_and_cleanup_cache("./tailscale_cache")
                            st.success("✅ Folder successfully uploaded and cached!")
                            time.sleep(1.5)
                            st.rerun()
                            
        else: # Upload ZIP Archive
            uploaded_zip = st.file_uploader(
                "Upload folder structure as .zip file:", 
                type=["zip"], 
                key="cache_import_zip_uploader",
                on_change=keep_importer_open
            )
            if uploaded_zip is not None:
                if st.button("🚀 Extract & Cache", key="cache_import_zip_btn", use_container_width=True):
                    st.session_state.importer_expanded = True
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                        tmp.write(uploaded_zip.getbuffer())
                        tmp_zip_path = tmp.name
                    
                    try:
                        import zipfile
                        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                            infolist = zip_ref.infolist()
                            infolist = [info for info in infolist if not info.filename.startswith('__MACOSX') and not os.path.basename(info.filename).startswith('.')]
                            total_size = sum(info.file_size for info in infolist)
                            
                        max_allowed_copy_bytes = 100 * 1024 * 1024 * 1024
                        if total_size > max_allowed_copy_bytes:
                            st.error(f"⚠️ Unpacked data size ({total_size / (1024**3):.2f} GB) exceeds the 100 GB limit.")
                        else:
                            before_dirs = set(get_cached_patients("./tailscale_cache"))
                            
                            status_meta = st.empty()
                            progress_meta = st.progress(0, text="Extracting ZIP archive...")
                            
                            extract_zip_with_progress(tmp_zip_path, "./tailscale_cache", status_meta, progress_meta)
                            
                            status_meta.empty()
                            progress_meta.empty()
                            
                            after_dirs = set(get_cached_patients("./tailscale_cache"))
                            new_dirs = after_dirs - before_dirs
                            for d in new_dirs:
                                touch_directory(d)
                                try:
                                    sentinel_path = os.path.join(d, ".cache_complete")
                                    with open(sentinel_path, "w") as f:
                                        f.write("completed")
                                except Exception:
                                    pass
                                # Trigger eCRF extraction for newly cached directories
                                try:
                                    match = re.search(r'\b\d{4}-\d{4}\b', d)
                                    if match:
                                        pid = match.group(0)
                                        ecrf_data_dir = os.path.abspath("./ecrf_data")
                                        os.makedirs(ecrf_data_dir, exist_ok=True)
                                        json_path = os.path.join(ecrf_data_dir, f"result_{pid}.json")
                                        if not os.path.exists(json_path):
                                            login_val = "roledert"
                                            pass_val = "Troleder79!"
                                            rc = {"data": None, "error": None}
                                            threading.Thread(
                                                target=_run_extraction,
                                                args=(login_val, pass_val, pid, rc),
                                                daemon=True
                                            ).start()
                                except Exception:
                                    pass
                                
                            check_and_cleanup_cache("./tailscale_cache")
                            st.success("✅ ZIP archive successfully unpacked and cached!")
                            time.sleep(1.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                    finally:
                        try:
                            os.remove(tmp_zip_path)
                        except Exception:
                            pass

        st.markdown("---")
        st.markdown("##### 🗑️ Cache Management")
        cache_dir = "./tailscale_cache"
        if os.path.exists(cache_dir):
            curr_size_bytes = get_dir_size(cache_dir)
            curr_size_gb = curr_size_bytes / (1024**3)
            cached_pts = get_cached_patients(cache_dir)
            st.markdown(f"**Current cache size:** `{curr_size_gb:.2f} GB` ({len(cached_pts)} patients)")
            
            confirm = st.checkbox(
                "Confirm cache wipe", 
                key="cache_wipe_confirm",
                on_change=keep_importer_open
            )
            if st.button("🗑️ Clear Local Cache", key="cache_wipe_btn", use_container_width=True, type="secondary", disabled=not confirm):
                st.session_state.importer_expanded = True
                import shutil
                try:
                    for item in os.listdir(cache_dir):
                        item_path = os.path.join(cache_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                    load_dicom_data.clear()
                    get_recursive_preview_files_cached.clear()
                    st.toast("🧹 Local cache wiped successfully!")
                    time.sleep(1.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
        else:
            st.caption("Cache is empty.")

# Initialize Firebase
firebase_available = init_firebase()

# Enforce login only if Firebase initialized successfully
if firebase_available:
    if st.session_state.user is None:
        render_login_page()
        st.stop()
else:
    # If offline, use a fallback account
    if st.session_state.user is None:
        st.session_state.user = {
            "username": "offline_analyst",
            "name": "Offline Analyst",
            "role": "admin"
        }

# Render active user info and log out button in the sidebar
with st.sidebar:
    if st.session_state.user:
        st.markdown(f"👤 **Logged in as:** {st.session_state.user['name']}")
        if not firebase_available:
            st.warning("⚠️ Offline mode (no database)")
        else:
            st.success("🟢 Firebase database active")
            
        # App Mode selection for Admin vs Analyst
        if st.session_state.user.get("role") == "admin":
            if "app_mode" not in st.session_state:
                st.session_state.app_mode = "🔍 Angiography Analysis"
            modes = ["🔍 Angiography Analysis", "👑 Admin Panel", "📖 Instructions"]
            selected_mode = st.selectbox("Application Mode", modes, index=modes.index(st.session_state.app_mode) if st.session_state.app_mode in modes else 0, key="app_mode_select")
            if selected_mode != st.session_state.app_mode:
                st.session_state.app_mode = selected_mode
                st.rerun()
        else:
            if "app_mode" not in st.session_state:
                st.session_state.app_mode = "🔍 Angiography Analysis"
            modes = ["🔍 Angiography Analysis", "📊 My Statistics", "📖 Instructions"]
            selected_mode = st.selectbox("Application Mode", modes, index=modes.index(st.session_state.app_mode) if st.session_state.app_mode in modes else 0, key="app_mode_select")
            if selected_mode != st.session_state.app_mode:
                st.session_state.app_mode = selected_mode
                st.rerun()
            
        # If the user is an admin, let them manage other analysts
        if st.session_state.user.get("role") == "admin" and firebase_available:
            with st.expander("👥 Analyst Management"):
                render_analyst_management()
                
        # Analyst Unassign Form
        active_pid = st.session_state.get("patient_id", "").strip()
        if active_pid and re.match(r"^\d{4}-\d{4}$", active_pid) and firebase_available:
            st.markdown("---")
            with st.expander("🚪 Unassign Patient (Odpisz)", expanded=False):
                st.markdown(f"Unassign patient **{active_pid}** from your workspace.")
                reason_opts = [
                    "Brak plików do ładowania (No files to load)",
                    "Trudna analiza (Difficult analysis)",
                    "Inne (Other - specify below)"
                ]
                selected_reason = st.selectbox("Reason:", reason_opts, key="unassign_reason_select")
                comment = ""
                if selected_reason.startswith("Inne"):
                    comment = st.text_input("Enter custom comment:", key="unassign_comment_input")
                else:
                    comment = selected_reason
                
                if st.button("Confirm Unassignment", key="unassign_confirm_btn", type="primary", use_container_width=True):
                    if comment.strip() == "":
                        st.error("Please enter a reason/comment.")
                    else:
                        try:
                            # Resolve site dynamically
                            site_val = ""
                            if "tailscale_nav_path" in st.session_state and st.session_state.tailscale_nav_path:
                                s_match = re.search(r'(?:Ośrodek|Site)\s*(\d+)', st.session_state.tailscale_nav_path)
                                if s_match:
                                    site_val = s_match.group(1)
                            
                            # Fallback scan all_patients
                            if not site_val:
                                for p in scan_all_tailscale_patients():
                                    if p["patient_id"] == active_pid:
                                        site_val = p["site"]
                                        break
                                        
                            db = st.session_state.firestore_db
                            db.collection("assignments").document(active_pid).set({
                                "patient_id": active_pid,
                                "site": site_val if site_val else "unknown",
                                "assigned_to": "unassigned",
                                "status": "unassigned",
                                "unassigned_by": st.session_state.user.get("username", "unknown"),
                                "unassigned_reason": comment,
                                "unassigned_at": firestore.SERVER_TIMESTAMP
                            })
                            
                            # Reset active workspace state
                            st.session_state.patient_id = ""
                            st.session_state.dicom_registry = {}
                            st.session_state.selected_dicom = None
                            st.session_state.tailscale_nav_path = ""
                            st.session_state.tailscale_selected_locations = []
                            clear_db_caches()
                            
                            st.toast("✅ Case successfully unassigned!")
                            time.sleep(1.5)
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error unassigning: {ex}")
                            
        if st.button("🚪 Log out", key="logout_btn", use_container_width=True):
            keys_to_preserve = [
                'firebase_init', 'firebase_error', 'firestore_db', 'firebase_bucket', 
                'firebase_api_key'
            ]
            for key in list(st.session_state.keys()):
                if key not in keys_to_preserve:
                    del st.session_state[key]
            st.session_state.user = None
            st.session_state.app_mode = "🔍 Angiography Analysis"
            st.rerun()
        st.markdown("---")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

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
def touch_directory(path):
    try:
        now = time.time()
        os.utime(path, (now, now))
    except Exception:
        pass

def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for root, dirs, files in os.walk(path):
        for file in files:
            fp = os.path.join(root, file)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total

def get_cache_destination(src_path, base_tailscale="/mnt/dane_dicom/"):
    src_abs = os.path.abspath(src_path)
    try:
        base_abs = os.path.abspath(base_tailscale)
        if src_abs.startswith(base_abs):
            rel = os.path.relpath(src_abs, base_abs)
            return os.path.abspath(os.path.join("./tailscale_cache", rel))
    except Exception:
        pass
        
    parts = src_abs.replace("\\", "/").split("/")
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        rel = os.path.join(parts[-2], parts[-1])
    elif len(parts) == 1:
        rel = parts[-1]
    else:
        rel = "imported_folder"
    return os.path.abspath(os.path.join("./tailscale_cache", rel))

def extract_zip_with_progress(zip_path, extract_to, status_meta, progress_meta):
    import zipfile
    import time
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        infolist = zip_ref.infolist()
        infolist = [info for info in infolist if not info.filename.startswith('__MACOSX') and not os.path.basename(info.filename).startswith('.')]
        
        total_files = len(infolist)
        if total_files == 0:
            return
            
        total_size_bytes = sum(info.file_size for info in infolist)
        start_time = time.time()
        extracted_bytes = 0
        
        for idx, info in enumerate(infolist):
            zip_ref.extract(info, extract_to)
            extracted_bytes += info.file_size
            
            percent = int((idx + 1) / total_files * 100)
            elapsed = time.time() - start_time
            
            speed_mb = 0.0
            if elapsed > 0:
                speed_mb = (extracted_bytes / (1024 * 1024)) / elapsed
                
            est_left = 0
            if extracted_bytes > 0:
                bytes_left = total_size_bytes - extracted_bytes
                est_left = bytes_left / (extracted_bytes / elapsed)
                
            time_str = ""
            if est_left > 60:
                time_str = f" (~{int(est_left // 60)}m {int(est_left % 60)}s left)"
            elif est_left > 0:
                time_str = f" (~{int(est_left)}s left)"
                
            extracted_mb = extracted_bytes / (1024 * 1024)
            total_mb = total_size_bytes / (1024 * 1024)
            
            status_meta.markdown(f"📥 **Extracting ZIP:** {idx + 1} of {total_files} files ({extracted_mb:.1f}/{total_mb:.1f} MB, {speed_mb:.1f} MB/s)")
            progress_meta.progress(percent / 100.0, text=f"{percent}% complete{time_str}")

def get_cached_patients(cache_dir):
    import re
    patients = []
    if not os.path.exists(cache_dir):
        return patients
    
    # Identify directories representing patient cases (e.g. 3002-0025)
    patient_pattern = re.compile(r"^\d{4}[-_\s]\d+")
    
    for root, dirs, files in os.walk(cache_dir):
        for d in list(dirs):
            if patient_pattern.match(d):
                patients.append(os.path.join(root, d))
                # Do not recurse into matched patient folder
                dirs.remove(d)
                
    # Fallback to depth 1 or 2 if no folders match the patient regex
    if not patients:
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if not os.path.isdir(item_path):
                continue
            subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
            if not subdirs:
                patients.append(item_path)
            else:
                for subdir in subdirs:
                    patients.append(os.path.join(item_path, subdir))
    return patients

def check_and_cleanup_cache(cache_dir, max_size_gb=150, target_size_gb=100):
    max_size = max_size_gb * 1024 * 1024 * 1024
    target_size = target_size_gb * 1024 * 1024 * 1024
    current_size = get_dir_size(cache_dir)
    if current_size <= max_size:
        return
    
    patient_dirs = get_cached_patients(cache_dir)
    dir_mtimes = []
    for d in patient_dirs:
        try:
            mtime = os.path.getmtime(d)
            dir_mtimes.append((d, mtime))
        except Exception:
            pass
    dir_mtimes.sort(key=lambda x: x[1])
    
    import shutil
    deleted_folders = []
    for d, mtime in dir_mtimes:
        if current_size <= target_size:
            break
        try:
            shutil.rmtree(d)
            deleted_folders.append(os.path.basename(d))
            parent = os.path.dirname(d)
            if parent != cache_dir and os.path.exists(parent) and not os.listdir(parent):
                os.rmdir(parent)
            current_size = get_dir_size(cache_dir)
        except Exception:
            pass
            
    if deleted_folders:
        st.toast(f"🧹 Cache cleanup: removed oldest folders: {', '.join(deleted_folders)}")

def copy_folder_with_progress(src, dst, status_meta, progress_meta):
    if not os.path.exists(src):
        return
    os.makedirs(dst, exist_ok=True)
    import shutil
    import time
    
    files_to_copy = []
    total_size_bytes = 0
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.startswith('.'):
                continue
            src_fp = os.path.join(root, file)
            try:
                sz = os.path.getsize(src_fp)
            except Exception:
                sz = 0
            dst_fp = get_cache_file_path(src_fp)
            files_to_copy.append((src_fp, dst_fp, sz))
            total_size_bytes += sz
            
    total_files = len(files_to_copy)
    if total_files == 0:
        return
        
    start_time = time.time()
    copied_bytes = 0
    
    for idx, (src_fp, dst_fp, sz) in enumerate(files_to_copy):
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        robust_copy(src_fp, dst_fp)
        copied_bytes += sz
        
        percent = int((idx + 1) / total_files * 100)
        elapsed = time.time() - start_time
        
        # Speed in MB/s
        speed_mb = 0.0
        if elapsed > 0:
            speed_mb = (copied_bytes / (1024 * 1024)) / elapsed
            
        # Time remaining
        est_left = 0
        if copied_bytes > 0:
            bytes_left = total_size_bytes - copied_bytes
            est_left = bytes_left / (copied_bytes / elapsed)
            
        time_str = ""
        if est_left > 60:
            time_str = f" (~{int(est_left // 60)}m {int(est_left % 60)}s left)"
        elif est_left > 0:
            time_str = f" (~{int(est_left)}s left)"
            
        copied_mb = copied_bytes / (1024 * 1024)
        total_mb = total_size_bytes / (1024 * 1024)
        
        status_meta.markdown(f"📥 **Caching Tailscale files:** {idx + 1} of {total_files} files ({copied_mb:.1f}/{total_mb:.1f} MB, {speed_mb:.1f} MB/s)")
        progress_meta.progress(percent / 100.0, text=f"{percent}% complete{time_str}")

def scan_dicom_folder(folder_path):
    registry = {}
    if not os.path.exists(folder_path):
        return registry
        
    from collections import Counter
    base_names = []
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.startswith('.'):
                continue
            if file.lower().endswith(('.zip', '.rar', '.bmp', '.dat', '.log', '.angioframes', '.coreg', '.oct', '.params', '.dbf', '.png', '.jpg', '.txt', '.pdf', '.xlsx', '.docx', '.exe', '.dll', '.ini', '.sys')):
                continue
            full_path = os.path.join(root, file)
            base_names.append(file)
            all_files.append(full_path)
            
    name_counts = Counter(base_names)
    
    for full_path in all_files:
        file_name = os.path.basename(full_path)
        if name_counts[file_name] > 1:
            parent_dir = os.path.basename(os.path.dirname(full_path))
            key = f"{parent_dir}_{file_name}"
        else:
            key = file_name
            
        if key in registry:
            counter = 1
            base, ext = os.path.splitext(key)
            unique_key = f"{base}_{counter}{ext}"
            while unique_key in registry:
                counter += 1
                unique_key = f"{base}_{counter}{ext}"
            registry[unique_key] = full_path
        else:
            registry[key] = full_path
    return registry

def load_registry_with_progress(registry):
    total_files = len(registry)
    if total_files > 0:
        # Auto-detect Patient ID from file names or paths (pattern XXXX-YYYY like 4001-0002)
        import pydicom
        found_pat = False
        for name, path in registry.items():
            # Search in filename key first
            match = re.search(r'\b\d{4}-\d{4,}\b', name)
            if not match:
                match = re.search(r'\d+-\d+', name)
            # Fallback to absolute path
            if not match:
                match = re.search(r'\b\d{4}-\d{4,}\b', path)
            if not match:
                match = re.search(r'\d+-\d+', path)
                
            if match:
                st.session_state.patient_id = match.group(0)
                found_pat = True
                break
                
        # Fallback: check DICOM header PatientID in the first file
        if not found_pat:
            try:
                first_path = list(registry.values())[0]
                ds = pydicom.dcmread(first_path, stop_before_pixels=True)
                p_id = getattr(ds, "PatientID", None)
                if p_id:
                    st.session_state.patient_id = str(p_id).strip()
            except:
                pass

        # Trigger automatic eCRF background fetch if file does not exist
        patient_id = st.session_state.get("patient_id", "").strip()
        is_valid_pid = bool(re.match(r"^\d{4}-\d{4}$", patient_id))
        
        # Save full registry as backup
        st.session_state.original_dicom_registry = registry.copy()
        is_filtered_on_load = False
        
        if is_valid_pid and not st.session_state.get("show_all_sequences", False):
            chosen_names = get_chosen_dicoms_from_firestore(patient_id)
            if chosen_names:
                filtered_registry = {name: path for name, path in registry.items() if name in chosen_names}
                if filtered_registry:
                    registry = filtered_registry
                    is_filtered_on_load = True
                    
        total_files = len(registry)
        
        if is_valid_pid:
            ecrf_data_dir = os.path.abspath("./ecrf_data")
            os.makedirs(ecrf_data_dir, exist_ok=True)
            json_path = os.path.join(ecrf_data_dir, f"result_{patient_id}.json")
            if not os.path.exists(json_path):
                thread_key = f"ecrf_thread_{patient_id}"
                if thread_key not in st.session_state:
                    rc = {"data": None, "error": None}
                    login_val = "roledert"
                    pass_val = "Troleder79!"
                    
                    thread = threading.Thread(
                        target=_run_extraction, 
                        args=(login_val, pass_val, patient_id, rc), 
                        daemon=True
                    )
                    st.session_state[thread_key] = {
                        "thread": thread,
                        "rc": rc,
                        "start_time": time.time(),
                        "patient_id": patient_id
                    }
                    thread.start()

        st.sidebar.markdown("### 📥 DICOM Import Status")
        
        # Determine if we need to copy files (Tailscale mode)
        is_tailscale = st.session_state.get('dicom_source') == "🌐 Tailscale Shared Folder"
        
        # Phase 1: Initialize registry
        status_meta = st.sidebar.empty()
        progress_meta = st.sidebar.progress(0, text="Initializing...")
        
        # Phase 2: Flow analysis containers
        status_flow = st.sidebar.empty()
        progress_flow = st.sidebar.progress(0, text="Waiting for analysis...")
        
        st.session_state.dicom_registry = {}
        st.session_state.dicom_metadata = {}
        st.session_state.grid_order = None
        st.session_state.patient_cart = []
        st.session_state.is_dicom_registry_filtered = is_filtered_on_load
        st.session_state.original_dicom_registry = {}
        
        import shutil
        import pydicom
        
        # --- PHASE 1: INITIALIZE REGISTRY ---
        start_time_dl = time.time()
        for idx, (name, path) in enumerate(registry.items()):
            percent_meta = int((idx + 1) / total_files * 100)
            percent_left = 100 - percent_meta
            files_left = total_files - (idx + 1)
            
            elapsed = time.time() - start_time_dl
            avg_time = elapsed / (idx + 1)
            est_left = avg_time * files_left
            
            time_str = ""
            if est_left > 60:
                time_str = f" (~{int(est_left // 60)}m {int(est_left % 60)}s left)"
            elif est_left > 0:
                time_str = f" (~{int(est_left)}s left)"
                
            status_meta.markdown(f"📄 **Initializing:** {idx + 1} of {total_files} processed")
            progress_meta.progress(percent_meta / 100.0, text=f"{percent_meta}% complete ({percent_left}% / {files_left} files left){time_str}")
            st.session_state.dicom_registry[name] = path
            
        status_meta.markdown(f"✅ **Import:** All {total_files} files ready (100%)")
        progress_meta.progress(1.0, text="100% complete")
        
        # --- PHASE 2: LOCAL METADATA & FLOW ANALYSIS ---
        start_time_an = time.time()
        for idx, (name, path) in enumerate(st.session_state.dicom_registry.items()):
            percent_flow = int((idx + 1) / total_files * 100)
            percent_left = 100 - percent_flow
            files_left = total_files - (idx + 1)
            
            # Estimate remaining analysis time (should be extremely fast now)
            elapsed = time.time() - start_time_an
            avg_time = elapsed / (idx + 1)
            est_left = avg_time * files_left
            
            time_str = ""
            if est_left > 60:
                time_str = f" (~{int(est_left // 60)}m {int(est_left % 60)}s left)"
            elif est_left > 0:
                time_str = f" (~{int(est_left)}s left)"
                
            status_flow.markdown(f"🔬 **Flow Analysis:** {idx + 1} of {total_files} analyzed")
            progress_flow.progress(percent_flow / 100.0, text=f"{percent_flow}% complete ({percent_left}% / {files_left} files left){time_str}")
            
            # 1. Read metadata (instant from local SSD)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                date_val = getattr(ds, "AcquisitionDate", getattr(ds, "SeriesDate", getattr(ds, "StudyDate", "19700101")))
                time_val = getattr(ds, "AcquisitionTime", getattr(ds, "SeriesTime", getattr(ds, "ContentTime", "999999")))
                dt_str = str(date_val) + str(time_val)
            except:
                dt_str = "99999999999999"
                
            # Run flow analysis (fast from local SSD / cached)
            try:
                best_ix, start_ix, end_ix, tfc, timi, just = analyze_series_flow(path, os.path.getsize(path))
            except Exception:
                best_ix, start_ix, end_ix, tfc, timi, just = 0, 0, 0, 0, 3, "Error, defaulted."

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
                "best_ix": best_ix,
                "start_ix": start_ix,
                "end_ix": end_ix,
                "tfc": tfc,
                "timi": timi,
                "just": just,
            }
                
        status_flow.markdown(f"✅ **Flow Analysis:** All {total_files} analyzed (100%)")
        progress_flow.progress(1.0, text="100% complete")
        
        time.sleep(2.0)
        
        status_meta.empty()
        progress_meta.empty()
        status_flow.empty()
        progress_flow.empty()
        if not st.session_state.is_dicom_registry_filtered:
            st.session_state.original_dicom_registry = st.session_state.dicom_registry.copy()
        st.session_state.pop("show_all_sequences", None)
        st.sidebar.success(f"✅ Loaded {total_files} files successfully!")
    else:
        st.session_state.dicom_registry = {}
        st.session_state.dicom_metadata = {}
        st.session_state.grid_order = None
        st.session_state.patient_cart = []
        st.session_state.original_dicom_registry = {}
        st.session_state.is_dicom_registry_filtered = False

# Check available directories
has_tailscale = os.path.exists("/mnt/dane_dicom/")
has_local_folder = os.path.exists("Dicom/")

source_options = []
if has_tailscale:
    source_options.append("🌐 Tailscale Shared Folder")
if has_local_folder:
    source_options.append("📁 Local Dicom/ Folder")
source_options.append("💻 Upload Local Files")

if 'dicom_source' not in st.session_state:
    if has_tailscale:
        st.session_state.dicom_source = "🌐 Tailscale Shared Folder"
    elif has_local_folder:
        st.session_state.dicom_source = "📁 Local Dicom/ Folder"
    else:
        st.session_state.dicom_source = "💻 Upload Local Files"

if 'selected_tailscale_case' not in st.session_state:
    st.session_state.selected_tailscale_case = None

if 'tailscale_nav_path' not in st.session_state:
    st.session_state.tailscale_nav_path = ""

if 'tailscale_selected_locations' not in st.session_state:
    st.session_state.tailscale_selected_locations = []

if 'dicom_registry' not in st.session_state:
    st.session_state.dicom_registry = {}
    if st.session_state.dicom_source == "📁 Local Dicom/ Folder":
        registry = scan_dicom_folder("Dicom/")
        load_registry_with_progress(registry)


if 'patient_id' not in st.session_state:
    st.session_state.patient_id = ""
if 'patient_cart' not in st.session_state:
    st.session_state.patient_cart = []
if 'dicom_metadata' not in st.session_state:
    st.session_state.dicom_metadata = {}

# ── AHA Vessel Segment Hierarchy ──────────────────────────────────────────────
AHA_VESSEL_SEGMENTS = {
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

@st.cache_data(max_entries=200, ttl=1800, show_spinner=False)
def analyze_series_flow(dicom_path, file_size=None):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pixelArray = dcm.pixel_array
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        elif len(pixelArray.shape) == 3 and pixelArray.shape[2] in (3, 4):
            return 0, 0, 0, 0, 3, "Single image (color capture), TIMI 3 assumed."
        if len(pixelArray.shape) == 2:
            pixelArray = numpy.expand_dims(pixelArray, axis=0)
        n_slices = pixelArray.shape[0]
        
        if n_slices == 1:
            return 0, 0, 0, 0, 3, "Single image, TIMI 3 assumed."
            
        # Normalize to [0, 255] float so gradient scores are bit-depth agnostic
        pa_f = pixelArray.astype(numpy.float32)
        pmin, pmax = pa_f.min(), pa_f.max()
        if pmax > pmin:
            pa_f = (pa_f - pmin) / (pmax - pmin) * 255.0

        scores = []
        for i in range(n_slices):
            frame = pa_f[i]
            small = cv2.resize(frame, (256, 256))
            blurred = scipy.ndimage.gaussian_filter(small.astype(float), 2)
            grad = numpy.abs(numpy.gradient(blurred))
            scores.append(numpy.sum(grad))
            
        scores = numpy.array(scores)
        best_ix = int(numpy.argmax(scores))
        
        # Avoid artifacts in the first few frames by ignoring them if there's a good peak later
        if best_ix < 3 and n_slices > 10:
            alt_best = int(numpy.argmax(scores[3:])) + 3
            if scores[alt_best] > scores[best_ix] * 0.8:
                best_ix = alt_best
                
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
            if detect_contrast(dicom_path, file_size):
                timi = 3
                justification = "Contrast present but peak is immediate (TFC <= 3)."
            else:
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
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0, 0, "Error during analysis calculation."

@st.cache_data(max_entries=200, ttl=1800, show_spinner=False)
def detect_contrast(dicom_path, file_size=None):
    """Return True if the series likely contains angiographic contrast.
    
    Normalizes pixels to [0, 255] float before computing statistics so that
    thresholds work correctly for both 8-bit and 16-bit DICOM series.
    """
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pa = dcm.pixel_array
        if len(pa.shape) == 4: pa = pa[:, :, :, 0]
        elif len(pa.shape) == 3 and pa.shape[2] in (3, 4):
            return True
        if len(pa.shape) == 2: pa = numpy.expand_dims(pa, axis=0)
        n = pa.shape[0]

        # Normalize the full array to [0, 255] float32 so thresholds are
        # bit-depth agnostic (works for 8-bit and 16-bit XA series).
        pa_f = pa.astype(numpy.float32)
        pmin, pmax = pa_f.min(), pa_f.max()
        if pmax > pmin:
            pa_f = (pa_f - pmin) / (pmax - pmin) * 255.0

        if n < 2:
            # Single frame — check gradient variance heuristic
            small = cv2.resize(pa_f[0], (128, 128))
            gx = numpy.diff(small, axis=1)
            gy = numpy.diff(small, axis=0)
            grad_var = float(numpy.var(gx)) + float(numpy.var(gy))
            return grad_var > 500

        # Multi-frame: compare temporal variance across frames
        sample_idxs = numpy.linspace(0, n - 1, min(n, 10), dtype=int)
        frames = [cv2.resize(pa_f[i], (64, 64)) for i in sample_idxs]
        stack = numpy.stack(frames, axis=0)
        temporal_var = float(numpy.var(stack, axis=0).mean())

        # Also check gradient of brightest frame
        frame_means = [float(f.mean()) for f in frames]
        best_idx = int(numpy.argmax(frame_means))
        best = frames[best_idx]
        gx = numpy.diff(best, axis=1)
        gy = numpy.diff(best, axis=0)
        grad_var = float(numpy.var(gx)) + float(numpy.var(gy))

        # Additional check: max inter-frame absolute difference
        # A contrast bolus causes a large pixel shift between adjacent frames.
        diffs = [float(numpy.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean())
                 for i in range(1, len(frames))]
        max_interframe_diff = float(numpy.max(diffs)) if diffs else 0.0

        # Peak-to-mean ratio: contrast makes one frame significantly brighter
        mean_frame_mean = float(numpy.mean(frame_means))
        peak_frame_mean = float(numpy.max(frame_means))
        peak_ratio = (peak_frame_mean / mean_frame_mean) if mean_frame_mean > 0 else 1.0

        return (temporal_var > 150
                or grad_var > 500
                or max_interframe_diff > 8.0   # >8/255 mean pixel shift between any two sampled frames
                or peak_ratio > 1.20)           # peak frame is >20% brighter than average
    except Exception:
        return True  # assume contrast if we can't tell

@st.cache_data(max_entries=100, ttl=1800, show_spinner=False)
def get_preview_image(dicom_path, start_ix, best_ix, end_ix, file_size=None):
    try:
        dcm = pydicom.dcmread(dicom_path, force=True)
        pa = dcm.pixel_array
        if len(pa.shape) == 4: pa = pa[:, :, :, 0]
        elif len(pa.shape) == 3 and pa.shape[2] in (3, 4):
            pa = pa[:, :, 0]
            pa = numpy.expand_dims(pa, axis=0)
        if len(pa.shape) == 2: pa = numpy.expand_dims(pa, axis=0)
        
        def _to_uint8(img):
            img_f = img.astype(numpy.float32)
            f_min, f_max = img_f.min(), img_f.max()
            if f_max > f_min:
                return ((img_f - f_min) / (f_max - f_min) * 255.0).astype(numpy.uint8)
            else:
                return numpy.zeros_like(img, dtype=numpy.uint8)
                
        frame_start = _to_uint8(pa[start_ix])
        frame_best  = _to_uint8(pa[best_ix])
        frame_end   = _to_uint8(pa[end_ix])
        img_start = cv2.resize(frame_start, (512, 512))
        img_best  = cv2.resize(frame_best,  (512, 512))
        img_end   = cv2.resize(frame_end,   (512, 512))
        return numpy.concatenate((img_start, img_best, img_end), axis=1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None



@st.fragment
def render_dicom_card(name, path, key_idx):
    meta = st.session_state.dicom_metadata[name]

    # ── Frame analysis (best frame detection, no contrast filtering) ──
    best_ix = meta.get("best_ix")
    if best_ix is None:
        best_ix, start_ix, end_ix, tfc, timi, just = analyze_series_flow(path, os.path.getsize(path))
        meta["best_ix"] = best_ix
        meta["start_ix"] = start_ix
        meta["end_ix"] = end_ix
        meta["tfc"] = tfc
        meta["timi"] = timi
        meta["just"] = just
        st.session_state.dicom_metadata[name] = meta
    else:
        start_ix = meta.get("start_ix", 0)
        end_ix = meta.get("end_ix", 0)
        tfc = meta.get("tfc", 0)
        timi = meta.get("timi", 3)
        just = meta.get("just", "N/A")

    _card_has_contrast = True  # show all series regardless of contrast detection
    chosen_badge = "⭐ " if meta.get("chosen_for_analysis", False) else ""
    
    with st.expander(f"{chosen_badge}{name}", expanded=True):
        try:
            # ── Chosen for Analysis checkbox ──────────────────────────
            prev_chosen = meta.get("chosen_for_analysis", False)
            meta["chosen_for_analysis"] = st.checkbox(
                "⭐ Chosen for Analysis",
                value=prev_chosen,
                key=f"chosen_{name}"
            )
            if meta["chosen_for_analysis"] != prev_chosen:
                if prev_chosen and not meta["chosen_for_analysis"]:
                    st.session_state.grid_order = None
                
                pat_id = st.session_state.get("patient_id")
                if pat_id:
                    chosen_dicoms = []
                    for k, m in st.session_state.dicom_metadata.items():
                        val = meta["chosen_for_analysis"] if k == name else m.get("chosen_for_analysis", False)
                        if val:
                            chosen_dicoms.append(k)
                    save_chosen_dicoms_to_firestore(pat_id, chosen_dicoms)

            # ── Procedure Phase ────────────────────────────────────────
            meta["phase"] = st.selectbox(
                "Procedure Phase", ["PRE-PCI", "POST-PCI"],
                index=0 if meta["phase"] == "PRE-PCI" else 1,
                key=f"phase_{name}"
            )

            # ── Vessel System + Segment (no labels, inline) ───────────
            PLACEHOLDER_SYSTEM = "— Select vessel —"
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
                vcol2.warning("← Select vessel")
                st.session_state.pop(f"seg_{name}", None)
            else:
                if chosen_system != prev_system:
                    meta["vessel_explicitly_set"] = True
                    st.session_state.pop(f"seg_{name}", None)
                    meta["aha_label"] = None
                    meta["aha"] = None
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
                st.caption("⚠️ No vessel selected — the sequence will not be grouped")

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
                    if st.session_state.get(f"play_{name}", False):
                        import tempfile
                        tmp_gif_path = os.path.join(tempfile.gettempdir(), f"angio_{name}_{os.path.getsize(path)}.gif")
                        if not os.path.exists(tmp_gif_path):
                            dcm = pydicom.dcmread(path, force=True)
                            pa = dcm.pixel_array
                            if len(pa.shape) == 4: pa = pa[:, :, :, 0]
                            if len(pa.shape) == 2: pa = numpy.expand_dims(pa, axis=0)
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
                        combined = get_preview_image(path, start_ix, best_ix, end_ix, os.path.getsize(path))
                        if combined is not None:
                            st.image(Image.fromarray(combined), use_column_width=True,
                                     caption="Start | Peak (QCA) | Last Frame")

                    calc_timi = str(timi)
                    current_ovr = meta.get("timi_override", "Auto")
                    if current_ovr not in ["Auto", "0", "1", "2", "3"]: current_ovr = "Auto"
                    meta["timi_override"] = st.selectbox("TIMI Grade", ["Auto", "0", "1", "2", "3"], index=["Auto", "0", "1", "2", "3"].index(current_ovr), key=f"timi_{name}")
                    st.caption(f"Calculated: Grade {timi} - {just}")
                    st.session_state.dicom_metadata[name] = meta

                    c_btn1, c_btn2, c_btn3 = st.columns(3)
                    vid_btn_text = "⏹️ Stop" if st.session_state.get(f"play_{name}", False) else "🎥 Play"
                    if c_btn1.button(vid_btn_text, key=f"play_btn_{name}", use_container_width=True):
                        st.session_state[f"play_{name}"] = not st.session_state.get(f"play_{name}", False)
                        st.rerun()
                    if c_btn2.button("🔬 Analyze (QCA)", key=f"btn_{name}", use_container_width=True):
                        if st.session_state.selected_dicom != path:
                            for k in list(st.session_state.keys()):
                                if k not in ['current_view', 'stage', 'dicom_registry', 'dicom_metadata', 'patient_cart', 'patient_id', 'firebase_init', 'firebase_error', 'firestore_db', 'firebase_bucket', 'firebase_api_key', 'user', 'app_mode', 'app_mode_select', 'importer_tailscale_nav_path', 'tailscale_nav_path']:
                                    del st.session_state[k]
                        st.session_state.current_view = 'analysis'
                        st.session_state.selected_dicom = path
                        st.session_state.dicomLabel = name
                        st.session_state.best_frame_ix = best_ix
                        st.rerun()
                    
                    try:
                        dl_filename = os.path.basename(path)
                        if not (dl_filename.lower().endswith(".dcm") or dl_filename.lower().endswith(".dicom")):
                            dl_filename += ".dcm"
                        if st.session_state.get(f"prepare_dl_{name}", False):
                            with open(path, "rb") as f:
                                dicom_bytes = f.read()
                            c_btn3.download_button(
                                "📥 Click to Download",
                                data=dicom_bytes,
                                file_name=dl_filename,
                                mime="application/dicom",
                                key=f"dl_dicom_{name}",
                                use_container_width=True
                            )
                        else:
                            if c_btn3.button("📥 Download DICOM", key=f"prep_dl_btn_{name}", use_container_width=True):
                                st.session_state[f"prepare_dl_{name}"] = True
                                st.rerun()
                    except Exception as e:
                        c_btn3.error("Read Error")

                except Exception as e:
                    st.warning(f"Error reading preview for {name}: {e}")
        except Exception as e:
            st.warning(f"Error rendering card for {name}: {e}")



# ── Admin Dashboard Routing ───────────────────────────────────────────────────
if st.session_state.get("app_mode") == "👑 Admin Panel" and st.session_state.user.get("role") == "admin":
    render_admin_panel()
    st.stop()
elif st.session_state.get("app_mode") == "📊 My Statistics":
    render_analyst_stats_panel()
    st.stop()
elif st.session_state.get("app_mode") == "📖 Instructions":
    render_instructions_panel()
    st.stop()

# ── GRID MODE ─────────────────────────────────────────────────────────────────
if st.session_state.current_view == 'grid':
    st.markdown("<h1 style='text-align: center;'>AngioPy Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Welcome to <b>AngioPy Segmentation</b>, an AI-driven, coronary angiography segmentation tool.</h5>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>DICOM Series Selection Grid</h2>", unsafe_allow_html=True)
    st.write("Automatically extracted optimal contrast frames from all available DICOMs.")
    
    selected_source = st.sidebar.radio(
        "Select DICOM Source",
        options=source_options,
        index=source_options.index(st.session_state.dicom_source) if st.session_state.dicom_source in source_options else 0,
        key="dicom_source_radio"
    )

    # Reactive transition when changing source
    if selected_source != st.session_state.dicom_source:
        st.session_state.dicom_source = selected_source
        st.session_state.dicom_registry = {}
        st.session_state.dicom_metadata = {}
        st.session_state.selected_tailscale_case = None
        st.session_state.tailscale_nav_path = ""
        st.session_state.tailscale_selected_locations = []
        st.session_state.importer_expanded = False
        if selected_source == "📁 Local Dicom/ Folder":
            st.session_state.dicom_registry = scan_dicom_folder("Dicom/")
        st.rerun()

    # Now render controls based on active source
    if st.session_state.dicom_source == "🌐 Tailscale Shared Folder":
        st.sidebar.subheader("Folder Explorer")
        
        base_dir = "/mnt/dane_dicom/"
        resolved_nav = resolve_clean_path_to_raw(st.session_state.tailscale_nav_path, base_dir)
        current_abs_path = os.path.abspath(os.path.join(base_dir, resolved_nav))
        if not current_abs_path.startswith(os.path.abspath(base_dir)):
            current_abs_path = os.path.abspath(base_dir)
            st.session_state.tailscale_nav_path = ""
            
        display_path = st.session_state.tailscale_nav_path if st.session_state.tailscale_nav_path != "" else "Root"
        st.sidebar.markdown(f"📁 **Location:** `{display_path}`")
        
        typed_path = st.sidebar.text_input("Or paste subfolder path:", value=st.session_state.tailscale_nav_path, placeholder="e.g. 4001/4001-0002")
        normalized_typed = typed_path.replace("\\", "/").strip().strip("/")
        if normalized_typed != st.session_state.tailscale_nav_path:
            resolved_typed = resolve_clean_path_to_raw(normalized_typed, base_dir)
            proposed_abs = os.path.abspath(os.path.join(base_dir, resolved_typed))
            if proposed_abs.startswith(os.path.abspath(base_dir)) and os.path.exists(proposed_abs):
                st.session_state.tailscale_nav_path = auto_detect_angio_subfolder(normalized_typed, base_dir)
                if st.session_state.tailscale_nav_path == ".":
                    st.session_state.tailscale_nav_path = ""
                st.rerun()
            elif typed_path:
                st.sidebar.error("⚠️ Directory does not exist.")
                
        subdirs = []
        try:
            if os.path.exists(current_abs_path):
                raw_dirs = []
                for entry in os.scandir(current_abs_path):
                    if entry.is_dir() and not entry.name.startswith('.'):
                        raw_dirs.append(entry.name)
                subdirs = filter_subdirs_by_assignments(st.session_state.tailscale_nav_path, raw_dirs)
        except Exception as e:
            st.sidebar.error(f"Error reading folder: {e}")
        
        col_nav1, col_nav2 = st.sidebar.columns([1, 1])
        with col_nav1:
            if st.session_state.tailscale_nav_path != "":
                if st.button("⬅️ Parent Dir", use_container_width=True, key="parent_dir_btn"):
                    parent = os.path.dirname(st.session_state.tailscale_nav_path)
                    st.session_state.tailscale_nav_path = parent
                    st.rerun()
        with col_nav2:
            if st.session_state.tailscale_nav_path != "":
                if st.button("🏠 Root", use_container_width=True, key="root_dir_btn"):
                    st.session_state.tailscale_nav_path = ""
                    st.rerun()
                    
        if subdirs:
            selected_subdir = st.sidebar.selectbox(
                "Go into subfolder:", 
                ["-- Select Subfolder --"] + subdirs, 
                key="go_into_subfolder_select",
                format_func=format_patient_folder_option
            )
            if selected_subdir != "-- Select Subfolder --":
                new_path = os.path.join(st.session_state.tailscale_nav_path, selected_subdir)
                st.session_state.tailscale_nav_path = auto_detect_angio_subfolder(new_path, base_dir)
                st.rerun()
        else:
            st.sidebar.info("No subfolders here.")
            
        # Show preview files list recursively if a patient is selected (len(parts) >= 2)
        parts = [p for p in st.session_state.tailscale_nav_path.replace("\\", "/").split("/") if p]
        if len(parts) >= 2:
            show_preview = st.sidebar.checkbox("🔍 Show files preview", value=False, key="tailscale_show_files_preview")
            if show_preview:
                max_preview_files = 100
                files_found, total_files_count = get_recursive_preview_files_cached(st.session_state.tailscale_nav_path, base_dir)
                                        
                if files_found:
                    with st.sidebar.expander("📄 Files in selected folder", expanded=True):
                        st.markdown(f"**Number of files on disk:** {total_files_count}")
                        if total_files_count > max_preview_files:
                            st.caption(f"Showing first {max_preview_files} files:")
                        for rel_name, sz_str, path in sorted(files_found, key=lambda x: x[0]):
                            st.markdown(f"- `{rel_name}` {sz_str}", help=f"Ścieżka: {path}")
            
        # --- Multi-Location Management ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("📍 **Selected Locations (max 5):**")
        
        if len(st.session_state.tailscale_selected_locations) < 5:
            curr_nav = st.session_state.tailscale_nav_path
            is_added = curr_nav in st.session_state.tailscale_selected_locations
            if not is_added:
                curr_loc_disp = curr_nav if curr_nav != "" else "Root"
                if st.sidebar.button(f"➕ Add '{curr_loc_disp}' folder", use_container_width=True, key="add_location_btn"):
                    st.session_state.tailscale_selected_locations.append(curr_nav)
                    st.rerun()
            else:
                st.sidebar.caption("📍 Current folder is already added")
        else:
            st.sidebar.warning("Maximum of 5 locations reached!")
            
        # Display and remove added locations
        if st.session_state.tailscale_selected_locations:
            for idx, loc in enumerate(st.session_state.tailscale_selected_locations):
                loc_disp = loc if loc != "" else "Root"
                c_loc_txt, c_loc_btn = st.sidebar.columns([4, 1])
                c_loc_txt.markdown(f"• `{loc_disp}`")
                if c_loc_btn.button("❌", key=f"remove_loc_{idx}"):
                    st.session_state.tailscale_selected_locations.pop(idx)
                    st.rerun()
        else:
            st.sidebar.caption("No locations added. Will default to currently navigated folder.")
            
        st.sidebar.markdown("---")
        if st.sidebar.button("📂 Load DICOM files", use_container_width=True, type="primary", key="load_dicoms_btn"):
            load_dicom_data.clear()
            get_recursive_preview_files_cached.clear()
            st.session_state.importer_expanded = False
            
            # Determine selected locations
            if st.session_state.tailscale_selected_locations:
                locs = st.session_state.tailscale_selected_locations
            else:
                locs = [st.session_state.tailscale_nav_path]
                
            # Verify total size of files to copy does not exceed 100 GB
            total_size_to_copy = 0
            for loc in locs:
                if not check_cache_status_and_heal(loc, base_dir):
                    for r_loc in get_raw_folders_for_clean_path(loc, base_dir):
                        src_abs = os.path.abspath(os.path.join(base_dir, r_loc))
                        total_size_to_copy += get_dir_size(src_abs)
                        
            max_allowed_copy_bytes = 100 * 1024 * 1024 * 1024
            if total_size_to_copy > max_allowed_copy_bytes:
                st.sidebar.error(f"⚠️ Selected data size ({total_size_to_copy / (1024**3):.2f} GB) exceeds the 100 GB limit.")
                st.stop()
                
            registry = {}
            for loc in locs:
                dst_abs = os.path.abspath(os.path.join("./tailscale_cache", loc))
                
                # Check if patient folder is already cached on the server
                if check_cache_status_and_heal(loc, base_dir):
                    # Already cached!
                    touch_directory(dst_abs)
                    # Scan files from local cache instead of Tailscale
                    loc_registry = scan_dicom_folder(dst_abs)
                else:
                    # Not cached, copy from Tailscale to local cache
                    # Clear dirty cache directory if it exists
                    if os.path.exists(dst_abs) and os.path.isdir(dst_abs):
                        import shutil
                        try:
                            shutil.rmtree(dst_abs)
                        except Exception as e:
                            print(f"Error clearing incomplete cache {dst_abs}: {e}")
                            
                    r_locs = get_raw_folders_for_clean_path(loc, base_dir)
                    status_meta = st.sidebar.empty()
                    progress_meta = st.sidebar.progress(0, text="Caching Tailscale files...")
                    
                    for idx_r, r_loc in enumerate(r_locs):
                        src_abs = os.path.abspath(os.path.join(base_dir, r_loc))
                        status_meta.markdown(f"📥 **Caching subfolder {idx_r + 1}/{len(r_locs)}:** `{os.path.basename(r_loc)}`")
                        copy_folder_with_progress(src_abs, dst_abs, status_meta, progress_meta)
                        
                    status_meta.empty()
                    progress_meta.empty()
                    
                    # Write sentinel to mark cache as complete
                    sentinel_path = os.path.join(dst_abs, ".cache_complete")
                    try:
                        os.makedirs(dst_abs, exist_ok=True)
                        with open(sentinel_path, "w") as f:
                            f.write("completed")
                    except Exception as e:
                        print(f"Error writing sentinel {sentinel_path}: {e}")
                        
                    touch_directory(dst_abs)
                    check_and_cleanup_cache("./tailscale_cache")
                    
                    loc_registry = scan_dicom_folder(dst_abs)
                
                registry.update(loc_registry)
            
            # Display name for selected cases
            if st.session_state.tailscale_selected_locations:
                loc_names = [l if l != "" else "Root" for l in st.session_state.tailscale_selected_locations]
                st.session_state.selected_tailscale_case = f"Multi [{', '.join(loc_names)}]"
            else:
                st.session_state.selected_tailscale_case = st.session_state.tailscale_nav_path
                
            load_registry_with_progress(registry)
            st.rerun()

    elif st.session_state.dicom_source == "📁 Local Dicom/ Folder":
        if not st.session_state.dicom_registry:
            registry = scan_dicom_folder("Dicom/")
            load_registry_with_progress(registry)
            
        if st.sidebar.button("🔄 Refresh DICOM List", use_container_width=True, key="refresh_dicom_list_btn"):
            load_dicom_data.clear()
            get_recursive_preview_files_cached.clear()
            registry = scan_dicom_folder("Dicom/")
            load_registry_with_progress(registry)
            st.rerun()

    elif st.session_state.dicom_source == "💻 Upload Local Files":
        uploadedDicoms = st.sidebar.file_uploader("Upload DICOM file(s)", key="gridDicomUploader", accept_multiple_files=True)
        if uploadedDicoms:
            import tempfile
            if "uploaded_dicom_names" not in st.session_state:
                st.session_state.uploaded_dicom_names = set()
            new_files_registry = {}
            for uploadedDicom in uploadedDicoms:
                if uploadedDicom.name not in st.session_state.uploaded_dicom_names:
                    tmpPath = os.path.join(tempfile.gettempdir(), uploadedDicom.name)
                    with open(tmpPath, "wb") as f:
                        f.write(uploadedDicom.getbuffer())
                    new_files_registry[uploadedDicom.name] = tmpPath
                    st.session_state.uploaded_dicom_names.add(uploadedDicom.name)
            
            if new_files_registry:
                load_registry_with_progress(new_files_registry)
                st.rerun()
        if "uploaded_dicom_names" in st.session_state and st.session_state.uploaded_dicom_names:
            st.sidebar.caption(f"📂 Loaded: {len(st.session_state.dicom_registry)} files")
            if st.sidebar.button("🗑️ Delete uploaded files", key="clear_uploads_btn"):
                for name in list(st.session_state.uploaded_dicom_names):
                    st.session_state.dicom_registry.pop(name, None)
                st.session_state.uploaded_dicom_names.clear()
                st.session_state.original_dicom_registry = st.session_state.dicom_registry.copy()
                st.session_state.is_dicom_registry_filtered = False
                st.rerun()

    if st.sidebar.button("🔃 Sort sequences", use_container_width=True, key="sidebar_sort_btn"):
        st.session_state._sidebar_sort_requested = True

    render_ecrf_sidebar()
    render_cache_importer_sidebar()
    render_loaded_files_sidebar()

    if not st.session_state.dicom_registry:
        if st.session_state.dicom_source == "🌐 Tailscale Shared Folder":
            st.info("🌐 **Tailscale Shared Folder mode active**\nPlease use the sidebar navigation on the left to locate and load a patient folder containing DICOM files.")
        elif st.session_state.dicom_source == "📁 Local Dicom/ Folder":
            st.warning("No DICOM files found in Dicom/ folder.")
        else:
            st.warning("No DICOM files uploaded yet. Please upload DICOM files in the sidebar.")
        st.stop()
        
    if st.session_state.get("show_saved_toast"):
        st.toast("💾 PDF report saved successfully and Excel database updated!", icon="✅")
        st.session_state.pop("show_saved_toast", None)
                
    col_pid, col_finish = st.columns([3, 1])
    with col_pid:
        st.session_state.patient_id = st.text_input("Patient ID Number", st.session_state.patient_id)
    with col_finish:
        st.write("##") # alignment spacer
        is_finish_enabled = bool(st.session_state.get("dicom_registry") and st.session_state.get("patient_id"))
        if st.button("🏁 Finish Patient Analysis", use_container_width=True, type="primary", key="btn_finish_patient_analysis", disabled=not is_finish_enabled):
            confirm_finish_dialog()

    # ── Stenosis Checklist Widget ─────────────────────────────────────────────
    if st.session_state.get("firebase_init") and st.session_state.get("firestore_db") is not None and st.session_state.get("patient_id"):
        patient_id = st.session_state.patient_id.strip()
        if patient_id:
            try:
                db = st.session_state.firestore_db
                reports_ref = db.collection("analysis_results")\
                                .where("patient_id", "==", patient_id)\
                                .stream()
                db_reports = []
                for r in reports_ref:
                    d = r.to_dict()
                    if d.get("dicom_name") != "marked_completed" and d.get("phase") != "COMPLETED":
                        db_reports.append(d)
                
                db_reports.sort(key=lambda x: x.get("timestamp") or 0)
                
                with st.expander("📋 Saved Stenoses Checklist", expanded=True):
                    if not db_reports:
                        st.info("⚪ No saved analyses found in Firestore for this patient ID yet.")
                    else:
                        st.markdown(f"**Saved stenoses in database for Patient ID `{patient_id}`:**")
                        for idx, rep in enumerate(db_reports):
                            vessel = rep.get("vessel") or "N/A"
                            aha = rep.get("aha") or "N/A"
                            phase = rep.get("phase") or "PRE-PCI"
                            dcm_name = rep.get("dicom_name") or "N/A"
                            metrics = rep.get("metrics") or {}
                            mld = metrics.get("mld_mm")
                            ref = metrics.get("ref_diam_mm")
                            pct_ds = metrics.get("pct_diameter_stenosis")
                            
                            mld_str = f"{mld:.2f} mm" if isinstance(mld, (int, float)) else "N/A"
                            ref_str = f"{ref:.2f} mm" if isinstance(ref, (int, float)) else "N/A"
                            pct_ds_str = f"{pct_ds:.1f}%" if isinstance(pct_ds, (int, float)) else "N/A"
                            
                            phase_color = "🔴" if phase == "PRE-PCI" else "🔵"
                            st.markdown(f"- {phase_color} **Stenosis {idx + 1} ({phase})**: `{vessel}` (AHA {aha}) | MLD: `{mld_str}`, Ref: `{ref_str}`, DS: `{pct_ds_str}` (DICOM: `{dcm_name}`)")
            except Exception as e:
                st.error(f"Error loading checklist from database: {e}")
    
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
    # Filter display_list to only include files that exist in the active registry
    active_keys = set(st.session_state.dicom_registry.keys())
    display_list = [item for item in display_list if item[0] in active_keys]
    
    # keep display_list in sync with registry (handle newly added DICOMs)
    display_names = {n for n, _, _ in display_list}
    for item in temp_sort_list:
        if item[0] not in display_names:
            display_list = list(display_list) + [item]

                
    firestore_pdf_bytes = None
    has_db_reports = False
    if st.session_state.get("firebase_init") and st.session_state.get("firestore_db") is not None and st.session_state.get("patient_id"):
        pid = st.session_state.patient_id.strip()
        if pid:
            firestore_pdf_bytes = get_master_pdf_bytes(pid)
            has_db_reports = (firestore_pdf_bytes is not None)
            
    has_cart = len(st.session_state.patient_cart) > 0
    
    if has_cart or has_db_reports:
        st.markdown("---")
        if has_db_reports:
            st.success(f"📄 **Persistent Master Report loaded from Firebase** (Patient ID: `{st.session_state.patient_id}`). You can export the consolidated report below.")
        else:
            st.success(f"🛒 **{len(st.session_state.patient_cart)} sequence(s) analyzed** and saved to the Patient Master Report Cart.")
            
        pdf_bytes_to_download = None
        if has_db_reports:
            pdf_bytes_to_download = firestore_pdf_bytes
        else:
            # Fallback to local session cart compilation
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
            pdf_bytes_to_download = masterPdfBuf.getvalue()

        c_dl_master1, c_dl_master2 = st.columns(2)
        with c_dl_master1:
            if pdf_bytes_to_download:
                st.download_button("📄 Export Master Patient PDF", data=pdf_bytes_to_download, file_name=f"Patient_{st.session_state.patient_id if st.session_state.patient_id else 'Report'}_Master.pdf", mime="application/pdf", use_container_width=True, key="btn_download_master_pdf")
        with c_dl_master2:
            save_dir = st.session_state.get("report_save_dir", os.path.abspath("./reports/"))
            target_xlsx = os.path.join(save_dir, "AngioPy.xlsx")
            if os.path.exists(target_xlsx):
                with open(target_xlsx, "rb") as f:
                    xlsx_bytes = f.read()
                st.download_button("📊 Export Excel Database (AngioPy.xlsx)", data=xlsx_bytes, file_name="AngioPy.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="grid_export_xlsx_btn")
        st.markdown("---")

    # --- DICOM View Filtering Widget ---
    is_filtered = st.session_state.get("is_dicom_registry_filtered", False)
    if not is_filtered:
        if st.button("🗑️ Clear DICOMs not chosen for analysis", use_container_width=True, key="btn_filter_dicoms"):
            chosen_names = [name for name, m in st.session_state.dicom_metadata.items() if m.get("chosen_for_analysis", False)]
            if not chosen_names:
                st.warning("⚠️ Please choose at least one DICOM sequence for analysis (check the '⭐ Chosen for Analysis' box) before hiding the rest.")
            else:
                st.session_state.original_dicom_registry = st.session_state.dicom_registry.copy()
                st.session_state.dicom_registry = {name: path for name, path in st.session_state.dicom_registry.items() if name in chosen_names}
                st.session_state.is_dicom_registry_filtered = True
                patient_id = st.session_state.get("patient_id")
                if patient_id:
                    save_chosen_dicoms_to_firestore(patient_id, chosen_names)
                st.rerun()
    else:
        c_info, c_restore = st.columns([2, 1])
        with c_info:
            st.info("ℹ️ Showing only sequences chosen for analysis.")
        with c_restore:
            if st.button("🔓 Load all sequences", use_container_width=True, type="secondary", key="btn_restore_dicoms"):
                st.session_state.show_all_sequences = True
                st.session_state.pop("dicom_registry", None)
                st.rerun()

    cols = st.columns(1)
    for key_idx, (name, path, _) in enumerate(display_list):
        with cols[0]:
            render_dicom_card(name, path, key_idx)

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
        f"⭐ {n_chosen} selected &nbsp;|&nbsp; 🩺 {n_assigned} with vessel assigned</div>",
        unsafe_allow_html=True
    )
    if sort_col2.button("🔃 Sort sequences", use_container_width=True):
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

# Render permanently available eCRF Sidebar
render_ecrf_sidebar()
render_cache_importer_sidebar()
render_loaded_files_sidebar()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Segmentation", "Analysis"])

st.markdown('''<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size:16px; }
</style>''', unsafe_allow_html=True)

# ── Load DICOM ────────────────────────────────────────────────────────────────
if selectedDicom is not None:
    try:
        pixelArray, dist_p, dist_d, spacing, cine_rate, rec_rate = load_dicom_data(selectedDicom)
        dcm = DicomMetadataMock(spacing, dist_p, dist_d, cine_rate, rec_rate)
        n_slices = pixelArray.shape[0]
    except Exception as e:
        st.error(f"Could not load DICOM file: {e}")
        st.stop()

    default_ix = st.session_state.best_frame_ix if st.session_state.best_frame_ix < n_slices else int(n_slices / 2)
    
    def _update_all_widgets(val):
        for suffix in ["stepOne", "stepTwo"]:
            st.session_state[f"num_frame_{suffix}"] = val
            st.session_state[f"slider_frame_{suffix}"] = val

    if 'current_slice_ix' not in st.session_state or st.session_state.get('last_dicom_for_slice') != selectedDicom:
        st.session_state.current_slice_ix = default_ix
        st.session_state.last_dicom_for_slice = selectedDicom
        _update_all_widgets(default_ix)

    def _change_frame(delta):
        new_val = st.session_state.current_slice_ix + delta
        if 0 <= new_val < n_slices:
            st.session_state.current_slice_ix = new_val
            _update_all_widgets(new_val)

    def _sync_frame(key):
        new_val = st.session_state[key]
        st.session_state.current_slice_ix = new_val
        _update_all_widgets(new_val)

    def _render_frame_controls(suffix):
        if n_slices > 1:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                st.button("◀", key=f"prev_frame_{suffix}", on_click=_change_frame, args=(-1,), use_container_width=True)
            with c2:
                if f"num_frame_{suffix}" not in st.session_state:
                    st.session_state[f"num_frame_{suffix}"] = st.session_state.current_slice_ix
                st.number_input("Frame", min_value=0, max_value=n_slices-1, 
                                key=f"num_frame_{suffix}", 
                                label_visibility="collapsed",
                                on_change=_sync_frame, args=(f"num_frame_{suffix}",))
            with c3:
                st.button("▶", key=f"next_frame_{suffix}", on_click=_change_frame, args=(1,), use_container_width=True)
            if f"slider_frame_{suffix}" not in st.session_state:
                st.session_state[f"slider_frame_{suffix}"] = st.session_state.current_slice_ix
            st.slider('Frame Slider', 0, n_slices - 1,
                      key=f"slider_frame_{suffix}", label_visibility="collapsed",
                      on_change=_sync_frame, args=(f"slider_frame_{suffix}",))

    with stepOne:
        st.write("Select frame for annotation. The optimal frame has been pre-selected.")
        if n_slices > 1:
            _render_frame_controls("stepOne")
            slice_ix = st.session_state.current_slice_ix
        else:
            slice_ix = 0
            
        _mask_key = f"predictedMask_{st.session_state.get('dicomLabel','')}_{slice_ix}"
        if _mask_key in st.session_state:
            predictedMask = st.session_state[_mask_key]
        else:
            predictedMask = numpy.zeros_like(pixelArray[slice_ix, :, :])

    with stepTwo:
        meta = st.session_state.dicom_metadata.get(st.session_state.dicomLabel, {}) if "dicomLabel" in st.session_state else {}
        _cur_sys = meta.get("vessel_system")
        if _cur_sys and _cur_sys in ALL_SYSTEM_NAMES:
            selectedArtery = _colour_key_from_system(_cur_sys)
            st.info(
                f"**Vessel:** {_cur_sys}  \n"
                f"**Segment:** {meta.get('aha_label', '—')}  |  **AHA:** {meta.get('aha', '—')}  |  "
                f"**FFR:** {meta.get('ffr_registered', 'N/A')}"
            )
        else:
            selectedArtery = "LAD"
            st.warning("⚠️ **No vessel selected.** Go back to the selection grid and assign a vessel to this sequence before QCA analysis.")
        st.write("Beginning with the desired start point and finishing at the desired end point, click along the artery aiming for ~5-10 points.")
        if n_slices > 1:
            st.write("Adjust frame if needed:")
            _render_frame_controls("stepTwo")



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
                    "Catheter size:",
                    list(CATHETER_SIZES.keys()),
                    key="catheterSize"
                )
                catheterMm = CATHETER_SIZES[catheterChoice]

                c_head1, c_head2 = st.columns([3, 1])
                c_head1.caption("Select two points (start and end) along the catheter to define the calibration segment.")
                if "calib_key_suffix" not in st.session_state:
                    st.session_state.calib_key_suffix = 0

                if c_head2.button("🔄 Redo Calib"):
                    for suffix in range(st.session_state.calib_key_suffix + 1):
                        st.session_state.pop(f"canvas_calib_{dicomLabel}_{suffix}", None)
                    for k in ["calibPoints", "mmPerPixelCalib", "calibLinePx"]:
                        st.session_state.pop(k, None)
                    st.session_state.calib_key_suffix += 1
                    st.rerun()

                # Pobieramy punkty i rysujemy na żywo NA LEWYM obrazie!
                calibBgFrame = selectedFrameRGB.copy()
                calibPointsList = st.session_state.get('calibPoints', [])
                if calibPointsList:
                    for item in calibPointsList:
                        if isinstance(item, dict):
                            ref_pt1, ref_pt2 = item['ref']
                            cv2.line(calibBgFrame, ref_pt1, ref_pt2, (0, 255, 0), 1)
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
                    key=f"canvas_calib_{dicomLabel}_{st.session_state.calib_key_suffix}",
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
                        r1 = dotObjs.iloc[0].get("radius", 0)
                        if pd.isna(r1): r1 = 0
                        r1 = float(r1)

                        r2 = dotObjs.iloc[1].get("radius", 0)
                        if pd.isna(r2): r2 = 0
                        r2 = float(r2)

                        cx1 = int((float(dotObjs.iloc[0].get("left", 0)) + r1) * _cs)
                        cy1 = int((float(dotObjs.iloc[0].get("top",  0)) + r1) * _cs)
                        cx2 = int((float(dotObjs.iloc[1].get("left", 0)) + r2) * _cs)
                        cy2 = int((float(dotObjs.iloc[1].get("top",  0)) + r2) * _cs)

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
                            
                            sf_f = selectedFrame.astype(float)
                            sf_min = sf_f.min()
                            norm_frame = (sf_f - sf_min) / (sf_f.max() - sf_min + 1e-5) * 255.0
                            
                            profile = scipy.ndimage.map_coordinates(norm_frame, coords, mode='nearest')
                            smoothed = numpy.convolve(profile, numpy.ones(3)/3.0, mode='same')
                            grad = numpy.abs(numpy.gradient(smoothed))
                            
                            bestDiam = 15 # fallback diameter in pixels
                            peaks = scipy.signal.find_peaks(grad, prominence=1.0, distance=4)[0]
                            if len(peaks) >= 2:
                                left_peaks = [p for p in peaks if p < 60]
                                right_peaks = [p for p in peaks if p >= 60]
                                if left_peaks and right_peaks:
                                    p1 = max(left_peaks)
                                    p2 = min(right_peaks)
                                    bestDiam = abs(get_subpixel_peak(p2, grad) - get_subpixel_peak(p1, grad))
                            
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
                                track_L = max(10, int(bestDiam * 1.2))
                                track_t = numpy.arange(-track_L, track_L)
                                tx = ccx + track_t * numpy.cos(cs_theta)
                                ty = ccy + track_t * numpy.sin(cs_theta)
                                t_coords = numpy.vstack((ty, tx))
                                t_prof = scipy.ndimage.map_coordinates(norm_frame, t_coords, mode='nearest')
                                t_smooth = numpy.convolve(t_prof, numpy.ones(3)/3.0, mode='same')
                                t_grad = numpy.abs(numpy.gradient(t_smooth))
                                
                                t_peaks = scipy.signal.find_peaks(t_grad, prominence=1.0, distance=max(2, int(bestDiam*0.3)))[0]
                                if len(t_peaks) >= 2:
                                    left_peaks = [p for p in t_peaks if p < track_L]
                                    right_peaks = [p for p in t_peaks if p >= track_L]
                                    if left_peaks and right_peaks:
                                        tp0 = max(left_peaks)
                                        tp1 = min(right_peaks)
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
                                    'diam': avg_diam,
                                    'click1': (cx1, cy1),
                                    'click2': (cx2, cy2)
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
                if "seg_key_suffix" not in st.session_state:
                    st.session_state.seg_key_suffix = 0

                if a_head2.button("🔄 Redo Segment"):
                    for suffix in range(st.session_state.seg_key_suffix + 1):
                        st.session_state.pop(f"canvas_seg_{dicomLabel}_{suffix}", None)
                    for k in [_mask_key, f"objects_{_mask_key}"]:
                        st.session_state.pop(k, None)
                    st.session_state.seg_key_suffix += 1
                    st.rerun()

                # Draw green mask contour on the left canvas background image if it is already computed
                annot_bg = selectedFrameRGB.copy()
                if numpy.sum(predictedMask) > 0:
                    if len(predictedMask.shape) == 3:
                        v_mask = (numpy.any(predictedMask > 0, axis=-1) * 255).astype(numpy.uint8)
                    else:
                        v_mask = (predictedMask > 0).astype(numpy.uint8) * 255
                    contour_preview = angioPyFunctions.maskOutliner(labelledArtery=v_mask, outlineThickness=1)
                    annot_bg[contour_preview, :] = [0, 255, 0]

                annotationCanvas = st_canvas(
                    fill_color="red",
                    stroke_width=2,
                    stroke_color="red",
                    background_color='black',
                    background_image=Image.fromarray(annot_bg),
                    update_streamlit=True,
                    height=600,
                    width=600,
                    drawing_mode="point",
                    point_display_radius=2,
                    key=f"canvas_seg_{dicomLabel}_{st.session_state.seg_key_suffix}",
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

                        # Check if points have changed compared to the cached mask
                        cached_objects = st.session_state.get(f"objects_{_mask_key}", None)
                        points_changed = True
                        if cached_objects is not None and len(objects) == len(cached_objects):
                            try:
                                if objects[['left', 'top']].equals(cached_objects[['left', 'top']]):
                                    points_changed = False
                            except Exception:
                                pass

                        if points_changed:
                            st.info("📍 New annotation points selected. Click the button below to run segmentation.")
                            if st.button("🚀 Run Artery Segmentation", type="primary", key=f"run_seg_{_mask_key}", use_container_width=True):
                                with st.spinner(f"Running segmentation on {len(objects)} points (30–60 s on CPU)…"):
                                    try:
                                        mask = angioPyFunctions.arterySegmentation(
                                            selectedFrame,
                                            groundTruthPoints,
                                        )
                                        import predict
                                        predictedMask = predict.CoronaryDataset.mask2image(mask)
                                        predictedMask = numpy.array(predictedMask)
                                        st.session_state[_mask_key] = predictedMask
                                        st.session_state[f"objects_{_mask_key}"] = objects
                                        # Force both canvases to refresh so the mask outline is displayed instantly
                                        st.session_state.seg_key_suffix = st.session_state.get("seg_key_suffix", 0) + 1
                                        st.session_state.canvas_key_suffix = st.session_state.get("canvas_key_suffix", 0) + 1
                                        st.rerun()
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

                # Reset cumulative_xor if shape doesn't match current mask
                if st.session_state.cumulative_xor.shape != v_mask_base.shape:
                    st.session_state.cumulative_xor = numpy.zeros_like(v_mask_base, dtype=bool)

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
    if numpy.sum(predictedMask) > 0:
        b_channel, g_channel, r_channel = cv2.split(predictedMask)
        a_channel = numpy.full_like(predictedMask[:, :, 0], fill_value=255)
        predictedMaskRGBA = cv2.merge((predictedMask, a_channel))

        with tab2:
            current_sig = f"{selectedDicom}_{slice_ix}_{selectedArtery}"

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

            # Arc length along centreline (needed by Interpolated method and sliders)
            spX    = splinePointsX[clippingLength:-clippingLength]
            spY    = splinePointsY[clippingLength:-clippingLength]
            _diffs  = numpy.sqrt(numpy.diff(spX) ** 2 + numpy.diff(spY) ** 2)
            cumLen = numpy.concatenate([[0], numpy.cumsum(_diffs)])  # in 512-scale pixels

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

            def on_swap_toggle():
                _new_swap = st.session_state.get(f"swap_chk_{current_sig}", False)
                _p_key = f"sl_prox_{current_sig}"
                _d_key = f"sl_dist_{current_sig}"
                if _p_key in st.session_state and _d_key in st.session_state:
                    _p = st.session_state[_p_key]
                    _d = st.session_state[_d_key]
                    st.session_state[_p_key] = _d
                    st.session_state[_d_key] = _p
                _corr_key = f"corr_{current_sig}"
                if _corr_key in st.session_state and st.session_state[_corr_key] is not None:
                    _saved = st.session_state[_corr_key]
                    if _saved.get("swapped", False) != _new_swap:
                        _p = _saved["prox"]
                        _d = _saved["dist"]
                        st.session_state[_corr_key]["prox"] = _d
                        st.session_state[_corr_key]["dist"] = _p
                        st.session_state[_corr_key]["swapped"] = _new_swap

            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 2])
            with ctrl1:
                st.radio("⏳ Procedure Phase", ["PRE-PCI", "POST-PCI"],
                         index=0 if current_phase == "PRE-PCI" else 1,
                         key="analysis_phase_toggle",
                         horizontal=True,
                         on_change=update_phase)
            swap_direction = ctrl2.checkbox("🔄 Swap Prox/Dist", value=False, key=f"swap_chk_{current_sig}", on_change=on_swap_toggle)
            ostial_lesion  = ctrl3.checkbox("🚨 Ostial Lesion",  value=False)
            with ctrl4:
                ref_method = st.radio("📐 Reference Method", ["Interpolated", "Average"],
                                      index=0, horizontal=True, key="ref_method_radio")

            # ── QCA indices ──────────────────────────────────────────────────
            refLen   = max(1, int(len(vesselThicknesses) * 0.20))
            edge_avg = max(1, int(len(vesselThicknesses) * 0.05))

            if current_phase == "POST-PCI":
                prox_raw = edge_avg
                dist_raw = len(vesselThicknesses) - edge_avg - 1
            else:
                prox_raw = numpy.argmax(vesselThicknesses[:refLen])
                dist_raw = len(vesselThicknesses) - refLen + numpy.argmax(vesselThicknesses[-refLen:])

            if swap_direction:
                prox_idx, dist_idx = dist_raw, prox_raw
            else:
                prox_idx, dist_idx = prox_raw, dist_raw

            idx_start = min(prox_idx, dist_idx)
            idx_end   = max(prox_idx, dist_idx)
            if ostial_lesion and current_phase != "POST-PCI":
                idx_start = 0

            if idx_start >= idx_end:
                idx_start, idx_end = 0, len(vesselThicknesses) - 1

            mld_idx = idx_start + numpy.argmin(vesselThicknesses[idx_start:idx_end+1])

            # Keep auto values for revert/fallback
            auto_prox_idx = int(prox_idx)
            auto_dist_idx = int(dist_idx)
            auto_mld_idx  = int(mld_idx)

            # Check for manual landmark corrections in session state
            _corr_key = f"corr_{current_sig}"
            if _corr_key not in st.session_state:
                st.session_state[_corr_key] = None

            if st.session_state[_corr_key] is not None:
                _saved = st.session_state[_corr_key]
                prox_idx = int(_saved["prox"])
                dist_idx = int(_saved["dist"])
                mld_idx  = int(_saved["mld"])
                _landmarks_corrected = True
            else:
                _landmarks_corrected = False

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
            if _landmarks_corrected:
                st.markdown("<p style='text-align:center; color:#f0c040; font-size:13px;'>🔧 Corrected landmark positions shown above</p>", unsafe_allow_html=True)

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

            # ── Landmark correction UI ─────────────────────────────────────────
            _n_pts = len(vesselThicknesses)
            with st.expander("📍 Correct landmark positions", expanded=True):
                st.caption("Drag sliders to reposition Proximal Reference, Distal Reference, and MLD along the centreline. Use ↩ Revert to restore automatic detection.")

                if st.session_state[_corr_key] is not None:
                    _prox_def = int(prox_idx)
                    _dist_def = int(dist_idx)
                    _mld_def  = int(mld_idx)
                else:
                    _prox_def = auto_prox_idx
                    _dist_def = auto_dist_idx
                    _mld_def  = auto_mld_idx

                _sc1, _sc2, _sc3 = st.columns(3)
                _prox_override = _sc1.slider("Proximal Ref index", 0, _n_pts-1, _prox_def, key=f"sl_prox_{current_sig}")
                _dist_override = _sc2.slider("Distal Ref index",   0, _n_pts-1, _dist_def, key=f"sl_dist_{current_sig}")
                _mld_override  = _sc3.slider("MLD index",          0, _n_pts-1, _mld_def,  key=f"sl_mld_{current_sig}")

                def on_auto_mld_cb(v_thick, ostial, swap_dir):
                    _p_key = f"sl_prox_{current_sig}"
                    _d_key = f"sl_dist_{current_sig}"
                    _p_val = st.session_state.get(_p_key, 0)
                    _d_val = st.session_state.get(_d_key, 0)
                    
                    _idx_start = min(_p_val, _d_val)
                    _idx_end   = max(_p_val, _d_val)
                    if ostial:
                        _idx_start = 0
                    _new_mld = _idx_start + int(numpy.argmin(v_thick[_idx_start:_idx_end+1]))
                    
                    st.session_state[f"sl_mld_{current_sig}"] = _new_mld
                    st.session_state[_corr_key] = {
                        "prox": _p_val,
                        "dist": _d_val,
                        "mld":  _new_mld,
                        "swapped": swap_dir
                    }

                def on_revert_auto_cb(p_idx, d_idx, m_idx):
                    st.session_state[_corr_key] = None
                    st.session_state[f"sl_prox_{current_sig}"] = p_idx
                    st.session_state[f"sl_dist_{current_sig}"] = d_idx
                    st.session_state[f"sl_mld_{current_sig}"]  = m_idx

                _btn1, _btn3, _btn2 = st.columns(3)
                if _btn1.button("✅ Apply corrections", key=f"apply_corr_{current_sig}", use_container_width=True):
                    st.session_state[_corr_key] = {
                        "prox": _prox_override,
                        "dist": _dist_override,
                        "mld":  _mld_override,
                        "swapped": swap_direction
                    }
                    st.rerun()
                _btn3.button("🤖 Auto MLD", key=f"auto_mld_{current_sig}", use_container_width=True,
                             on_click=on_auto_mld_cb, args=(vesselThicknesses, ostial_lesion, swap_direction))
                _btn2.button("↩ Revert to Auto", key=f"revert_corr_{current_sig}", use_container_width=True,
                             disabled=st.session_state[_corr_key] is None,
                             on_click=on_revert_auto_cb, args=(auto_prox_idx, auto_dist_idx, auto_mld_idx))

                if st.session_state[_corr_key] is not None:
                    st.success("🔧 Using **corrected** landmark positions")
                else:
                    st.info("ℹ️ Using **automatic** landmark positions")

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
            if mmPerPixelCalib:
                # Catheter calibration: measured on 512-canvas → already mm/512px, no rescaling
                pxToMm = mmPerPixelCalib
            else:
                # DICOM metadata: mm/original-pixel → scale to 512-space
                pxToMm = (origW / 512.0) * mmPerPixel if mmPerPixel else None

            # Resolve phase from metadata HERE (needed before QCA branching)
            meta  = st.session_state.dicom_metadata.get(dicomLabel, {})
            phase = meta.get("phase", "UNKNOWN")

            refLen = max(1, int(len(vesselThicknesses) * 0.20))

            is_post_pci = phase == "POST-PCI"

            if is_post_pci:
                # POST-PCI: measure at mask edges (proximal = start of centreline, distal = end)
                if _landmarks_corrected:
                    proxDiamMm = vesselThicknesses[prox_idx] * (pxToMm or 1.0)
                    distDiamMm = vesselThicknesses[dist_idx] * (pxToMm or 1.0)
                    mldMm      = vesselThicknesses[mld_idx]  * (pxToMm or 1.0)
                else:
                    edge_avg = max(1, int(len(vesselThicknesses) * 0.05))  # avg over 5% edge
                    if swap_direction:
                        proxDiamMm = numpy.mean(vesselThicknesses[-edge_avg:]) * (pxToMm or 1.0)
                        distDiamMm = numpy.mean(vesselThicknesses[:edge_avg])  * (pxToMm or 1.0)
                        prox_idx = len(vesselThicknesses) - edge_avg - 1
                        dist_idx = edge_avg
                    else:
                        proxDiamMm = numpy.mean(vesselThicknesses[:edge_avg])  * (pxToMm or 1.0)
                        distDiamMm = numpy.mean(vesselThicknesses[-edge_avg:]) * (pxToMm or 1.0)
                        prox_idx = edge_avg
                        dist_idx = len(vesselThicknesses) - edge_avg - 1

                    mld_idx = int(numpy.argmin(vesselThicknesses))
                    mldMm = vesselThicknesses[mld_idx] * (pxToMm or 1.0)

                if ref_method == "Interpolated":
                    dist_diff = cumLen[dist_idx] - cumLen[prox_idx]
                    if dist_diff != 0:
                        ratio = (cumLen[mld_idx] - cumLen[prox_idx]) / dist_diff
                        refDiamMm = proxDiamMm + ratio * (distDiamMm - proxDiamMm)
                    else:
                        refDiamMm = (proxDiamMm + distDiamMm) / 2.0
                else:
                    refDiamMm = (proxDiamMm + distDiamMm) / 2.0
            else:
                # PRE-PCI: auto-locate prox/dist peaks unless corrected
                if _landmarks_corrected:
                    proxDiamMm = vesselThicknesses[prox_idx] * (pxToMm or 1.0)
                    distDiamMm = vesselThicknesses[dist_idx] * (pxToMm or 1.0)
                    mldMm      = vesselThicknesses[mld_idx]  * (pxToMm or 1.0)
                else:
                    prox_raw = numpy.argmax(vesselThicknesses[:refLen])
                    dist_raw = len(vesselThicknesses) - refLen + numpy.argmax(vesselThicknesses[-refLen:])
                    if swap_direction:
                        prox_idx, dist_idx = dist_raw, prox_raw
                    else:
                        prox_idx, dist_idx = prox_raw, dist_raw
                    proxDiamMm = vesselThicknesses[prox_idx] * (pxToMm or 1.0)
                    distDiamMm = vesselThicknesses[dist_idx] * (pxToMm or 1.0)

                    idx_start = min(prox_idx, dist_idx)
                    idx_end   = max(prox_idx, dist_idx)
                    if ostial_lesion:
                        idx_start = 0
                    mld_idx = idx_start + numpy.argmin(vesselThicknesses[idx_start:idx_end+1])
                    mldMm   = vesselThicknesses[mld_idx] * (pxToMm or 1.0)

                if ostial_lesion:
                    refDiamMm = distDiamMm
                elif ref_method == "Interpolated":
                    dist_diff = cumLen[dist_idx] - cumLen[prox_idx]
                    if dist_diff != 0:
                        ratio = (cumLen[mld_idx] - cumLen[prox_idx]) / dist_diff
                        refDiamMm = proxDiamMm + ratio * (distDiamMm - proxDiamMm)
                    else:
                        refDiamMm = (proxDiamMm + distDiamMm) / 2.0
                else:
                    refDiamMm = (proxDiamMm + distDiamMm) / 2.0

            pctDiam = (1.0 - mldMm / refDiamMm) * 100.0 if refDiamMm > 0 else 0.0
            pctArea = (1.0 - (mldMm / refDiamMm) ** 2) * 100.0 if refDiamMm > 0 else 0.0

            totalLenMm = cumLen[-1] * pxToMm if pxToMm else cumLen[-1]

            # stenosis length: segment where diameter < 50 % of reference
            vesselThicknessMm = vesselThicknesses * (pxToMm or 1.0)
            stenosisMask = vesselThicknessMm < (refDiamMm * 0.5)
            if numpy.any(stenosisMask):
                idxs = numpy.where(stenosisMask)[0]
                stenosisLenMm = (cumLen[min(idxs[-1], len(cumLen)-1)] - cumLen[idxs[0]]) * (pxToMm or 1.0)
            else:
                stenosisLenMm = 0.0

            st.markdown("---")
            st.markdown("<h5 style='color:white;'>📐 QCA Metrics</h5>", unsafe_allow_html=True)
            if mmPerPixelCalib:
                st.success(f"✅ **{calibSource} calibration** ({mmPerPixel:.4f} mm/px)")
            else:
                if mmPerPixel is not None:
                    st.warning(f"⚠️ **DICOM metadata** ({mmPerPixel:.4f} mm/px) — switch to 📏 mode in Segmentation for accuracy")
                else:
                    st.warning("⚠️ **No calibration data** — switch to 📏 mode in Segmentation and perform catheter calibration for measurements")

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
                ref_lbl = "Interpolated Ref. Diam." if ref_method == "Interpolated" else "Average Ref. Diam."
                m6.metric(ref_lbl,  f"{refDiamMm:.2f} mm")
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
                {"centreline_point": f"{ref_method.lower()}_reference_diameter_mm",   "thickness_px": round(refDiamMm, 3)},
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
            
            # Default directory resolution (silently resolved in background)
            default_dir = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/")
            if not os.path.exists(os.path.dirname(default_dir)):
                default_dir = os.path.abspath("./reports/")
            st.session_state.report_save_dir = default_dir
            
            has_pid = bool(st.session_state.patient_id and st.session_state.patient_id.strip() != "")
            if not has_pid:
                st.error("🚨 Patient ID missing! Please fill in the field above to save the analysis.")

            # Generate the PDF report in memory (so we can pass it to st.download_button)
            pdf_bytes = b""
            pdf_filename = "report.pdf"
            cart_item = None
            row = None
            
            try:
                try:
                    if "tfc" in meta:
                        tfc = meta["tfc"]
                        timi_calc = meta["timi"]
                        just = meta["just"]
                    else:
                        _, _, _, tfc, timi_calc, just = analyze_series_flow(selectedDicom, os.path.getsize(selectedDicom))
                except:
                    tfc, timi_calc, just = "N/A", "N/A", "N/A"
                    
                final_timi = meta.get("timi_override", str(timi_calc))
                final_timi = str(timi_calc) if final_timi == "Auto" else final_timi
                
                if meta["known_occlude"] == "Yes":
                    final_timi = "0"
                    pdf_dist = "N/A"
                    pdf_ref = "N/A"
                    pdf_mld = "N/A"
                    pdf_pctD = "N/A"
                    pdf_pctA = "N/A"
                    pdf_len = "N/A"
                else:
                    pdf_dist = distDiamMm
                    pdf_ref = refDiamMm
                    pdf_mld = mldMm
                    pdf_pctD = pctDiam
                    pdf_pctA = pctArea
                    pdf_len = totalLenMm
                    
                safe_patient_id = st.session_state.patient_id if st.session_state.patient_id else "NoID"
                pdf_filename = f"{safe_patient_id}_{meta['vessel']}_AHA{meta['aha']}_{meta['phase']}.pdf"
                
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
                        "dist": pdf_dist,
                        "ref": pdf_ref,
                        "mld": pdf_mld,
                        "pct_diam": pdf_pctD,
                        "pct_area": pdf_pctA,
                        "lesion_len": pdf_len,
                        "is_post_pci": is_post_pci,
                        "tfc": tfc,
                        "timi": final_timi,
                        "just": just
                    },
                    "image": cv2.cvtColor(selectedFrameRGBA, cv2.COLOR_RGBA2RGB)
                }
                
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
                    "Max Dist [mm]": "N/A" if pdf_dist == "N/A" else round(pdf_dist, 2),
                    f"{ref_method} Reference [mm]": "N/A" if pdf_ref == "N/A" else round(pdf_ref, 2),
                    "MLD [mm]": "N/A" if pdf_mld == "N/A" else round(pdf_mld, 2),
                    "% Diameter Stenosis": "N/A" if pdf_pctD == "N/A" else round(pdf_pctD, 1),
                    "% Area Stenosis": "N/A" if pdf_pctA == "N/A" else round(pdf_pctA, 1),
                    "Lesion Length [mm]": "N/A" if pdf_len == "N/A" else round(pdf_len, 2),
                    "TIMI Grade": final_timi,
                    "TFC": tfc
                }
                
                import io
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages
                
                pdf_buffer = io.BytesIO()
                with PdfPages(pdf_buffer) as pdf:
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
                    
                    str_dist = "N/A" if pdf_dist == "N/A" else f"{pdf_dist:.2f} mm"
                    str_ref = "N/A" if pdf_ref == "N/A" else f"{pdf_ref:.2f} mm"
                    str_mld = "N/A" if pdf_mld == "N/A" else f"{pdf_mld:.2f} mm"
                    str_pctD = "N/A" if pdf_pctD == "N/A" else f"{pdf_pctD:.1f} %"
                    str_pctA = "N/A" if pdf_pctA == "N/A" else f"{pdf_pctA:.1f} %"
                    str_len = "N/A" if pdf_len == "N/A" else f"{pdf_len:.2f} mm"
                    
                    m_text = (
                        f"Max Proximal Reference:      {proxDiamMm:.2f} mm\n\n"
                        f"Max Distal Reference:        {str_dist}\n\n"
                        f"{ref_method} Reference:      {str_ref}\n\n"
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
                
                pdf_bytes = pdf_buffer.getvalue()
            except Exception as pdf_err:
                st.error(f"Error generating PDF: {pdf_err}")
                
            def on_save_pdf():
                if cart_item is not None:
                    st.session_state.patient_cart.append(cart_item)
                
                # Upload PDF to Firebase Storage and save record to Firestore
                if st.session_state.get("firebase_init") and firebase_available:
                    try:
                        pdf_url = upload_pdf_to_firebase(pdf_bytes, pdf_filename)
                        if pdf_url and row is not None:
                            save_analysis_to_firestore(row, pdf_url)
                    except Exception as fe:
                        print(f"Error saving to Firebase: {fe}")
                
                # Silent VPS save of PDF
                try:
                    save_dir = st.session_state.get("report_save_dir", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/"))
                    os.makedirs(save_dir, exist_ok=True)
                    pdf_path = os.path.join(save_dir, pdf_filename)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_bytes)
                    st.session_state.last_saved_pdf = pdf_path
                    st.session_state.last_saved_pdf_filename = pdf_filename
                except Exception as e:
                    pass
                
                # Silent VPS save of Excel
                try:
                    import pandas as pd
                    save_dir = st.session_state.get("report_save_dir", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/"))
                    target_xlsx = os.path.join(save_dir, "AngioPy.xlsx")
                    if row is not None:
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
                        st.session_state.last_saved_xlsx = target_xlsx
                except Exception as e:
                    pass
                
                st.session_state.current_view = 'grid'
                st.session_state.show_saved_toast = True

            st.download_button(
                label="💾 Save PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
                disabled=not has_pid,
                on_click=on_save_pdf,
                key="save_pdf_main_btn"
            )



