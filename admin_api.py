import os
import re
import time
import shutil
import threading
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore, storage, auth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("admin_api")

app = FastAPI(title="AngioPy Admin API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin
cred_file = "google_credentials.json"
if os.path.exists(cred_file):
    try:
        with open(cred_file, "r") as f:
            cred_data = json.load(f)
        project_id = cred_data.get("project_id", "")
        bucket_name = f"{project_id}.appspot.com"
        
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
        except Exception as e:
            logger.warning(f"Failed to fetch bucket names from storage: {e}")
            
        cred = credentials.Certificate(cred_file)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
        db = firestore.client()
        bucket = storage.bucket()
        logger.info("Firebase initialized successfully in Admin API")
    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        raise e
else:
    logger.error("google_credentials.json not found")
    raise FileNotFoundError("google_credentials.json not found")

# --- Authentication Dependency ---
def get_admin_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    id_token = authorization.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token.get("email", "").lower()
        is_admin = email == "tomaszroleder@gmail.com" or email.startswith("tomaszroleder@")
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User does not have admin permissions",
            )
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {e}",
        )

# --- Active Caching Tasks Tracker ---
class DiskSyncedActiveTasks:
    def __init__(self, directory=None):
        if directory is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            directory = os.path.join(script_dir, "local_cache", "tasks")
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        self.memory_tasks = {}
        
    def _safe_filename(self, key):
        safe = re.sub(r'[^a-zA-Z0-9_-]', '_', key)
        return safe

    def _save_to_disk(self, key, value):
        try:
            safe_key = self._safe_filename(key)
            temp_path = os.path.join(self.directory, f"{safe_key}.tmp")
            final_path = os.path.join(self.directory, f"{safe_key}.json")
            with open(temp_path, "w") as f:
                json.dump(value, f)
            os.replace(temp_path, final_path)
        except Exception as e:
            logger.error(f"[DISK TASKS] Error saving task {key} to disk: {e}")
            
    def _read_from_disk(self, key):
        safe_key = self._safe_filename(key)
        path = os.path.join(self.directory, f"{safe_key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[DISK TASKS] Error reading task {key} from disk: {e}")
        return None
        
    def get_all_tasks(self):
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
                        recent = (time.time() - mtime) < 3600  # 1 hour limit
                        if is_running:
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
            logger.error(f"[DISK TASKS] Error listing tasks from disk: {e}")
            
        for k, v in self.memory_tasks.items():
            safe_k = self._safe_filename(k)
            if safe_k not in tasks:
                tasks[safe_k] = v
        return tasks

    def set_task(self, key, value):
        self.memory_tasks[key] = value
        self._save_to_disk(key, value)
        
    def pop_task(self, key):
        safe_key = self._safe_filename(key)
        self.memory_tasks.pop(key, None)
        path = os.path.join(self.directory, f"{safe_key}.json")
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.error(f"[DISK TASKS] Error removing task {key} from disk: {e}")

_active_copy_tasks = DiskSyncedActiveTasks()
_copy_tasks_lock = threading.Lock()

# --- Helper Functions (Reused from angioPySegmentation) ---

def get_clean_patient_id(dir_name):
    match = re.match(r"^(\d{4}-\d{4})", dir_name)
    if match:
        return match.group(1)
    return dir_name

def get_raw_folders_for_clean_path(nav_path, base_dir="/mnt/dane_dicom/"):
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
    dst_abs = os.path.abspath(os.path.join("./tailscale_cache", loc))
    sentinel_path = os.path.join(dst_abs, ".cache_complete")
    if os.path.exists(sentinel_path):
        return True
    if os.path.exists(dst_abs) and os.path.isdir(dst_abs):
        if not os.path.exists(base_dir):
            for root, dirs, files in os.walk(dst_abs):
                if any(not f.startswith(".") for f in files):
                    return True
            return False
        r_locs = get_raw_folders_for_clean_path(loc, base_dir)
        source_files = []
        for r_loc in r_locs:
            src_abs = os.path.abspath(os.path.join(base_dir, r_loc))
            if os.path.exists(src_abs) and os.path.isdir(src_abs):
                has_angio = False
                try:
                    for entry in os.scandir(src_abs):
                        if entry.is_dir() and entry.name.upper() == "ANGIO":
                            has_angio = True
                            break
                except:
                    pass

                for root, dirs, files in os.walk(src_abs):
                    for file in files:
                        if not file.startswith("."):
                            src_fp = os.path.join(root, file)
                            if has_angio:
                                rel_to_patient = os.path.relpath(src_fp, src_abs)
                                rel_parts_upper = [p.upper() for p in rel_to_patient.replace("\\", "/").split("/")]
                                if "ANGIO" not in rel_parts_upper:
                                    continue
                            source_files.append(src_fp)
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
            except:
                pass
    return False

def is_patient_cached(site, patient_id):
    return check_cache_status_and_heal(f"{site}/{patient_id}")

def robust_copy(src, dst):
    try:
        shutil.copy2(src, dst)
    except OSError as e:
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
            except:
                pass
        except:
            raise e

# --- In-Memory Cache for Scanning Disk Patients (10-minute TTL) ---
_patients_cache = {"data": None, "timestamp": 0.0}

def get_scanned_patients_cached():
    now = time.time()
    if _patients_cache["data"] is not None and (now - _patients_cache["timestamp"]) < 600:
        return _patients_cache["data"]
        
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
                        if re.match(r"^\d{4}-\d{4}$", clean_pid):
                            key = (site, clean_pid)
                            if key not in seen:
                                seen.add(key)
                                patients.append({
                                    "site": site,
                                    "patient_id": clean_pid
                                })
    except Exception as e:
        logger.error(f"Error scanning tailscale patients: {e}")
        
    _patients_cache["data"] = patients
    _patients_cache["timestamp"] = time.time()
    return patients

def estimate_patient_download_size(site, patient_id, base_dir="/mnt/dane_dicom/"):
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

# Cache size estimation (24-hour TTL)
_sizes_cache = {}

def get_estimated_size_cached(site, patient_id):
    key = f"{site}/{patient_id}"
    now = time.time()
    if key in _sizes_cache:
        size, ts = _sizes_cache[key]
        if (now - ts) < 86400:
            return size
    size = estimate_patient_download_size(site, patient_id)
    _sizes_cache[key] = (size, now)
    return size

# --- Prefetch Caching Background Worker ---
def _prefetch_cases_worker(cases: List[dict], task_id: str):
    base_dir = "/mnt/dane_dicom/"
    try:
        with _copy_tasks_lock:
            _active_copy_tasks.set_task(task_id, {
                "status": "running",
                "copied_files": 0,
                "total_files": len(cases),
                "copied_bytes": 0,
                "total_bytes": 0,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": "Scanning files on Tailscale..."
            })
            
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
                has_angio = False
                try:
                    for entry in os.scandir(s_path):
                        if entry.is_dir() and entry.name.upper() == "ANGIO":
                            has_angio = True
                            break
                except:
                    pass

                for root, dirs, files in os.walk(s_path):
                    for file in files:
                        if file.startswith('.'):
                            continue
                        src_fp = os.path.join(root, file)
                        if has_angio:
                            rel_to_patient = os.path.relpath(src_fp, s_path)
                            rel_parts_upper = [p.upper() for p in rel_to_patient.replace("\\", "/").split("/")]
                            if "ANGIO" not in rel_parts_upper:
                                continue

                        try:
                            sz = os.path.getsize(src_fp)
                        except:
                            sz = 0
                        dst_fp = get_cache_file_path(src_fp, base_dir)
                        files_to_copy.append((src_fp, dst_fp, sz))
                        total_size_bytes += sz
                        
        total_files = len(files_to_copy)
        if total_files == 0:
            with _copy_tasks_lock:
                _active_copy_tasks.set_task(task_id, {
                    "status": "success",
                    "total_files": 0,
                    "total_bytes": 0
                })
            return
            
        with _copy_tasks_lock:
            _active_copy_tasks.set_task(task_id, {
                "status": "running",
                "copied_files": 0,
                "total_files": total_files,
                "copied_bytes": 0,
                "total_bytes": total_size_bytes,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": ""
            })
            
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
                _active_copy_tasks.set_task(task_id, {
                    "status": "running",
                    "copied_files": idx + 1,
                    "total_files": total_files,
                    "copied_bytes": copied_bytes,
                    "total_bytes": total_size_bytes,
                    "speed": speed_mb,
                    "est_left": est_left,
                    "start_time": start_time
                })
                
        for _, dst_abs in case_folders:
            try:
                now = time.time()
                os.utime(dst_abs, (now, now))
                sentinel_path = os.path.join(dst_abs, ".cache_complete")
                with open(sentinel_path, "w") as f:
                    f.write("completed")
            except:
                pass
                
        with _copy_tasks_lock:
            _active_copy_tasks.set_task(task_id, {
                "status": "success",
                "copied_files": total_files,
                "total_files": total_files,
                "copied_bytes": total_size_bytes,
                "total_bytes": total_size_bytes,
                "speed": 0.0,
                "est_left": 0.0
            })
    except Exception as e:
        logger.error(f"Error in background copy worker: {e}")
        with _copy_tasks_lock:
            _active_copy_tasks.set_task(task_id, {
                "status": "error",
                "error_msg": str(e)
            })

# --- PDF QCA Parser ---
def parse_qca_pdf(pdf_bytes: bytes):
    try:
        from pypdf import PdfReader
        import io
        stream = io.BytesIO(pdf_bytes)
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
            try: return float(val)
            except: return val
                
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
        logger.error(f"Error parsing PDF: {e}")
        return None

def upload_pdf_to_firebase(pdf_bytes: bytes, pdf_filename: str) -> Optional[str]:
    try:
        blob_path = f"reports/{pdf_filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.generate_signed_url(expiration=3600*24*365)
    except Exception as e:
        logger.error(f"Error uploading PDF to Firebase Storage: {e}")
        return None

# --- API Endpoints ---

@app.post("/api/auth/verify-token")
def verify_token(user: dict = Depends(get_admin_user)):
    return {"status": "success", "user": {"email": user.get("email"), "name": user.get("name")}}

@app.get("/api/metrics")
def get_metrics(user: dict = Depends(get_admin_user)):
    try:
        completed_reports = [doc.to_dict() for doc in db.collection("analysis_results").stream()]
        assignments = [doc.to_dict() for doc in db.collection("assignments").stream()]
        site_assignments = [doc.to_dict() for doc in db.collection("site_assignments").stream()]
        
        completed_pids = {r.get("patient_id") for r in completed_reports if r.get("patient_id") and r.get("phase") == "COMPLETED"}
        
        all_patients = get_scanned_patients_cached()
        
        global_assigned_pids = set()
        for asg in assignments:
            pid = asg.get("patient_id")
            if pid and asg.get("status") != "unassigned":
                global_assigned_pids.add(pid)
        for sa in site_assignments:
            site = sa.get("site")
            if site:
                for p in all_patients:
                    if p["site"] == site:
                        global_assigned_pids.add(p["patient_id"])
                        
        total_completed = len([r for r in completed_reports if r.get("phase") == "COMPLETED"])
        unique_completed = len(completed_pids)
        total_assigned = len(global_assigned_pids)
        completed_assigned_count = sum(1 for pid in global_assigned_pids if pid in completed_pids)
        
        total_on_disk = len(all_patients)
        completed_on_disk = sum(1 for p in all_patients if p["patient_id"] in completed_pids)
        remaining_on_disk = max(0, total_on_disk - completed_on_disk)
        
        return {
            "total_completed_reports": total_completed,
            "unique_completed_patients": unique_completed,
            "total_assigned_patients": total_assigned,
            "completed_assigned_count": completed_assigned_count,
            "total_on_disk": total_on_disk,
            "completed_on_disk": completed_on_disk,
            "remaining_on_disk": remaining_on_disk
        }
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysts")
def get_analysts(user: dict = Depends(get_admin_user)):
    analysts = []
    try:
        page = auth.list_users()
        while page:
            for u in page.users:
                if u.email:
                    analysts.append({
                        "uid": u.uid,
                        "email": u.email,
                        "name": u.display_name or u.email.split("@")[0],
                        "username": u.email.split("@")[0]
                    })
            page = page.get_next_page()
        return analysts
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AnalystCreate(BaseModel):
    email: str
    password: str
    name: str

@app.post("/api/analysts")
def create_analyst(data: AnalystCreate, user: dict = Depends(get_admin_user)):
    try:
        user_record = auth.create_user(
            email=data.email,
            password=data.password,
            display_name=data.name
        )
        return {"status": "success", "uid": user_record.uid}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/sites")
def get_sites(user: dict = Depends(get_admin_user)):
    try:
        site_assignments = {doc.id: doc.to_dict() for doc in db.collection("site_assignments").stream()}
        assignments = [doc.to_dict() for doc in db.collection("assignments").stream()]
        
        completed_reports = [doc.to_dict() for doc in db.collection("analysis_results").stream()]
        completed_pids = {r.get("patient_id") for r in completed_reports if r.get("patient_id") and r.get("phase") == "COMPLETED"}
        
        all_patients = get_scanned_patients_cached()
        all_sites = sorted(list({p["site"] for p in all_patients}))
        
        sites_list = []
        for site in all_sites:
            site_pats = [p for p in all_patients if p["site"] == site]
            total_pats = len(site_pats)
            completed_pats = sum(1 for p in site_pats if p["patient_id"] in completed_pids)
            
            assignment_status = "unassigned"
            assigned_to = None
            assigned_by = None
            assigned_at = None
            
            if site in site_assignments:
                assignment_status = "assigned_full"
                assigned_to = site_assignments[site].get("assigned_to")
                assigned_by = site_assignments[site].get("assigned_by")
                assigned_at = site_assignments[site].get("assigned_at")
            else:
                site_inds = [asg for asg in assignments if asg.get("site") == site and asg.get("status") != "unassigned"]
                if site_inds:
                    assignment_status = "assigned_partial"
                    assigned_to = list({asg.get("assigned_to") for asg in site_inds})
            
            sites_list.append({
                "site": site,
                "status": assignment_status,
                "assigned_to": assigned_to,
                "assigned_by": assigned_by,
                "assigned_at": str(assigned_at) if assigned_at else None,
                "total_patients": total_pats,
                "completed_patients": completed_pats
            })
            
        return sites_list
    except Exception as e:
        logger.error(f"Error fetching sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients/unassigned-options")
def get_unassigned_patient_options(user: dict = Depends(get_admin_user)):
    try:
        assignments_dict = {doc.id: doc.to_dict() for doc in db.collection("assignments").stream()}
        site_assignments_dict = {doc.id: doc.to_dict() for doc in db.collection("site_assignments").stream()}
        
        completed_reports = [doc.to_dict() for doc in db.collection("analysis_results").stream()]
        completed_pids = {r.get("patient_id") for r in completed_reports if r.get("patient_id") and r.get("phase") == "COMPLETED"}
        
        all_patients = get_scanned_patients_cached()
        available = []
        
        for p in all_patients:
            pid = p["patient_id"]
            site = p["site"]
            
            is_individually_assigned = pid in assignments_dict and assignments_dict[pid].get("status") != "unassigned"
            is_site_assigned = site in site_assignments_dict
            
            if not is_individually_assigned and not is_site_assigned:
                is_comp = pid in completed_pids
                available.append({
                    "patient_id": pid,
                    "site": site,
                    "completed": is_comp
                })
        return available
    except Exception as e:
        logger.error(f"Error loading patient options: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AssignRequest(BaseModel):
    type: str # "patient" or "site"
    target: str # patient_id or site
    assigned_to: str # username
    assigned_by: str

@app.post("/api/assign")
def assign_target(data: AssignRequest, user: dict = Depends(get_admin_user)):
    try:
        if data.type == "patient":
            asg_doc = db.collection("assignments").document(data.target).get()
            site_asg_doc = db.collection("site_assignments").document(data.target.split("-")[0]).get()
            
            if asg_doc.exists and asg_doc.to_dict().get("status") != "unassigned":
                raise HTTPException(status_code=400, detail=f"Patient {data.target} is already assigned!")
            if site_asg_doc.exists:
                raise HTTPException(status_code=400, detail=f"Site is assigned to {site_asg_doc.to_dict().get('assigned_to')}")
                
            db.collection("assignments").document(data.target).set({
                "patient_id": data.target,
                "site": data.target.split("-")[0],
                "assigned_to": data.assigned_to,
                "assigned_by": data.assigned_by,
                "assigned_at": firestore.SERVER_TIMESTAMP,
                "status": "assigned"
            })
        else:
            site_asg_doc = db.collection("site_assignments").document(data.target).get()
            if site_asg_doc.exists:
                raise HTTPException(status_code=400, detail=f"Site {data.target} is already assigned!")
                
            conflicting_cases = [doc for doc in db.collection("assignments").where("site", "==", data.target).stream()]
            for doc in conflicting_cases:
                doc_data = doc.to_dict()
                if doc_data.get("status") != "unassigned" and doc_data.get("assigned_to") != data.assigned_to:
                    raise HTTPException(status_code=400, detail=f"Patient {doc.id} from this site is assigned to another analyst ({doc_data.get('assigned_to')})! Unassign them first.")
            
            db.collection("site_assignments").document(data.target).set({
                "site": data.target,
                "assigned_to": data.assigned_to,
                "assigned_by": data.assigned_by,
                "assigned_at": firestore.SERVER_TIMESTAMP
            })
            
            for doc in conflicting_cases:
                doc_data = doc.to_dict()
                if doc_data.get("assigned_to") == data.assigned_to:
                    db.collection("assignments").document(doc.id).delete()
                    
        return {"status": "success"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Assignment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class UnassignRequest(BaseModel):
    type: str # "patient" or "site"
    target: str # patient_id or site

@app.post("/api/unassign")
def unassign_target(data: UnassignRequest, user: dict = Depends(get_admin_user)):
    try:
        if data.type == "patient":
            db.collection("assignments").document(data.target).delete()
        else:
            db.collection("site_assignments").document(data.target).delete()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Unassignment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/unassigned")
def get_unassigned_cases(user: dict = Depends(get_admin_user)):
    try:
        assignments = [doc.to_dict() for doc in db.collection("assignments").stream()]
        unassigned_list = []
        for asg in assignments:
            if asg.get("status") == "unassigned" or asg.get("assigned_to") == "unassigned":
                uat = asg.get("unassigned_at")
                unassigned_list.append({
                    "patient_id": asg.get("patient_id"),
                    "site": asg.get("site"),
                    "unassigned_by": asg.get("unassigned_by"),
                    "unassigned_reason": asg.get("unassigned_reason", "No reason provided"),
                    "unassigned_at": str(uat) if uat else None
                })
        return unassigned_list
    except Exception as e:
        logger.error(f"Error loading unassigned list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ReassignRequest(BaseModel):
    patient_id: str
    site: str
    assigned_to: str
    assigned_by: str

@app.post("/api/reassign")
def reassign_case(data: ReassignRequest, user: dict = Depends(get_admin_user)):
    try:
        db.collection("assignments").document(data.patient_id).set({
            "patient_id": data.patient_id,
            "site": data.site,
            "assigned_to": data.assigned_to,
            "assigned_by": data.assigned_by,
            "assigned_at": firestore.SERVER_TIMESTAMP,
            "status": "assigned"
        })
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Reassignment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysts/progress")
def get_analysts_progress(user: dict = Depends(get_admin_user)):
    try:
        analysts = get_analysts(user)
        completed_reports = [doc.to_dict() for doc in db.collection("analysis_results").stream()]
        completed_pids = {r.get("patient_id") for r in completed_reports if r.get("patient_id") and r.get("phase") == "COMPLETED"}
        
        assignments = [doc.to_dict() for doc in db.collection("assignments").stream()]
        site_assignments = [doc.to_dict() for doc in db.collection("site_assignments").stream()]
        
        all_patients = get_scanned_patients_cached()
        
        results = []
        for a in analysts:
            uname = a["username"]
            assigned_set = set()
            for asg in assignments:
                pid = asg.get("patient_id")
                if pid and asg.get("assigned_to") == uname and asg.get("status") != "unassigned":
                    assigned_set.add(pid)
            for sa in site_assignments:
                site = sa.get("site")
                if site and sa.get("assigned_to") == uname:
                    for p in all_patients:
                        if p["site"] == site:
                            assigned_set.add(p["patient_id"])
                            
            total_assigned = len(assigned_set)
            completed_assigned = len({pid for pid in assigned_set if pid in completed_pids})
            remaining_assigned = max(0, total_assigned - completed_assigned)
            
            my_reports = [r for r in completed_reports if r.get("analyst") == uname]
            total_reports = len(my_reports)
            unique_patients = len({r.get("patient_id") for r in my_reports if r.get("patient_id")})
            
            tasks = []
            for sa in site_assignments:
                if sa.get("assigned_to") == uname:
                    site = sa.get("site")
                    site_pats = [p["patient_id"] for p in all_patients if p["site"] == site]
                    site_comp = [pid for pid in site_pats if pid in completed_pids]
                    tasks.append({
                        "type": "site",
                        "site": site,
                        "total_patients": len(site_pats),
                        "completed_patients": len(site_comp)
                    })
            for asg in assignments:
                if asg.get("assigned_to") == uname and asg.get("status") != "unassigned":
                    pid = asg.get("patient_id")
                    tasks.append({
                        "type": "patient",
                        "patient_id": pid,
                        "site": asg.get("site"),
                        "completed": pid in completed_pids,
                        "report_url": next((r.get("pdf_url") for r in my_reports if r.get("patient_id") == pid and r.get("phase") == "COMPLETED"), None)
                    })
                    
            results.append({
                "username": uname,
                "name": a["name"],
                "email": a["email"],
                "total_assigned": total_assigned,
                "completed_assigned": completed_assigned,
                "remaining_assigned": remaining_assigned,
                "total_reports": total_reports,
                "unique_patients": unique_patients,
                "tasks": tasks
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching analyst progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/estimate")
def estimate_cache_size(data: dict, user: dict = Depends(get_admin_user)):
    ctype = data.get("type")
    target = data.get("target")
    if not ctype or not target:
        raise HTTPException(status_code=400, detail="Missing type or target")
        
    try:
        completed_pids = {doc.to_dict().get("patient_id") for doc in db.collection("analysis_results").where("phase", "==", "COMPLETED").stream()}
        all_patients = get_scanned_patients_cached()
        
        target_patients = []
        cached_patients = []
        completed_skipped = []
        total_pats_count = 0
        
        if ctype == "site":
            site_pats = [p for p in all_patients if p["site"] == target]
            total_pats_count = len(site_pats)
            for p in site_pats:
                pid = p["patient_id"]
                is_cached = is_patient_cached(target, pid)
                is_completed = pid in completed_pids
                
                if is_cached:
                    cached_patients.append(pid)
                elif is_completed:
                    completed_skipped.append(pid)
                else:
                    target_patients.append(p)
        else:
            site = next((p["site"] for p in all_patients if p["patient_id"] == target), None)
            if site:
                total_pats_count = 1
                is_cached = is_patient_cached(site, target)
                is_completed = target in completed_pids
                
                if is_cached:
                    cached_patients.append(target)
                elif is_completed:
                    completed_skipped.append(target)
                else:
                    target_patients.append({"site": site, "patient_id": target})
                    
        total_bytes = 0
        for p in target_patients:
            total_bytes += get_estimated_size_cached(p["site"], p["patient_id"])
            
        size_gb = total_bytes / (1024**3)
        return {
            "size_bytes": total_bytes,
            "size_gb": size_gb,
            "total_patients": total_pats_count,
            "cached_patients": cached_patients,
            "completed_patients": completed_skipped,
            "scheduled_patients": [p["patient_id"] for p in target_patients]
        }
    except Exception as e:
        logger.error(f"Error estimating size: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/prefetch")
def trigger_prefetch(data: dict, user: dict = Depends(get_admin_user)):
    ctype = data.get("type")
    target = data.get("target")
    if not ctype or not target:
        raise HTTPException(status_code=400, detail="Missing type or target")
        
    try:
        completed_pids = {doc.to_dict().get("patient_id") for doc in db.collection("analysis_results").where("phase", "==", "COMPLETED").stream()}
        all_patients = get_scanned_patients_cached()
        
        target_patients = []
        task_id = ""
        if ctype == "site":
            task_id = f"prefetch_admin_site_{target}_{int(time.time())}"
            site_pats = [p for p in all_patients if p["site"] == target]
            for p in site_pats:
                pid = p["patient_id"]
                if pid not in completed_pids and not is_patient_cached(target, pid):
                    target_patients.append(p)
        else:
            task_id = f"prefetch_admin_patient_{target}_{int(time.time())}"
            site = next((p["site"] for p in all_patients if p["patient_id"] == target), None)
            if site and not is_patient_cached(site, target) and target not in completed_pids:
                target_patients.append({"site": site, "patient_id": target})
                
        if not target_patients:
            return {"status": "ignored", "message": "All matching patients are already cached or completed"}
            
        total_bytes = 0
        for p in target_patients:
            total_bytes += get_estimated_size_cached(p["site"], p["patient_id"])
            
        with _copy_tasks_lock:
            _active_copy_tasks.set_task(task_id, {
                "status": "running",
                "copied_files": 0,
                "total_files": len(target_patients),
                "copied_bytes": 0,
                "total_bytes": total_bytes,
                "speed": 0.0,
                "est_left": 0.0,
                "start_time": time.time(),
                "detail": "Scanning files on Tailscale..."
            })
            
        thread = threading.Thread(
            target=_prefetch_cases_worker,
            args=(target_patients, task_id),
            daemon=True
        )
        thread.start()
        
        return {"status": "success", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error starting prefetch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cache/tasks")
def get_cache_tasks(user: dict = Depends(get_admin_user)):
    return _active_copy_tasks.get_all_tasks()

@app.post("/api/cache/tasks/{tid}/dismiss")
def dismiss_cache_task(tid: str, user: dict = Depends(get_admin_user)):
    try:
        _active_copy_tasks.pop_task(tid)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error dismissing task {tid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/import")
async def bulk_import_reports(
    analyst: str = Form(...),
    files: List[UploadFile] = File(...),
    user: dict = Depends(get_admin_user)
):
    success_count = 0
    error_count = 0
    results = []
    
    for upload_file in files:
        try:
            pdf_bytes = await upload_file.read()
            parsed = parse_qca_pdf(pdf_bytes)
            if parsed is None:
                results.append({"filename": upload_file.filename, "status": "error", "error": "Not a valid AngioPy QCA report"})
                error_count += 1
                continue
                
            pdf_url = upload_pdf_to_firebase(pdf_bytes, upload_file.filename)
            if not pdf_url:
                results.append({"filename": upload_file.filename, "status": "error", "error": "Failed to upload to Storage"})
                error_count += 1
                continue
                
            ref_val = parsed["metrics"]["ref_diam_mm"]
            
            doc_data = {
                "patient_id": parsed["patient_id"],
                "dicom_name": f"imported_{upload_file.filename[:-4]}",
                "phase": parsed["phase"],
                "vessel": parsed["vessel"],
                "aha": parsed["aha"],
                "ffr_registered": parsed["ffr_registered"],
                "other_lesion_distal": parsed["other_lesion_distal"],
                "known_occlude": parsed["known_occlude"],
                "pdf_url": pdf_url,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "analyst": analyst,
                "metrics": {
                    "prox_diam_mm": parsed["metrics"]["prox_diam_mm"],
                    "dist_diam_mm": parsed["metrics"]["dist_diam_mm"],
                    "ref_diam_mm": ref_val,
                    "mld_mm": parsed["metrics"]["mld_mm"],
                    "pct_diameter_stenosis": parsed["metricsosis"] if "pct_diameter_stenosis" in parsed["metrics"] else parsed["metrics"].get("pct_diameter_stenosis", "N/A"),
                    "pct_area_stenosis": parsed["metrics"]["pct_area_stenosis"],
                    "lesion_length_mm": parsed["metrics"]["lesion_length_mm"],
                    "timi_grade": parsed["metrics"]["timi_grade"],
                    "tfc": parsed["metrics"]["tfc"]
                }
            }
            # Clean pct_diameter_stenosis
            doc_data["metrics"]["pct_diameter_stenosis"] = parsed["metrics"]["pct_diameter_stenosis"]
            
            db.collection("analysis_results").add(doc_data)
            results.append({"filename": upload_file.filename, "status": "success"})
            success_count += 1
            
        except Exception as ex:
            results.append({"filename": upload_file.filename, "status": "error", "error": str(ex)})
            error_count += 1
            
    return {
        "status": "success",
        "success_count": success_count,
        "error_count": error_count,
        "results": results
    }

@app.get("/api/cache/files")
def list_cached_files(user: dict = Depends(get_admin_user)):
    cache_dir = "./tailscale_cache"
    if not os.path.exists(cache_dir):
        return {"total_size_bytes": 0, "total_size_mb": 0, "total_patients": 0, "patients": []}
        
    patients = []
    total_size_bytes = 0
    
    try:
        for site in os.listdir(cache_dir):
            site_path = os.path.join(cache_dir, site)
            if not os.path.isdir(site_path) or site.startswith("."):
                continue
                
            for patient in os.listdir(site_path):
                patient_path = os.path.join(site_path, patient)
                if not os.path.isdir(patient_path) or patient.startswith("."):
                    continue
                    
                patient_size = 0
                file_count = 0
                is_complete = os.path.exists(os.path.join(patient_path, ".cache_complete"))
                
                for root, dirs, files in os.walk(patient_path):
                    for file in files:
                        if file.startswith("."):
                            continue
                        file_path = os.path.join(root, file)
                        try:
                            f_size = os.path.getsize(file_path)
                            patient_size += f_size
                            file_count += 1
                        except:
                            pass
                            
                try:
                    mtime = os.path.getmtime(patient_path)
                except:
                    mtime = time.time()
                    
                total_size_bytes += patient_size
                patients.append({
                    "patient_id": patient,
                    "site": site,
                    "file_count": file_count,
                    "size_bytes": patient_size,
                    "size_mb": round(patient_size / (1024 * 1024), 2),
                    "is_complete": is_complete,
                    "cached_at": mtime
                })
                
        patients.sort(key=lambda x: x["cached_at"], reverse=True)
        return {
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "total_patients": len(patients),
            "patients": patients
        }
    except Exception as e:
        logger.error(f"Error scanning cached files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cache/files/{site}/{patient_id}")
def delete_patient_cache(site: str, patient_id: str, user: dict = Depends(get_admin_user)):
    if not re.match(r"^[a-zA-Z0-9_-]+$", site) or not re.match(r"^\d{4}-\d{4}$", patient_id):
        raise HTTPException(status_code=400, detail="Invalid site or patient ID format")
        
    patient_dir = os.path.abspath(os.path.join("./tailscale_cache", site, patient_id))
    cache_dir_abs = os.path.abspath("./tailscale_cache")
    if not patient_dir.startswith(cache_dir_abs):
        raise HTTPException(status_code=400, detail="Access denied")
        
    if not os.path.exists(patient_dir):
        raise HTTPException(status_code=404, detail="Patient cache directory not found")
        
    try:
        shutil.rmtree(patient_dir)
        site_dir = os.path.dirname(patient_dir)
        if os.path.exists(site_dir) and not os.listdir(site_dir):
            os.rmdir(site_dir)
            
        global _sizes_cache
        _sizes_cache = {}
        
        return {"status": "success", "message": f"Successfully deleted cache for patient {patient_id}"}
    except Exception as e:
        logger.error(f"Error deleting cache for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refresh")
def force_refresh_caches(user: dict = Depends(get_admin_user)):
    global _patients_cache, _sizes_cache
    _patients_cache = {"data": None, "timestamp": 0.0}
    _sizes_cache = {}
    return {"status": "success", "message": "All backend caches invalidated"}
