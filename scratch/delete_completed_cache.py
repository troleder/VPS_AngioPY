import os
import sys
import shutil
import firebase_admin
from firebase_admin import credentials, firestore

# Set working directory to project root
os.chdir("/var/www/analiza-dicom")

cred_file = "google_credentials.json"
if not os.path.exists(cred_file):
    print("Error: google_credentials.json not found")
    sys.exit(1)

# Initialize Firebase
cred = credentials.Certificate(cred_file)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

print("Fetching completed patient IDs from Firestore...")
completed_docs = db.collection("analysis_results").where("phase", "==", "COMPLETED").stream()
completed_pids = set()
for doc in completed_docs:
    data = doc.to_dict()
    pid = data.get("patient_id")
    if pid:
        completed_pids.add(pid)

print(f"Found {len(completed_pids)} completed cases in database.")

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception:
        pass
    return total

cache_dir = "./tailscale_cache"
if not os.path.exists(cache_dir):
    print("Error: tailscale_cache directory not found")
    sys.exit(1)

deleted_count = 0
deleted_bytes = 0

print("\nScanning cache directory for completed cases to delete...")
for site_fn in os.listdir(cache_dir):
    site_path = os.path.join(cache_dir, site_fn)
    if not os.path.isdir(site_path) or site_fn.startswith('.'):
        continue
        
    for pat_fn in os.listdir(site_path):
        pat_path = os.path.join(site_path, pat_fn)
        if not os.path.isdir(pat_path) or pat_fn.startswith('.'):
            continue
            
        # Check if patient ID is completed in database
        if pat_fn in completed_pids:
            size = get_dir_size(pat_path)
            print(f"Deleting completed case: {site_fn}/{pat_fn} ({size / (1024*1024):.2f} MB)...")
            try:
                shutil.rmtree(pat_path)
                deleted_count += 1
                deleted_bytes += size
            except Exception as e:
                print(f"Error deleting {pat_path}: {e}")

print(f"\nSuccessfully deleted {deleted_count} completed patient cache folders.")
print(f"Freed {deleted_bytes / (1024*1024*1024):.2f} GB of disk space.")

# Clean up empty site folders
for site_fn in os.listdir(cache_dir):
    site_path = os.path.join(cache_dir, site_fn)
    if os.path.isdir(site_path) and not site_fn.startswith('.'):
        try:
            if not os.listdir(site_path):
                print(f"Removing empty site folder: {site_fn}")
                os.rmdir(site_path)
        except Exception:
            pass
