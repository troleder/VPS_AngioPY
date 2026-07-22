import os
import sys
import json
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

cachable_completed = []
cachable_active = []

print("\nScanning cache directory for cached cases...")
for site_fn in os.listdir(cache_dir):
    site_path = os.path.join(cache_dir, site_fn)
    if not os.path.isdir(site_path) or site_fn.startswith('.'):
        continue
        
    for pat_fn in os.listdir(site_path):
        pat_path = os.path.join(site_path, pat_fn)
        if not os.path.isdir(pat_path) or pat_fn.startswith('.'):
            continue
            
        # Check if patient ID is completed
        size_bytes = get_dir_size(pat_path)
        size_mb = size_bytes / (1024 * 1024)
        
        info = {
            "site": site_fn,
            "patient_id": pat_fn,
            "size_mb": size_mb,
            "path": pat_path
        }
        
        if pat_fn in completed_pids:
            cachable_completed.append(info)
        else:
            cachable_active.append(info)

print(f"\n--- SCRAPED CACHE STATISTICS ---")
print(f"Total Cached Cases: {len(cachable_completed) + len(cachable_active)}")
print(f"  - Completed (Safely Deletable): {len(cachable_completed)}")
print(f"  - Active/In-Progress (Keep): {len(cachable_active)}")

total_deletable_bytes = sum(c["size_mb"] for c in cachable_completed)
total_active_bytes = sum(c["size_mb"] for c in cachable_active)

print(f"\nPotential disk space to free: {total_deletable_bytes / 1024:.2f} GB")
print(f"Remaining active cache size: {total_active_bytes / 1024:.2f} GB")

print("\n--- COMPLETED CASES DETAIL (Safely Deletable) ---")
print(f"{'Site':<10}{'Patient ID':<20}{'Size (MB)':<15}")
print("-" * 45)
# Sort by size descending
cachable_completed.sort(key=lambda x: x["size_mb"], reverse=True)
for c in cachable_completed:
    print(f"{c['site']:<10}{c['patient_id']:<20}{c['size_mb']:<15.2f}")

print("\n--- ACTIVE/IN-PROGRESS CASES DETAIL (Do NOT Delete) ---")
print(f"{'Site':<10}{'Patient ID':<20}{'Size (MB)':<15}")
print("-" * 45)
cachable_active.sort(key=lambda x: x["size_mb"], reverse=True)
for c in cachable_active:
    print(f"{c['site']:<10}{c['patient_id']:<20}{c['size_mb']:<15.2f}")

# Save JSON breakdown to a file
try:
    with open("/var/www/analiza-dicom/scratch/cache_report.json", "w") as f:
        json.dump({
            "completed": cachable_completed,
            "active": cachable_active
        }, f, indent=2)
    print("\nSaved detailed report to /var/www/analiza-dicom/scratch/cache_report.json")
except Exception as e:
    print(f"\nWarning: Could not save JSON report to disk (disk full): {e}")

