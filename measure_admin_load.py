import time
import os
import sys

# Add virtualenv site-packages to path if needed (or we will run with the venv python)
print("Starting Admin Panel profiling...")

t0 = time.time()
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("google_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
print(f"Firebase initialization took: {time.time() - t0:.3f} seconds")

# 1. Profile listing auth users
t0 = time.time()
from firebase_admin import auth
analysts = []
page = auth.list_users()
while page:
    for user in page.users:
        if user.email:
            analysts.append(user.email)
    page = page.get_next_page()
print(f"Firebase Auth listing users ({len(analysts)} users) took: {time.time() - t0:.3f} seconds")

# 2. Profile scanning tailscale patients
t0 = time.time()
base_dir = "/mnt/dane_dicom/"
patients = []
if os.path.exists(base_dir):
    try:
        import re
        seen = set()
        for site in os.listdir(base_dir):
            site_path = os.path.join(base_dir, site)
            if os.path.isdir(site_path) and not site.startswith("."):
                for patient in os.listdir(site_path):
                    patient_path = os.path.join(site_path, patient)
                    if os.path.isdir(patient_path) and not patient.startswith("."):
                        clean_pid = patient.strip()
                        if "-" in clean_pid:
                            # Try to match the regex used in the app
                            if re.match(r"^\d{4}-\d{4}$", clean_pid):
                                key = (site, clean_pid)
                                if key not in seen:
                                    seen.add(key)
                                    patients.append({"site": site, "patient_id": clean_pid})
    except Exception as e:
        print(f"Error scanning: {e}")
print(f"Tailscale dir scan ({len(patients)} patients) took: {time.time() - t0:.3f} seconds")

# 3. Profile analysis_results streaming
t0 = time.time()
reports_ref = db.collection("analysis_results").stream()
completed_reports = [r.to_dict() for r in reports_ref]
print(f"Firestore collection 'analysis_results' stream ({len(completed_reports)} documents) took: {time.time() - t0:.3f} seconds")

# 4. Profile assignments streaming
t0 = time.time()
assign_ref = db.collection("assignments").stream()
assignments = [a.to_dict() for a in assign_ref]
print(f"Firestore collection 'assignments' stream ({len(assignments)} documents) took: {time.time() - t0:.3f} seconds")

# 5. Profile site_assignments streaming
t0 = time.time()
site_assign_ref = db.collection("site_assignments").stream()
site_assignments = [sa.to_dict() for sa in site_assign_ref]
print(f"Firestore collection 'site_assignments' stream ({len(site_assignments)} documents) took: {time.time() - t0:.3f} seconds")
