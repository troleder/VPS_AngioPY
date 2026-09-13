import os
import firebase_admin
from firebase_admin import credentials, firestore, auth
from collections import defaultdict

os.chdir("/var/www/analiza-dicom")
cred = credentials.Certificate("google_credentials.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

print("=== REGISTERED USERS IN FIREBASE AUTH ===")
try:
    for u in auth.list_users().iterate_all():
        claims = u.custom_claims or {}
        role = claims.get("role", "analyst")
        print(f"User: {u.email} | Role: {role}")
except Exception as e:
    print(f"Error listing auth users: {e}")

print("\n=== ASSIGNMENTS IN FIRESTORE ===")
docs = db.collection("assignments").stream()
user_sites = defaultdict(set)
user_patients = defaultdict(set)
assigned_sites_global = set()
assigned_patients_global = set()

for doc in docs:
    d = doc.to_dict()
    assigned_to = d.get("assigned_to", "Unknown")
    site = d.get("site")
    pid = d.get("patient_id")
    type_ = d.get("type", "patient")
    
    if type_ == "site" or (site and not pid):
        user_sites[assigned_to].add(site)
        assigned_sites_global.add(site)
    else:
        if pid:
            user_patients[assigned_to].add(pid)
            assigned_patients_global.add(pid)
            if site:
                assigned_sites_global.add(site)

for user in sorted(set(list(user_sites.keys()) + list(user_patients.keys()))):
    sites = sorted(list(user_sites[user]))
    patients = sorted(list(user_patients[user]))
    print(f"\nAnalityk: {user}")
    print(f"  - Przypisane całe ośrodki ({len(sites)}): {', '.join(sites) if sites else 'Brak'}")
    print(f"  - Przypisane pojedyncze badania pacjentów: {len(patients)}")

print("\n=== SITES ON /mnt/dane_dicom VS ASSIGNMENTS ===")
base_dir = "/mnt/dane_dicom"
disk_sites = sorted([s for s in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, s)) and not s.startswith(".")])

assigned_summary = []
unassigned_summary = []

for s in disk_sites:
    s_path = os.path.join(base_dir, s)
    patients_on_disk = [p for p in os.listdir(s_path) if os.path.isdir(os.path.join(s_path, p)) and not p.startswith(".")]
    
    # Check assignment
    assigned_to_site = [u for u, sites in user_sites.items() if s in sites]
    assigned_pids = [p for p in assigned_patients_global if p.startswith(s + "-")]
    
    if assigned_to_site:
        assigned_summary.append((s, f"Cały ośrodek -> {', '.join(assigned_to_site)} ({len(patients_on_disk)} pacjentów)"))
    elif assigned_pids:
        assigned_summary.append((s, f"Częściowo ({len(assigned_pids)}/{len(patients_on_disk)} pacjentów)"))
    else:
        unassigned_summary.append((s, len(patients_on_disk)))

print(f"Łącznie ośrodków na dysku: {len(disk_sites)}")
print(f"\n--- PRZYPISANE OŚRODKI ({len(assigned_summary)}) ---")
for s, desc in assigned_summary:
    print(f"  Ośrodek {s}: {desc}")

print(f"\n--- NIEPRZYPISANE OŚRODKI ({len(unassigned_summary)}) ---")
for s, count in unassigned_summary:
    print(f"  Ośrodek {s}: {count} pacjentów na dysku [BRAK PRZYPISANIA]")
