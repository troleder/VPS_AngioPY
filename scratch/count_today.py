import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone, timedelta

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    cred_file = os.path.join(base_dir, "google_credentials.json")
    
    if not os.path.exists(cred_file):
        # Fallback to VPS absolute path
        cred_file = "/var/www/analiza-dicom/google_credentials.json"
        
    if not os.path.exists(cred_file):
        print(f"Error: {cred_file} not found!")
        return

    cred = credentials.Certificate(cred_file)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # User timezone is +02:00 (Poland)
    # Today starts at 2026-06-14 00:00:00 +02:00, which is 2026-06-13 22:00:00 UTC
    local_tz = timezone(timedelta(hours=2))
    start_of_day_local = datetime(2026, 6, 14, 0, 0, 0, tzinfo=local_tz)
    start_of_day_utc = start_of_day_local.astimezone(timezone.utc)
    
    print(f"Counting patients worked on since (Local): {start_of_day_local}")
    print(f"Counting patients worked on since (UTC):   {start_of_day_utc}")

    # Query all results since start of day (indexed by default)
    docs = db.collection("analysis_results")\
             .where("timestamp", ">=", start_of_day_utc)\
             .stream()

    completed_patients = set()
    inprogress_patients = set()
    total_stenoses = 0

    for doc in docs:
        data = doc.to_dict()
        analyst = data.get("analyst")
        
        # Filter by analyst in memory
        if analyst != "tomaszroleder":
            continue
            
        pid = data.get("patient_id")
        phase = data.get("phase")
        ts = data.get("timestamp")
        
        # Convert UTC timestamp to local timezone for display
        if ts:
            ts_local = ts.astimezone(local_tz)
            ts_str = ts_local.strftime("%H:%M:%S")
        else:
            ts_str = "N/A"

        if phase == "COMPLETED":
            completed_patients.add(pid)
            print(f"[🟢 COMPLETED] Patient ID: {pid} at {ts_str}")
        else:
            inprogress_patients.add(pid)
            total_stenoses += 1
            print(f"[🔵 STENOSIS]  Patient ID: {pid} ({data.get('vessel', 'N/A')} {data.get('phase', 'N/A')}) at {ts_str}")

    # A patient is completed if they have a COMPLETED marker.
    # Otherwise, if they only have stenosis records today, they are in progress.
    only_inprogress = inprogress_patients - completed_patients

    print("\nSummary:")
    print(f"🟢 Completed Patients:   {len(completed_patients)} ({sorted(list(completed_patients))})")
    print(f"🟡 In-Progress Patients: {len(only_inprogress)} ({sorted(list(only_inprogress))})")
    print(f"Total Stenoses Saved:    {total_stenoses}")

if __name__ == "__main__":
    main()
