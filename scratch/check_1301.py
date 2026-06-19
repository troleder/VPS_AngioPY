import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
        
    print(f"Loading credentials from: {os.path.abspath(cred_file)}")
    cred = credentials.Certificate(cred_file)
    firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    # 1. Fetch analysis_results starting with 1301
    print("\n--- FETCHING ANALYSIS RESULTS FOR 1301 ---")
    results_ref = db.collection("analysis_results").stream()
    r_count = 0
    for doc in results_ref:
        data = doc.to_dict()
        pid = data.get("patient_id", "")
        if pid.startswith("1301"):
            print(f"Doc ID: {doc.id} | Patient ID: {pid} | DICOM Name: {data.get('dicom_name')} | Phase: {data.get('phase')} | User: {data.get('user')} | Timestamp: {data.get('timestamp')}")
            r_count += 1
    print(f"Total 1301 analysis results found: {r_count}")

    # 2. Fetch assignments starting with 1301
    print("\n--- FETCHING ASSIGNMENTS FOR 1301 ---")
    assign_ref = db.collection("assignments").stream()
    a_count = 0
    for doc in assign_ref:
        data = doc.to_dict()
        pid = data.get("patient_id", "")
        if pid.startswith("1301"):
            print(f"Doc ID: {doc.id} | Patient ID: {pid} | Assigned To: {data.get('assigned_to')} | Status: {data.get('status')} | Timestamp: {data.get('unassigned_at') or data.get('assigned_at')}")
            a_count += 1
    print(f"Total 1301 assignments found: {a_count}")

if __name__ == "__main__":
    main()
