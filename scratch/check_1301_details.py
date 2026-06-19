import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    db = firestore.client()
    
    # Fetch all analysis results
    results_ref = db.collection("analysis_results").stream()
    
    # Group results by patient_id
    patient_docs = {}
    for doc in results_ref:
        data = doc.to_dict()
        pid = data.get("patient_id", "")
        if pid.startswith("1301"):
            if pid not in patient_docs:
                patient_docs[pid] = []
            patient_docs[pid].append((doc.id, data))
            
    print(f"Total unique 1301 patients in analysis_results: {len(patient_docs)}")
    
    empty_completed = []
    has_results_completed = []
    not_completed = []
    
    for pid, docs in sorted(patient_docs.items()):
        has_completed_marker = False
        reports = []
        for doc_id, data in docs:
            dicom_name = data.get("dicom_name", "")
            phase = data.get("phase", "")
            if phase == "COMPLETED" or dicom_name == "marked_completed":
                has_completed_marker = True
            else:
                reports.append((doc_id, data))
                
        if has_completed_marker:
            if not reports:
                empty_completed.append((pid, len(docs)))
            else:
                has_results_completed.append((pid, len(reports), [r[1].get("dicom_name") for r in reports]))
        else:
            not_completed.append((pid, len(reports), [r[1].get("dicom_name") for r in reports]))
            
    print("\n--- 🟢 CASES MARKED AS COMPLETED BUT HAVE ZERO REPORTS ---")
    for pid, count in empty_completed:
        print(f"Patient ID: {pid} (has completed marker doc, but 0 reports)")
    print(f"Total empty completed cases: {len(empty_completed)}")
    
    print("\n--- 🟢 CASES MARKED AS COMPLETED AND HAVE REPORTS ---")
    for pid, r_count, r_names in has_results_completed:
        print(f"Patient ID: {pid} | Reports count: {r_count} | Dicom names: {r_names}")
    print(f"Total completed cases with reports: {len(has_results_completed)}")

    print("\n--- 🟡 CASES NOT COMPLETED (ONLY REPORTS, NO COMPLETED MARKER) ---")
    for pid, r_count, r_names in not_completed:
        print(f"Patient ID: {pid} | Reports count: {r_count} | Dicom names: {r_names}")
    print(f"Total in-progress cases: {len(not_completed)}")

if __name__ == "__main__":
    main()
