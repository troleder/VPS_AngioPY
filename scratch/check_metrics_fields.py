import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    results_ref = db.collection("analysis_results").stream()
    
    patient_docs = {}
    for doc in results_ref:
        data = doc.to_dict()
        pid = data.get("patient_id", "")
        if pid.startswith("1301"):
            if pid not in patient_docs:
                patient_docs[pid] = []
            patient_docs[pid].append((doc.id, data))
            
    print("Analyzing metrics for 1301 patients...")
    empty_metrics_cases = []
    valid_metrics_cases = []
    
    for pid, docs in sorted(patient_docs.items()):
        completed = False
        qca_docs = []
        for doc_id, data in docs:
            dicom_name = data.get("dicom_name", "")
            phase = data.get("phase", "")
            if phase == "COMPLETED" or dicom_name == "marked_completed":
                completed = True
            else:
                qca_docs.append((doc_id, data))
                
        if completed:
            # Check QCA docs
            has_valid_qca = False
            for doc_id, qca in qca_docs:
                metrics = qca.get("metrics") or {}
                # Check if it has actual values
                mld = metrics.get("mld_mm") or metrics.get("mld")
                ref = metrics.get("ref_diam_mm") or metrics.get("ref")
                pct_ds = metrics.get("pct_diameter_stenosis") or metrics.get("pct_diam")
                
                if mld is not None and mld != "N/A":
                    has_valid_qca = True
                    
            if not has_valid_qca:
                empty_metrics_cases.append(pid)
            else:
                valid_metrics_cases.append(pid)
                
    print(f"\n--- CASES MARKED AS COMPLETED BUT HAVE NO VALID MEASUREMENTS (MLD is missing/N/A) ---")
    for pid in empty_metrics_cases:
        print(f"  {pid}")
    print(f"Total: {len(empty_metrics_cases)}")

    print(f"\n--- CASES MARKED AS COMPLETED AND HAVE VALID MEASUREMENTS ---")
    for pid in valid_metrics_cases:
        print(f"  {pid}")
    print(f"Total: {len(valid_metrics_cases)}")

if __name__ == "__main__":
    main()
