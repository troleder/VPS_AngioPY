import os
import io
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    print("Fetching results from Firestore...")
    reports_ref = db.collection("analysis_results").stream()
    
    rows = []
    for r in reports_ref:
        d = r.to_dict()
        if d.get("dicom_name") == "marked_completed" or d.get("phase") == "COMPLETED":
            continue
            
        metrics = d.get("metrics") or {}
        
        def format_num(val, round_digits=None):
            if val is None or val == "N/A" or val == "—":
                return "N/A"
            try:
                f_val = float(val)
                if round_digits is not None:
                    return round(f_val, round_digits)
                return f_val
            except:
                return val
        
        row = {
            "Patient ID": d.get("patient_id"),
            "DICOM Name": d.get("dicom_name"),
            "Phase": d.get("phase"),
            "Vessel": d.get("vessel"),
            "AHA Segment": d.get("aha"),
            "FFR position registered": d.get("ffr_registered"),
            "Other lesion >50% distal": d.get("other_lesion_distal"),
            "Known Occluded Vessel": d.get("known_occlude"),
            "Max Prox [mm]": format_num(metrics.get("prox_diam_mm"), 2),
            "Max Dist [mm]": format_num(metrics.get("dist_diam_mm"), 2),
            "Reference [mm]": format_num(metrics.get("ref_diam_mm"), 2),
            "MLD [mm]": format_num(metrics.get("mld_mm"), 2),
            "% Diameter Stenosis": format_num(metrics.get("pct_diameter_stenosis"), 1),
            "% Area Stenosis": format_num(metrics.get("pct_area_stenosis"), 1),
            "Lesion Length [mm]": format_num(metrics.get("lesion_length_mm"), 2),
            "TIMI Grade": metrics.get("timi_grade"),
            "TFC": metrics.get("tfc")
        }
        rows.append(row)
        
    print(f"Total report rows found: {len(rows)}")
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        # Sort values
        df.sort_values(by=["Patient ID", "Phase"], inplace=True)
        
        # Save path on VPS
        target_path = "/var/www/analiza-dicom/Library/Mobile Documents/com~apple~CloudDocs/AngioPy/AngioPy.xlsx"
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        df.to_excel(target_path, index=False)
        print(f"Excel file successfully synchronized and saved to: {target_path}")
        
        # Also copy it to /var/www/analiza-dicom/reports/AngioPy.xlsx to make sure all instances can read it
        reports_path = "/var/www/analiza-dicom/reports/AngioPy.xlsx"
        os.makedirs(os.path.dirname(reports_path), exist_ok=True)
        df.to_excel(reports_path, index=False)
        print(f"Excel file successfully copied to: {reports_path}")
    else:
        print("No rows to save.")

if __name__ == "__main__":
    main()
