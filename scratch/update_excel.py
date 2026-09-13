import os
import sys
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

def update_excel():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"storageBucket": "angiopysegmentation.firebasestorage.app"})
    db = firestore.client()

    excel_file = "/tmp/AngioPy_Analysis_Results_2026.xlsx"

    print("Fetching results from Firestore...")
    reports_ref = db.collection("analysis_results").stream()

    rows_stenoses = []
    rows_completed = []

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

    for r in reports_ref:
        d = r.to_dict()
        pid = d.get("patient_id", "")
        phase = d.get("phase", "")
        dicom_name = d.get("dicom_name", "")
        ts = d.get("timestamp")
        ts_str = str(ts) if ts is not None else ""

        if phase == "COMPLETED" or dicom_name == "marked_completed":
            site = pid.split("-")[0] if "-" in pid else "Other"
            rows_completed.append({
                "Site": site,
                "Patient ID": pid,
                "Status": "COMPLETED",
                "Completion Date/Time": ts_str
            })
            continue

        metrics = d.get("metrics") or {}
        site = pid.split("-")[0] if "-" in pid else "Other"
        rows_stenoses.append({
            "Site": site,
            "Patient ID": pid,
            "DICOM Name": dicom_name,
            "Phase": phase,
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
            "Lesion/Stent Length [mm]": format_num(metrics.get("lesion_length_mm"), 2),
            "TIMI Grade": metrics.get("timi_grade")
        })

    print(f"Loaded {len(rows_stenoses)} stenoses and {len(rows_completed)} completed records.")

    df_stenoses = pd.DataFrame(rows_stenoses)
    if not df_stenoses.empty:
        df_stenoses.sort_values(by=["Site", "Patient ID", "Phase"], inplace=True)

    df_completed = pd.DataFrame(rows_completed)
    if not df_completed.empty:
        df_completed.sort_values(by=["Site", "Patient ID"], inplace=True)

    print(f"Saving Excel to {excel_file}...")
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df_stenoses.to_excel(writer, sheet_name="Stenosis_Details", index=False)
        df_completed.to_excel(writer, sheet_name="Completed_Studies", index=False)
    print("Excel saved successfully.")

if __name__ == "__main__":
    update_excel()
