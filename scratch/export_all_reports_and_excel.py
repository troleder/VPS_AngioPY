import os
import sys
import shutil
import zipfile
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import firebase_admin
from firebase_admin import credentials, firestore, storage

def run_export():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"storageBucket": "angiopysegmentation.firebasestorage.app"})
    db = firestore.client()
    bucket = storage.bucket()

    export_dir = "/tmp/angiopy_reports_export"
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    excel_file = "/tmp/AngioPy_Wyniki_Analiz_2026.xlsx"
    zip_file = "/tmp/AngioPy_Wszystkie_Raporty_PDF_2026.zip"

    print("1. Fetching all analysis results from Firestore...")
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
        analyst = d.get("analyst", "")
        pdf_url = d.get("pdf_url", "")

        if phase == "COMPLETED" or dicom_name == "marked_completed":
            site = pid.split("-")[0] if "-" in pid else "Inne"
            rows_completed.append({
                "Ośrodek": site,
                "Patient ID": pid,
                "Status": "COMPLETED",
                "Analyst": analyst,
                "Zakończono (Data/Czas)": ts_str,
                "Master PDF URL": pdf_url
            })
            continue

        metrics = d.get("metrics") or {}
        site = pid.split("-")[0] if "-" in pid else "Inne"
        rows_stenoses.append({
            "Ośrodek": site,
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
            "Lesion Length [mm]": format_num(metrics.get("lesion_length_mm"), 2),
            "TIMI Grade": metrics.get("timi_grade"),
            "TFC": metrics.get("tfc"),
            "Analyst": analyst,
            "Timestamp": ts_str,
            "PDF URL": pdf_url
        })

    print(f"Loaded {len(rows_stenoses)} stenoses and {len(rows_completed)} completed patient records.")

    df_stenoses = pd.DataFrame(rows_stenoses)
    if not df_stenoses.empty:
        df_stenoses.sort_values(by=["Ośrodek", "Patient ID", "Phase"], inplace=True)

    df_completed = pd.DataFrame(rows_completed)
    if not df_completed.empty:
        df_completed.sort_values(by=["Ośrodek", "Patient ID"], inplace=True)

    print(f"Saving Excel file to {excel_file}...")
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df_stenoses.to_excel(writer, sheet_name="Stenozy_Szczegoly", index=False)
        df_completed.to_excel(writer, sheet_name="Zakonczone_Badania", index=False)
    print("Excel saved successfully.")

    print("2. Listing blobs in Firebase Storage 'reports/'...")
    blobs = list(bucket.list_blobs(prefix="reports/"))
    print(f"Found {len(blobs)} files in Firebase Storage reports/.")

    # Download blobs in parallel using ThreadPoolExecutor
    print("3. Downloading PDF reports in parallel (30 workers)...")
    def download_blob(blob):
        fn = os.path.basename(blob.name)
        if not fn or not fn.endswith(".pdf"):
            return
        dst = os.path.join(export_dir, fn)
        try:
            blob.download_to_filename(dst)
        except Exception as e:
            print(f"Error downloading {blob.name}: {e}")

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(download_blob, blobs))

    pdf_files = [f for f in os.listdir(export_dir) if f.endswith(".pdf")]
    print(f"Downloaded {len(pdf_files)} PDF files to {export_dir}.")

    print("4. Creating ZIP archive...")
    if os.path.exists(zip_file):
        os.remove(zip_file)

    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(excel_file, arcname="AngioPy_Wyniki_Analiz_2026.xlsx")
        for idx, pdf in enumerate(pdf_files):
            zipf.write(os.path.join(export_dir, pdf), arcname=os.path.join("PDF_Reports", pdf))
            if (idx + 1) % 500 == 0 or (idx + 1) == len(pdf_files):
                print(f"Zipped {idx + 1}/{len(pdf_files)} files...")

    zip_size_mb = os.path.getsize(zip_file) / (1024 * 1024)
    print(f"ZIP created successfully: {zip_file} ({zip_size_mb:.2f} MB)")

    # Cleanup unzipped folder
    shutil.rmtree(export_dir)
    print("Temporary folder cleaned up. Export complete!")

if __name__ == "__main__":
    run_export()
