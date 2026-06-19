import os
import json
import zipfile
import shutil
from google.cloud import storage as gcs
import firebase_admin
from firebase_admin import credentials, storage

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Work from the parent directory of the script
    base_dir = os.path.dirname(script_dir)
    
    cred_file = os.path.join(base_dir, "google_credentials.json")
    if not os.path.exists(cred_file):
        print(f"Error: {cred_file} not found!")
        return

    print("Initializing Firebase connection...")
    with open(cred_file, "r") as f:
        cred_data = json.load(f)
    project_id = cred_data.get("project_id")
    print(f"Project ID: {project_id}")

    # Resolve bucket name dynamically
    bucket_name = f"{project_id}.firebasestorage.app"
    try:
        client = gcs.Client.from_service_account_json(cred_file)
        buckets = [b.name for b in client.list_buckets()]
        if buckets:
            for b in buckets:
                if b.startswith(project_id):
                    bucket_name = b
                    break
            else:
                bucket_name = buckets[0]
    except Exception as e:
        print(f"Warning during bucket discovery: {e}")

    print(f"Using bucket: {bucket_name}")
    
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_file)
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })

    bucket = storage.bucket()
    
    # List all blobs in reports/
    print("Listing files in 'reports/' folder on Firebase Storage...")
    blobs = list(bucket.list_blobs(prefix="reports/"))
    pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")]
    
    print(f"Found {len(pdf_blobs)} PDF report(s) in Storage.")
    if not pdf_blobs:
        print("No PDF files to download. Exiting.")
        return

    # Create temporary download folder in script's directory
    temp_dir = os.path.join(script_dir, "temp_exported_reports")
    os.makedirs(temp_dir, exist_ok=True)
    
    downloaded_files = []
    for idx, blob in enumerate(pdf_blobs):
        filename = os.path.basename(blob.name)
        if not filename:
            continue
        
        local_path = os.path.join(temp_dir, filename)
        print(f"[{idx+1}/{len(pdf_blobs)}] Downloading {filename} ({blob.size / 1024:.1f} KB)...")
        blob.download_to_filename(local_path)
        downloaded_files.append(local_path)

    # Zip the downloaded files
    zip_filename = os.path.join(base_dir, "exported_reports_angiopy.zip")
    print(f"Zipping {len(downloaded_files)} files to {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in downloaded_files:
            zipf.write(file_path, arcname=os.path.basename(file_path))

    # Clean up temp folder
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    
    abs_zip_path = os.path.abspath(zip_filename)
    print(f"✅ Success! Exported archive saved at:\n{abs_zip_path}")

if __name__ == "__main__":
    main()
