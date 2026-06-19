import os
import firebase_admin
from firebase_admin import credentials, storage

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    if not os.path.exists(cred_file):
        cred_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "google_credentials.json")
    
    if not os.path.exists(cred_file):
        print("Error: credentials file not found")
        return

    cred = credentials.Certificate(cred_file)
    firebase_admin.initialize_app(cred, {
        "storageBucket": "angiopysegmentation.firebasestorage.app"
    })
    
    bucket = storage.bucket()
    print("Listing files in bucket under reports/ for 1701-0022...")
    blobs = bucket.list_blobs(prefix="reports/1701-0022")
    
    found = False
    for blob in blobs:
        found = True
        print(f"Name: {blob.name}, Size: {blob.size} bytes, Public URL: {blob.public_url}")
        
    if not found:
        print("No files found in Storage for patient 1701-0022")

if __name__ == "__main__":
    main()
