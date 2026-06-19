import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    docs = db.collection("site_assignments").stream()
    print("Site Assignments:")
    for doc in docs:
        print(f"Doc ID: {doc.id} | Data: {doc.to_dict()}")

if __name__ == "__main__":
    main()
