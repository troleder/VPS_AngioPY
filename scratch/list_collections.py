import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    collections = db.collections()
    print("Firestore Collections:")
    for col in collections:
        print(f" - {col.id}")

if __name__ == "__main__":
    main()
