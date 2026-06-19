import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    docs = db.collection("analysis_results").where("patient_id", "==", "1301-0004").stream()
    for doc in docs:
        print(f"Doc ID: {doc.id}")
        data = doc.to_dict()
        for k, v in sorted(data.items()):
            print(f"  {k}: {v}")
        print("-" * 50)

if __name__ == "__main__":
    main()
