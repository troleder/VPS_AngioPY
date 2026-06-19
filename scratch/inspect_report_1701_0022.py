import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    if not os.path.exists(cred_file):
        cred_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "google_credentials.json")
    
    if not os.path.exists(cred_file):
        print("Error: credentials file not found")
        return

    cred = credentials.Certificate(cred_file)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    print("Querying Firestore for patient 1701-0022...")
    docs = db.collection("analysis_results")\
             .where("patient_id", "==", "1701-0022")\
             .stream()

    found = False
    for doc in docs:
        found = True
        print(f"\nDocument ID: {doc.id}")
        data = doc.to_dict()
        print(json.dumps(data, indent=2, default=str))

    if not found:
        print("No documents found for patient 1701-0022")

if __name__ == "__main__":
    main()
