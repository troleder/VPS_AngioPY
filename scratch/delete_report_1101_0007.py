import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    cred_file = os.path.join(base_dir, "google_credentials.json")
    
    if not os.path.exists(cred_file):
        print(f"Error: {cred_file} not found!")
        return

    print("Initializing Firebase...")
    cred = credentials.Certificate(cred_file)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    print("Searching for matching report...")
    # Query for the specific report
    docs_ref = db.collection("analysis_results") \
                 .where("patient_id", "==", "1101-0007") \
                 .where("dicom_name", "==", "I_000005.dcm") \
                 .where("phase", "==", "POST-PCI") \
                 .where("aha", "==", "5") \
                 .stream()

    found_docs = list(docs_ref)
    print(f"Found {len(found_docs)} matching document(s).")

    for doc in found_docs:
        doc_id = doc.id
        doc_data = doc.to_dict()
        print(f"Document ID: {doc_id}")
        print(f"Data: {json.dumps(doc_data, indent=2, default=str)}")
        
        # Deleting the document
        print(f"Deleting document {doc_id}...")
        db.collection("analysis_results").document(doc_id).delete()
        print(f"✅ Document {doc_id} deleted successfully.")

    if not found_docs:
        print("❌ No matching document found in the database. Please verify the query fields.")

if __name__ == "__main__":
    main()
