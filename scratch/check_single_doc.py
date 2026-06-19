import firebase_admin
from firebase_admin import credentials, firestore

def main():
    cred_file = "/var/www/analiza-dicom/google_credentials.json"
    cred = credentials.Certificate(cred_file)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    # Check 1301-0011 LAD PRE-PCI report
    doc_id = "inpuYDF2unj7o98GCAQ8"
    doc = db.collection("analysis_results").document(doc_id).get()
    if doc.exists:
        print(f"Document {doc_id} exists. Data:")
        data = doc.to_dict()
        for k, v in sorted(data.items()):
            if k == "pdf_data":
                print(f"  {k}: [binary data, size {len(v)} bytes]")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"Document {doc_id} does not exist.")

if __name__ == "__main__":
    main()
