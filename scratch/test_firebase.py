import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

try:
    print("Initializing Firebase Admin SDK...")
    cred = credentials.Certificate("google_credentials.json")
    
    # Initialize with default credentials (which uses project_id from the JSON)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    print("Attempting to write a test document to 'test_connection' collection...")
    doc_ref = db.collection("test_connection").document("test_doc")
    doc_ref.set({
        "status": "success",
        "message": "Firebase connection established successfully!",
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    
    print("Successfully wrote document. Now reading it back...")
    doc = doc_ref.get()
    print("Document data:", doc.to_dict())
    print("\n--- CONNECTION TEST PASSED ---")
except Exception as e:
    print("\n--- CONNECTION TEST FAILED ---")
    print(e)
