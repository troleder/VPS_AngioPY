import os
import firebase_admin
from firebase_admin import credentials, firestore, auth

os.chdir("/var/www/analiza-dicom")
cred = credentials.Certificate("google_credentials.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

email = "syl.iwanczyk@gmail.com"
password = "Angio2026"
display_name = "Sylwia Iwańczyk"
role = "aneurysm_annotator"

print(f"Creating or updating user {email}...")
try:
    user = auth.get_user_by_email(email)
    print(f"User {email} already exists (UID: {user.uid}). Updating password and claims...")
    auth.update_user(user.uid, password=password, display_name=display_name)
except auth.UserNotFoundError:
    user = auth.create_user(
        email=email,
        password=password,
        display_name=display_name
    )
    print(f"User {email} successfully created with UID: {user.uid}")

# Set custom user claim
auth.set_custom_user_claims(user.uid, {"role": role})
print(f"Custom claims set for {email}: {{'role': '{role}'}}")

# Save in Firestore users collection
db.collection("users").document(email).set({
    "email": email,
    "name": display_name,
    "role": role,
    "uid": user.uid,
    "updated_at": firestore.SERVER_TIMESTAMP
}, merge=True)
print(f"Firestore user document updated in 'users/{email}'.")

# Create private storage directory on VPS
storage_dir = "/var/www/analiza-dicom/aneurysm_storage/sylwia"
os.makedirs(storage_dir, exist_ok=True)
import shutil
os.system(f"chown -R www-data:www-data /var/www/analiza-dicom/aneurysm_storage")
os.system(f"chmod -R 775 /var/www/analiza-dicom/aneurysm_storage")
print(f"Private storage directory created: {storage_dir} with www-data permissions.")
