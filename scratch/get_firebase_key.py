import os
import json
import base64
import urllib.request
import urllib.parse
import google.auth.transport.requests
import google.oauth2.credentials

def get_token():
    config_path = '/Users/_admin/.config/configstore/firebase-tools.json'
    if not os.path.exists(config_path):
        raise FileNotFoundError("firebase-tools.json not found")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    tokens = config.get('tokens', {})
    refresh_token = tokens.get('refresh_token')
    if not refresh_token:
        raise ValueError("refresh_token not found in config")
        
    # Refresh using google-auth library
    creds = google.oauth2.credentials.Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id='104910156361262347987',
        client_secret=None
    )
    
    request = google.auth.transport.requests.Request()
    creds.refresh(request)
    return creds.token

def list_and_create_key(access_token, project_id):
    url = f"https://iam.googleapis.com/v1/projects/{project_id}/serviceAccounts"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    })
    
    print(f"Listing service accounts for project '{project_id}'...")
    try:
        with urllib.request.urlopen(req) as res:
            accounts = json.loads(res.read().decode("utf-8")).get("accounts", [])
    except Exception as e:
        print("Failed to list service accounts:", e)
        return
        
    admin_sa = None
    for sa in accounts:
        email = sa.get("email", "")
        print(f"Found service account: {email}")
        if "firebase-adminsdk" in email:
            admin_sa = email
            
    if not admin_sa:
        print("No firebase-adminsdk service account found.")
        if accounts:
            admin_sa = accounts[0].get("email")
            print(f"Falling back to: {admin_sa}")
        else:
            print("No service accounts found in the project.")
            return
            
    print(f"\nCreating a new key for service account: {admin_sa}...")
    url = f"https://iam.googleapis.com/v1/projects/{project_id}/serviceAccounts/{admin_sa}/keys"
    req = urllib.request.Request(url, method="POST", headers={
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    })
    
    try:
        with urllib.request.urlopen(req) as res:
            key_data = json.loads(res.read().decode("utf-8"))
            private_key_json = base64.b64decode(key_data.get("privateKeyData")).decode("utf-8")
            
            with open("google_credentials.json", "w") as out:
                out.write(private_key_json)
            print("\n✅ Successfully generated and saved 'google_credentials.json'!")
            
    except Exception as e:
        print("Failed to create service account key:", e)

if __name__ == "__main__":
    try:
        token = get_token()
        list_and_create_key(token, "angiopysegmentation")
    except Exception as e:
        print("Error:", e)
