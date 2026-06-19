import os
import json
import urllib.request
import urllib.parse

def test_refresh():
    config_path = '/Users/_admin/.config/configstore/firebase-tools.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    refresh_token = config['tokens']['refresh_token']
    
    url = "https://oauth2.googleapis.com/token"
    data = urllib.parse.urlencode({
        "client_id": "104910156361262347987",
        "client_secret": "",
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req) as res:
            response = json.loads(res.read().decode("utf-8"))
            print("Access token generated:", response.get("access_token")[:15] + "...")
    except Exception as e:
        print("Refresh failed:", e)

if __name__ == "__main__":
    test_refresh()
