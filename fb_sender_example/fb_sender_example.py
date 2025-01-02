import random
import time
import threading

# load firebase client library from our submodule (supports anon auth)
# based on https://bitbucket.org/joetilsed/firebase/src/master/
# doesn't require admin access=
from firebase.firebase import firebase as _firebase
   
# auth
import google
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import uuid

SECRETS_FILE = "xanadu-secret-f5762-firebase-adminsdk-9oc2p-1fb50744fa.json"
FB_NAMESPACE = "/xanadu/fb_sender_example"
FB_DB_URL = "https://xanadu-f5762-default-rtdb.firebaseio.com"

def fb_callback(result):
    print("Firebase async return", result)

def fb_authsteps(credentials):
    fb = _firebase.FirebaseApplication(FB_DB_URL)
    uid = uuid.uuid4()
    def refresh_fb_token(fb, uid):
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        access_token = credentials.token
        expiration_time = credentials.expiry.astimezone()
        print(f"Refresh Firebase Access Token:")
        print(f"\tToken: {access_token[0:25]}...")
        print(f"\tExpiry: {expiration_time}")
        fb.setAccessToken(access_token)
        # Schedule the next refresh in 1 hour
        threading.Timer(3600, refresh_fb_token, args=[fb, uid]).start()
    refresh_fb_token(fb, uid)
    return fb
    
def main():
    
    # return client instance that is auth'd
    scopes = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/firebase.database"
    ]
    fb_credentials = service_account.Credentials.from_service_account_file(
        SECRETS_FILE, scopes=scopes)
    fb = fb_authsteps(fb_credentials)

    # -- Simulate data and publish

    data = {}
        
    while True:
        data["energy"] = random.uniform(-100.0, 100.0)
        data["accuracy"] = random.uniform(-100.0, 100.0)
        data["lag"] = random.uniform(-100.0, 100.0)
        print(data)
        # this updates a set of keys with the values
        result = fb.put_async(FB_NAMESPACE, "data", data, callback=fb_callback)
        # this would generate unique keys per message
        # result = fb.post_async(FB_NAMESPACE, data, callback=fb_callback)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
