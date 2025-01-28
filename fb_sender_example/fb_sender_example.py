import random
import time
import threading
import uuid
import json

import firebase_admin  # pip install firebase-admin
from firebase_admin import credentials, db

SECRETS_FILE = "../xanadu-secret-f5762-firebase-adminsdk-9oc2p-1fb50744fa.json"
FB_NAMESPACE = "/xanadu/fb_sender_example"
FB_DB_URL = "https://xanadu-f5762-default-rtdb.firebaseio.com"

CONSOLE_PRINT = False
PERIOD_SEC = 1.0/60.0 # 60 fps

def main():
    cred = credentials.Certificate(SECRETS_FILE)
    firebase_admin.initialize_app(cred, {
        'databaseURL': FB_DB_URL
    })

    ref = db.reference(FB_NAMESPACE)
    print(f"Firebase database reference: {repr(ref)}")
    print(f"Period = {(PERIOD_SEC*1000.0):0.3f} ms.")
    # -- Simulate data and publish

    data = {}
    T0 = time.time()
    while True:
        start_time = time.time() - T0
        data["energy"] = random.uniform(0, 1.0)
        data["accuracy"] = random.uniform(0, 1.0)
        data["lag"] = random.uniform(0.0, 1.0)
        ref.set(
            {"data": data}
        )
        if CONSOLE_PRINT: print(f"{start_time:0.3f}: {data}")
        elapsed_time = time.time() - T0 - start_time
        time.sleep(max(0, PERIOD_SEC - elapsed_time))

if __name__ == "__main__":
    main()
