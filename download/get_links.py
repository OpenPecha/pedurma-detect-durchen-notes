import json
import os.path
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Replace with your service account credentials
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = "data/credentials/gdrive_credentials.json"
SERVICE_ACCOUNT_EMAIL = "pedurma-images@pedurma.iam.gserviceaccount.com"


def get_credentails():
    creds = None
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
    else:
        raise Exception("No credentials found")
    return creds


def get_links(folder_id):
    creds = get_credentails()
    service = build("drive", "v3", credentials=creds)

    results = (
        service.files()
        .list(q=f"'{folder_id}' in parents", fields="files(id,name)")
        .execute()
    )
    items = results.get("files", [])

    name2id = {}

    if not items:
        print("No files found.")
    else:
        for item in items:
            name2id[item["name"]] = item["id"]
    return name2id


if __name__ == "__main__":
    pecha_id2folder_id = {
        "W1PD95844": "1lT9EI2K6akBHHRJ5lzmHI0xNk2bpD7Bh",
        "W1PD96682": "1l_eEW_skg2QKoOJjwl146REViKapF5PT",
    }
    vol_ids = {}
    vol_ids_fn = Path(__file__).parent / "vol_file_ids.json"

    for pecha_id, folder_id in pecha_id2folder_id.items():
        name2id = get_links(folder_id)
        vol_ids[pecha_id] = name2id

    json.dump(vol_ids, vol_ids_fn.open("w"), indent=2)
