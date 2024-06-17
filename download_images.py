import os.path

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Replace with your service account credentials
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = "data/credentials/gdrive_credentials.json"
SERVICE_ACCOUNT_EMAIL = "pedurma-images@pedurma.iam.gserviceaccount.com"

creds = None
if os.path.exists(SERVICE_ACCOUNT_FILE):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
else:
    raise Exception("No credentials found")

service = build("drive", "v3", credentials=creds)

# Replace with the ID of the public folder
folder_id = "1lT9EI2K6akBHHRJ5lzmHI0xNk2bpD7Bh"

results = (
    service.files().list(q=f"'{folder_id}' in parents", fields="files(id)").execute()
)
items = results.get("files", [])

if not items:
    print("No files found.")
else:
    for item in items:
        print(item["id"])
