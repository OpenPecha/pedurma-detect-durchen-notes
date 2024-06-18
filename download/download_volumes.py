import io
import json
import os
import subprocess
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

vol_ids_fn = Path(__file__).parent / "vol_file_ids.json"
output_dir = Path(__file__).parent.parent / "input"
vol_ids = json.load(vol_ids_fn.open())
creds_json_path = (
    Path(__file__).parent.parent / "data/credentials/gdrive_credentials.json"
)


def download_gdrive_file(file_id, creds_json_path, destination_path):
    """
    Downloads a file from Google Drive using a service account with a tqdm progress bar.

    Args:
        file_id (str): The ID of the file to download from Google Drive.
        creds_json_path (str): The path to the service account credentials JSON file.
        destination_path (str): The local path where the downloaded file should be saved.

    Returns:
        str: The path to the downloaded file.
    """
    # Authenticate using service account
    creds = service_account.Credentials.from_service_account_file(
        creds_json_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    # Build the Google Drive API service
    service = build("drive", "v3", credentials=creds)

    # Get the file size for the progress bar
    file_metadata = service.files().get(fileId=file_id, fields="size, name").execute()
    file_size = int(file_metadata["size"])
    file_name = file_metadata["name"]

    # Request to download the file
    request = service.files().get_media(fileId=file_id)

    # Initialize progress bar
    with io.FileIO(destination_path, "wb") as file, tqdm(
        desc=file_name, total=file_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as progress_bar:
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            progress_bar.update(status.resumable_progress - progress_bar.n)

    return destination_path


def download_volume(vol_file_name, vol_file_id, output_dir):
    destination_path = output_dir / f"{vol_file_name}"
    download_gdrive_file(vol_file_id, creds_json_path, destination_path)


def unzip_volume(vol_name, output_dir):
    subprocess.run(
        ["unzip", "-qq", f"{output_dir}/{vol_name}.zip", "-d", f"{output_dir}"]
    )
    subprocess.run(["rm", f"{output_dir}/{vol_name}.zip"])


for pecha_id, data in vol_ids.items():
    for vol_file_name, vol_file_id in data.items():
        vol_name = vol_file_name.split(".")[0]
        if (output_dir / vol_name).is_dir():
            continue
        print(f"Downloading {vol_name}...")
        download_volume(vol_file_name, vol_file_id, output_dir)
        unzip_volume(vol_name, output_dir)
