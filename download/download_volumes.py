import json
import subprocess
from pathlib import Path

vol_ids_fn = Path(__file__).parent / "vol_file_ids.json"

output_dir = Path(__file__).parent.parent / "input"

vol_ids = json.load(vol_ids_fn.open())


def download_volume(vol_file_name, vol_file_id, output_dir):
    subprocess.run(
        ["gdown", "--id", vol_file_id, "-O", f"{output_dir}/{vol_file_name}"]
    )


def unzip_volume(vol_name, output_dir):
    subprocess.run(["unzip", f"{output_dir}/{vol_name}.zip", "-d", f"{output_dir}"])
    subprocess.run(["rm", f"{output_dir}/{vol_name}.zip"])


for pecha_id, data in vol_ids.items():
    for vol_file_name, vol_file_id in data.items():
        vol_name = vol_file_name.split(".")[0]
        if (output_dir / vol_name).is_dir():
            continue
        print(f"Downloading {vol_name}...")
        download_volume(vol_file_name, vol_file_id, output_dir)
        unzip_volume(vol_name, output_dir)
