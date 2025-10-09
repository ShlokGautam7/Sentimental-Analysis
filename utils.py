# utils.py
import requests
import zipfile
import os

def send_slack(webhook_url: str, msg: str) -> bool:
    if not webhook_url:
        print("No slack webhook provided.")
        return False
    try:
        r = requests.post(webhook_url, json={"text": msg}, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print("Slack send error:", e)
        return False

def zip_files(file_paths, out_path):
    """
    Create a zip containing file_paths, return out_path.
    """
    with zipfile.ZipFile(out_path, "w") as zf:
        for p in file_paths:
            if os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
    return out_path
