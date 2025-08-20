import os
from pathlib import Path

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

def save_upload(file, name_hint):
    """Save a Streamlit UploadedFile to disk and return its path."""
    if file is None:
        return None
    suffix = Path(file.name).suffix.lower()
    out = TMP_DIR / f"{name_hint}{suffix if suffix else '.png'}"
    with open(out, 'wb') as f:
        f.write(file.read())
    return str(out)
