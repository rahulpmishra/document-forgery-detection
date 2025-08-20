from typing import Dict, Tuple
from PIL import Image
import pytesseract
import re

# If Tesseract is not on PATH, set it here (Windows example):
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def extract_text(img_path: str) -> str:
    try:
        img = Image.open(img_path).convert('RGB') 
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"[OCR ERROR] {e}"

def check_entities(text: str, expected: Dict[str, str]) -> Tuple[float, Dict[str, bool]]:
    """Very simple rule check: substring search (case-insensitive).
    Returns (score in 0..1, flags dict).
    """
    t = text.lower()
    flags = {}
    hits = 0
    for k, v in expected.items():
        if not v:
            flags[k] = True
            hits += 1
            continue
        found = v.lower() in t
        flags[k] = found
        if found:
            hits += 1
    score = hits / max(1, len(expected))
    return score, flags

def extract_and_check(img_path: str, expected: Dict[str, str]):
    text = extract_text(img_path)
    score, flags = check_entities(text, expected)
    return text, score, flags
