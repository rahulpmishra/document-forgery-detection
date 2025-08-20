import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def _preprocess(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    scale = 300 / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3,3), 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def verify_signature(path_ref: str, path_test: str) -> float:
    """Baseline similarity using template matching + optional SSIM.
       Returns a score normalized to [0,1]."""
    a = cv2.imread(path_ref, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(path_test, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0

    a = _preprocess(a)
    b = _preprocess(b)

    # Ensure template smaller than image
    if a.shape[0]*a.shape[1] < b.shape[0]*b.shape[1]:
        big, small = b, a
    else:
        big, small = a, b

    # Template matching
    res = cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED)
    score_tm = float(res.max()) if res is not None else 0.0
    score_tm = max(0.0, score_tm)  # clip negatives to 0

    # Optional: SSIM for additional metric
    small_resized = cv2.resize(small, (big.shape[1], big.shape[0]))
    score_ssim, _ = ssim(big, small_resized, full=True)
    
    # Combine scores (simple average)
    score = (score_tm + score_ssim) / 2
    return float(score)
