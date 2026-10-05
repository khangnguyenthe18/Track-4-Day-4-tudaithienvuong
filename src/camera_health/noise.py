import cv2
import numpy as np
from typing import Tuple

def assess_noise(
    frame: np.ndarray,
    noise_clean_thresh: float = 3.5,
    noise_heavy_thresh: float = 25.0
) -> Tuple[float, float, str]:
    """
    Estimate sensor noise by comparing original grayscale with median-filtered version.
    
    Returns:
        score: Normalized health score in [0.0, 1.0] (1.0 = pristine clean, 0.0 = high noise)
        noise_level: Standard deviation of residual noise
        status: 'CLEAN', 'SLIGHT_NOISE', 'HEAVY_NOISE'
    """
    if frame is None or frame.size == 0:
        return 0.0, 0.0, "INVALID_FRAME"

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Fast downscale if large for performance
    h, w = gray.shape
    if h > 480:
        gray = cv2.resize(gray, (w // 2, h // 2))

    # Fast median filter residual
    smoothed = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, smoothed)
    noise_std = float(np.std(diff))

    if noise_std <= noise_clean_thresh:
        score = 1.0
        status = "CLEAN"
    elif noise_std >= noise_heavy_thresh:
        score = 0.1
        status = "HEAVY_NOISE"
    else:
        score = 1.0 - 0.9 * ((noise_std - noise_clean_thresh) / (noise_heavy_thresh - noise_clean_thresh))
        status = "SLIGHT_NOISE"

    score = float(np.clip(score, 0.0, 1.0))
    return score, noise_std, status
