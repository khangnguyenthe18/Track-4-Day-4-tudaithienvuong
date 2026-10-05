import cv2
import numpy as np
from typing import Tuple

def assess_brightness(
    frame: np.ndarray,
    min_bright: float = 40.0,
    optimal_low: float = 85.0,
    optimal_high: float = 175.0,
    max_bright: float = 230.0
) -> Tuple[float, float, float, str]:
    """
    Assess lighting condition and contrast of the frame.
    
    Returns:
        score: Normalized health score in [0.0, 1.0] (1.0 = balanced, 0.0 = too dark/blown out)
        mean_brightness: Average luminance in [0, 255]
        contrast: Standard deviation of luminance
        status: 'OPTIMAL', 'LOW_LIGHT', 'VERY_DARK', or 'OVEREXPOSED'
    """
    if frame is None or frame.size == 0:
        return 0.0, 0.0, 0.0, "INVALID_FRAME"

    if len(frame.shape) == 3:
        # Convert to YCrCb or grayscale to get illumination
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    mean_b = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Evaluate luminance
    if optimal_low <= mean_b <= optimal_high:
        bright_score = 1.0
        status = "OPTIMAL"
    elif mean_b < optimal_low:
        if mean_b <= min_bright:
            bright_score = max(0.05, (mean_b / min_bright) * 0.4)
            status = "VERY_DARK"
        else:
            bright_score = 0.4 + 0.6 * ((mean_b - min_bright) / (optimal_low - min_bright))
            status = "LOW_LIGHT"
    else: # mean_b > optimal_high
        if mean_b >= max_bright:
            bright_score = 0.2
            status = "OVEREXPOSED"
        else:
            bright_score = 1.0 - 0.7 * ((mean_b - optimal_high) / (max_bright - optimal_high))
            status = "SLIGHTLY_OVEREXPOSED"

    # Contrast penalty: flat gray or muddy scenes lose points
    contrast_factor = np.clip(contrast / 35.0, 0.5, 1.0)
    final_score = float(np.clip(bright_score * contrast_factor, 0.0, 1.0))

    return final_score, mean_b, contrast, status
