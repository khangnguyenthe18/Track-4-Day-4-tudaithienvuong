import cv2
import numpy as np
from typing import Tuple

def assess_blur(
    frame: np.ndarray,
    healthy_var: float = 120.0,
    critical_var: float = 25.0
) -> Tuple[float, float, str]:
    """
    Assess image blur using the Variance of Laplacian method.
    
    Args:
        frame: BGR or grayscale numpy image
        healthy_var: Variance threshold above which image is considered sharp (score ~1.0)
        critical_var: Variance threshold below which image is severely blurred (score ~0.0)
        
    Returns:
        score: Normalized health score in [0.0, 1.0] (1.0 = sharp, 0.0 = severe blur)
        raw_variance: Raw variance of Laplacian
        status: Descriptive tag ('SHARP', 'MODERATE_BLUR', 'HEAVY_BLUR')
    """
    if frame is None or frame.size == 0:
        return 0.0, 0.0, "INVALID_FRAME"
        
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    # Map to [0.0, 1.0]
    if variance >= healthy_var:
        score = 1.0
    elif variance <= critical_var:
        score = max(0.0, variance / critical_var * 0.25)
    else:
        # Linear interpolation between critical and healthy
        score = 0.25 + 0.75 * ((variance - critical_var) / (healthy_var - critical_var))

    score = float(np.clip(score, 0.0, 1.0))

    if score > 0.7:
        status = "SHARP"
    elif score > 0.35:
        status = "MODERATE_BLUR"
    else:
        status = "HEAVY_BLUR"

    return score, variance, status
