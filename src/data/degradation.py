import cv2
import numpy as np
from typing import List, Tuple, Optional

def apply_blur(frame: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    severity: 0.0 (none) to 1.0 (extreme blur)
    """
    if severity <= 0.0:
        return frame
    k = int(severity * 30) * 2 + 1
    k = max(3, min(61, k))
    return cv2.GaussianBlur(frame, (k, k), 0)

def apply_low_light(frame: np.ndarray, brightness_factor: float = 0.3) -> np.ndarray:
    """
    brightness_factor: 1.0 (normal) to 0.1 (pitch dark)
    """
    f = np.clip(brightness_factor, 0.05, 1.0)
    table = np.array([((i / 255.0) ** 1.3 * f) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, table)

def apply_noise(frame: np.ndarray, noise_sigma: float = 20.0) -> np.ndarray:
    """
    Apply additive Gaussian noise to simulate sensor gain noise in low-light.
    """
    if noise_sigma <= 0:
        return frame
    gauss = np.random.normal(0, noise_sigma, frame.shape)
    noisy = np.clip(frame.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

def apply_dirty_lens(
    frame: np.ndarray,
    smudge_centers: Optional[List[Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Applies persistent water droplets or dirty mud smudges on camera lens.
    smudge_centers: list of (center_x, center_y, radius)
    """
    h, w = frame.shape[:2]
    if smudge_centers is None:
        smudge_centers = [
            (int(w * 0.45), int(h * 0.52), 45),
            (int(w * 0.72), int(h * 0.38), 28),
        ]

    output = frame.copy()
    overlay = frame.copy()

    for cx, cy, r in smudge_centers:
        # Draw dirty smudge with blurry edge
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        # Distort / blur inside smudge region
        sub_blur = cv2.GaussianBlur(frame, (25, 25), 0)
        # Add brownish/water droplet tint
        tint = np.array([40, 60, 70], dtype=np.uint8) # mud/dirty tint
        sub_tinted = cv2.addWeighted(sub_blur, 0.65, np.full_like(sub_blur, tint), 0.35, 0)
        
        # Soft blending mask
        soft_mask = cv2.GaussianBlur(mask, (21, 21), 0)
        alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, None] * 0.82
        overlay = (overlay.astype(np.float32) * (1.0 - alpha) + sub_tinted.astype(np.float32) * alpha).astype(np.uint8)
        
        # Draw small specular ring
        cv2.circle(overlay, (cx, cy), r, (180, 200, 220), 1)

    return overlay
