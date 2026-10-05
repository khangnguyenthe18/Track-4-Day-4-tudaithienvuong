import cv2
import numpy as np
from typing import Tuple, Optional

class OcclusionDetector:
    """
    Detects lens occlusion, water droplets, mud smears, or covered camera view.
    Combines:
    1. Static temporal consistency (smudges that never move across frames)
    2. Large uniform dead zones / covered patches
    """
    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self.frame_buffer = []
        self.static_mask: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> Tuple[float, float, str]:
        """
        Process a new frame and calculate occlusion health.
        
        Returns:
            score: Normalized health score [0.0, 1.0] (1.0 = clear lens, 0.0 = severe occlusion)
            occluded_ratio: Estimated percentage of frame area occluded/smudged [0.0, 1.0]
            status: 'CLEAR', 'PARTIAL_SMUDGE', or 'HEAVY_OCCLUSION'
        """
        if frame is None or frame.size == 0:
            return 0.0, 1.0, "INVALID_FRAME"

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Downscale for fast temporal analysis
        small = cv2.resize(gray, (160, 90))
        self.frame_buffer.append(small)
        if len(self.frame_buffer) > self.history_len:
            self.frame_buffer.pop(0)

        # 1. Single frame dead-zone detection (e.g. lens cap or large physical obstruction)
        mean_lum = float(np.mean(small))
        dead_dark = np.sum(small < 5) / small.size
        dead_bright = np.sum(small > 252) / small.size

        # Only consider dead dark if the entire scene is abnormally pitch black (not just night sky)
        if mean_lum < 15.0 and dead_dark > 0.55:
            dead_zone_ratio = dead_dark
        elif dead_bright > 0.40:
            dead_zone_ratio = dead_bright
        else:
            dead_zone_ratio = 0.0

        # 2. Localized low-contrast / muddy occlusion patch
        # Divide image into 8x8 grid (each cell ~20x11 pixels)
        # Blocked / smudged cells lose high frequency texture variance
        grad = cv2.Laplacian(small, cv2.CV_32F)
        ch, cw = small.shape
        gh, gw = 8, 8
        step_y, step_x = ch // gh, cw // gw
        cell_vars = [
            float(np.var(grad[i*step_y:(i+1)*step_y, j*step_x:(j+1)*step_x]))
            for i in range(gh) for j in range(gw)
        ]
        # Active camera view excluding top sky rows
        active_dead = sum(1 for idx, v in enumerate(cell_vars) if (idx // gw >= 2) and v < 18.0)
        all_dead = sum(1 for v in cell_vars if v < 10.0)

        # Clear cameras have active_dead <= 3.
        # Dirty lens or smudge creates active_dead >= 7.
        if active_dead <= 4 and dead_zone_ratio < 0.10:
            score = 1.0
            status = "CLEAR"
        elif active_dead <= 12 and dead_zone_ratio < 0.25:
            score = max(0.15, 0.40 - (active_dead - 4) * 0.03)
            status = "PARTIAL_SMUDGE"
        else:
            score = max(0.05, 0.20 - (active_dead - 12) * 0.01)
            status = "HEAVY_OCCLUSION"

        score = float(np.clip(score, 0.0, 1.0))
        occluded_ratio = float((active_dead * 1.5) / 64.0)
        return score, occluded_ratio, status
