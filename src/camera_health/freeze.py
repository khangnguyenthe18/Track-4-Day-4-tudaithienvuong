import cv2
import numpy as np
from typing import Tuple, Optional

class FreezeDetector:
    """
    Detects frozen video streams or static sensor hangs.
    Computes normalized absolute frame difference D_t = mean(|I_t - I_{t-1}|).
    """
    def __init__(self, diff_eps: float = 0.08, patience_frames: int = 6):
        self.diff_eps = diff_eps
        self.patience_frames = patience_frames
        self.last_frame: Optional[np.ndarray] = None
        self.frozen_counter = 0

    def update(self, frame: np.ndarray) -> Tuple[float, float, str]:
        """
        Process frame and return freeze score.
        
        Returns:
            score: Normalized health score [0.0, 1.0] (1.0 = normal active video, 0.0 = frozen)
            mean_diff: Average absolute pixel difference from previous frame
            status: 'ACTIVE', 'POSSIBLY_STATIC', or 'FROZEN'
        """
        if frame is None or frame.size == 0:
            return 0.0, 0.0, "INVALID_FRAME"

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        small = cv2.resize(gray, (160, 90))

        if self.last_frame is None:
            self.last_frame = small
            return 1.0, 10.0, "ACTIVE"

        diff = cv2.absdiff(small, self.last_frame)
        mean_diff = float(np.mean(diff))
        self.last_frame = small

        if mean_diff < self.diff_eps:
            self.frozen_counter += 1
        else:
            # Decay counter quickly when motion resumes
            self.frozen_counter = max(0, self.frozen_counter - 2)

        if self.frozen_counter >= self.patience_frames:
            score = 0.0
            status = "FROZEN"
        elif self.frozen_counter >= (self.patience_frames // 2):
            score = 0.5
            status = "POSSIBLY_STATIC"
        else:
            score = 1.0
            status = "ACTIVE"

        return score, mean_diff, status

    def reset(self):
        self.last_frame = None
        self.frozen_counter = 0
