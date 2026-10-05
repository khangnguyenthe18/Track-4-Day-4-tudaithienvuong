from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..tracking.trajectory import TrackInfo
from ..camera_health.health_score import CameraHealthResult

FEATURE_NAMES = [
    "detection_confidence",
    "tracking_score",
    "motion_score",
    "camera_health",
    "blur_health",
    "brightness_health",
    "occlusion_health",
    "noise_health",
    "freeze_health",
]

@dataclass
class RareCaseFeatureVector:
    track_id: int
    frame_id: int
    features: Dict[str, float]
    raw_array: np.ndarray

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "features": {k: round(v, 4) for k, v in self.features.items()}
        }

class RareCaseFeatureExtractor:
    """
    Fuses detection, tracking, and multi-dimensional camera health
    into a structured feature vector for rare-case classification.
    """
    def __init__(self):
        self.feature_names = FEATURE_NAMES

    def extract(
        self,
        track: TrackInfo,
        health: CameraHealthResult,
        frame_id: int
    ) -> RareCaseFeatureVector:
        """
        Extract fusion features for a single tracked pedestrian.
        """
        feat_dict = {
            "detection_confidence": float(np.clip(track.confidence, 0.0, 1.0)),
            "tracking_score": float(np.clip(track.tracking_consistency, 0.0, 1.0)),
            "motion_score": float(np.clip(track.motion_score, 0.0, 1.0)),
            "camera_health": float(np.clip(health.overall_health, 0.0, 1.0)),
            "blur_health": float(np.clip(health.blur_health, 0.0, 1.0)),
            "brightness_health": float(np.clip(health.brightness_health, 0.0, 1.0)),
            "occlusion_health": float(np.clip(health.occlusion_health, 0.0, 1.0)),
            "noise_health": float(np.clip(health.noise_health, 0.0, 1.0)),
            "freeze_health": float(np.clip(health.freeze_health, 0.0, 1.0)),
        }

        raw_array = np.array([feat_dict[k] for k in self.feature_names], dtype=np.float32)

        return RareCaseFeatureVector(
            track_id=track.track_id,
            frame_id=frame_id,
            features=feat_dict,
            raw_array=raw_array
        )
