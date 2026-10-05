from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .blur import assess_blur
from .brightness import assess_brightness
from .noise import assess_noise
from .occlusion import OcclusionDetector
from .freeze import FreezeDetector

@dataclass
class CameraHealthResult:
    frame_id: int
    blur_health: float
    brightness_health: float
    occlusion_health: float
    noise_health: float
    freeze_health: float
    overall_health: float
    status_tags: List[str]
    raw_metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "blur_health": round(self.blur_health, 3),
            "brightness_health": round(self.brightness_health, 3),
            "occlusion_health": round(self.occlusion_health, 3),
            "noise_health": round(self.noise_health, 3),
            "freeze_health": round(self.freeze_health, 3),
            "overall_health": round(self.overall_health, 3),
            "status_tags": self.status_tags,
            "raw_metrics": {k: round(v, 2) for k, v in self.raw_metrics.items()}
        }

class CameraHealthAssessor:
    """
    Main Camera Health Assessment orchestrator.
    Combines blur, low-light, occlusion, noise, and freeze checks.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        blur_healthy_var: float = 120.0,
        blur_critical_var: float = 25.0
    ):
        self.weights = weights or {
            "blur": 0.30,
            "brightness": 0.20,
            "occlusion": 0.25,
            "noise": 0.15,
            "freeze": 0.10,
        }
        self.blur_healthy_var = blur_healthy_var
        self.blur_critical_var = blur_critical_var
        
        self.occlusion_detector = OcclusionDetector()
        self.freeze_detector = FreezeDetector()
        self.frame_counter = 0

    def assess(self, frame: np.ndarray, frame_id: Optional[int] = None) -> CameraHealthResult:
        if frame_id is None:
            self.frame_counter += 1
            f_id = self.frame_counter
        else:
            f_id = frame_id

        # 1. Blur
        h_blur, raw_var, blur_tag = assess_blur(
            frame, healthy_var=self.blur_healthy_var, critical_var=self.blur_critical_var
        )

        # 2. Brightness
        h_bright, mean_lum, contrast, bright_tag = assess_brightness(frame)

        # 3. Noise
        h_noise, noise_std, noise_tag = assess_noise(frame)

        # 4. Occlusion
        h_occl, occl_ratio, occl_tag = self.occlusion_detector.update(frame)

        # 5. Freeze
        h_freeze, mean_diff, freeze_tag = self.freeze_detector.update(frame)

        # Overall weighted health
        w = self.weights
        overall = (
            w["blur"] * h_blur +
            w["brightness"] * h_bright +
            w["occlusion"] * h_occl +
            w["noise"] * h_noise +
            w["freeze"] * h_freeze
        )
        overall = float(np.clip(overall, 0.0, 1.0))

        # Collect warning/error tags
        tags = []
        if blur_tag != "SHARP":
            tags.append(blur_tag)
        if bright_tag != "OPTIMAL":
            tags.append(bright_tag)
        if occl_tag != "CLEAR":
            tags.append(occl_tag)
        if noise_tag != "CLEAN":
            tags.append(noise_tag)
        if freeze_tag != "ACTIVE":
            tags.append(freeze_tag)
        if not tags:
            tags.append("NORMAL_HEALTH")

        raw_metrics = {
            "laplacian_var": raw_var,
            "mean_luminance": mean_lum,
            "contrast": contrast,
            "noise_std": noise_std,
            "occluded_ratio": occl_ratio,
            "frame_diff": mean_diff,
        }

        return CameraHealthResult(
            frame_id=f_id,
            blur_health=h_blur,
            brightness_health=h_bright,
            occlusion_health=h_occl,
            noise_health=h_noise,
            freeze_health=h_freeze,
            overall_health=overall,
            status_tags=tags,
            raw_metrics=raw_metrics
        )

    def reset(self):
        self.frame_counter = 0
        self.occlusion_detector = OcclusionDetector()
        self.freeze_detector.reset()
