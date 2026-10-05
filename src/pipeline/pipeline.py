import cv2
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..detection.detector import PedestrianDetector, DetectionResult
from ..tracking.tracker import PedestrianTracker, TrackInfo
from ..camera_health.health_score import CameraHealthAssessor, CameraHealthResult
from ..rare_case.features import RareCaseFeatureExtractor, RareCaseFeatureVector
from ..rare_case.classifier import RareCaseClassifier, RareCasePrediction

@dataclass
class PipelineFrameResult:
    frame_id: int
    annotated_frame: np.ndarray
    camera_health: CameraHealthResult
    tracks: List[TrackInfo]
    predictions: List[RareCasePrediction]
    fps: float
    latency_ms: float
    primary_decision: str
    primary_confidence: float
    primary_explanation: str

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "primary_decision": self.primary_decision,
            "primary_confidence": round(self.primary_confidence, 3),
            "primary_explanation": self.primary_explanation,
            "camera_health": self.camera_health.to_dict(),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "bbox": [round(c, 1) for c in t.bbox],
                    "confidence": round(t.confidence, 3),
                    "tracking_consistency": round(t.tracking_consistency, 3),
                    "motion_score": round(t.motion_score, 3),
                    "persistence": t.persistence,
                    "is_stationary": t.is_stationary,
                }
                for t in self.tracks
            ],
            "predictions": [p.to_dict() for p in self.predictions]
        }

class RarePedestrianPipeline:
    """
    End-to-end processing pipeline integrating:
    1. Camera Health Assessment (Blur, Brightness, Noise, Occlusion, Freeze)
    2. Pedestrian Detection (YOLOv8)
    3. Multi-Object Tracking (ByteTrack)
    4. Feature Fusion & Rare Case Classification (Normal vs Real Rare Event vs Camera Artifact)
    """
    def __init__(
        self,
        detector: Optional[PedestrianDetector] = None,
        tracker: Optional[PedestrianTracker] = None,
        health_assessor: Optional[CameraHealthAssessor] = None,
        classifier: Optional[RareCaseClassifier] = None,
        model_path: str = "weights/rare_case_model.pkl"
    ):
        self.detector = detector or PedestrianDetector()
        self.tracker = tracker or PedestrianTracker()
        self.health_assessor = health_assessor or CameraHealthAssessor()
        self.feature_extractor = RareCaseFeatureExtractor()
        self.classifier = classifier or RareCaseClassifier(model_path=model_path)
        
        self.frame_count = 0
        self.last_time = time.perf_counter()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        simulated_bbox: Optional[List[float]] = None
    ) -> PipelineFrameResult:
        t0 = time.perf_counter()
        if frame_id is None:
            self.frame_count += 1
            f_id = self.frame_count
        else:
            f_id = frame_id

        # 1. Camera Health
        health = self.health_assessor.assess(frame, frame_id=f_id)

        # 2. Pedestrian Detection (YOLOv8 + simulated fallback)
        det_result = self.detector.detect(frame, frame_id=f_id, simulated_bbox=simulated_bbox)

        # 3. ByteTrack Tracking
        tracks = self.tracker.update(det_result.detections, frame_id=f_id)

        # 4. Feature Fusion & Rare Case Classification
        predictions = []
        for track in tracks:
            feat_vec = self.feature_extractor.extract(track, health, frame_id=f_id)
            pred = self.classifier.predict(feat_vec)
            predictions.append(pred)

        # Determine overall frame decision & Explainable AI reasoning
        primary_explanation = ""
        if predictions:
            has_rare = any(p.prediction == "REAL_RARE_EVENT" for p in predictions)
            has_artifact = any(p.prediction == "CAMERA_ARTIFACT" for p in predictions)
            
            # If any track is an artifact (e.g. dirty lens / freeze) or lens is occluded:
            if has_artifact:
                primary_decision = "CAMERA_ARTIFACT"
                primary_conf = max(p.probabilities.get("CAMERA_ARTIFACT", 0.0) for p in predictions)
                art_pred = next(p for p in predictions if p.prediction == "CAMERA_ARTIFACT")
                primary_explanation = art_pred.explanation
            elif has_rare:
                primary_decision = "REAL_RARE_EVENT"
                primary_conf = max(p.probabilities.get("REAL_RARE_EVENT", 0.0) for p in predictions)
                rare_pred = next(p for p in predictions if p.prediction == "REAL_RARE_EVENT")
                primary_explanation = rare_pred.explanation
            else:
                primary_decision = "NORMAL"
                primary_conf = max(p.probabilities.get("NORMAL", 0.0) for p in predictions)
                primary_explanation = f"Camera hoạt động tốt ({health.overall_health*100:.0f}%), người đi bộ di chuyển bình thường theo quỹ đạo chuẩn."
        else:
            # When no tracks are active, check if camera has optical faults or artifacts
            if health.freeze_health < 0.25:
                primary_decision = "CAMERA_ARTIFACT"
                primary_conf = 0.96
                primary_explanation = "Cảm biến camera bị đứng hình (Freeze Delta = 0); xác định là lỗi phần cứng hoặc rớt gói tín hiệu mạng."
            elif health.occlusion_health < 0.45 or "PARTIAL_SMUDGE" in health.status_tags or "HEAVY_OCCLUSION" in health.status_tags:
                primary_decision = "CAMERA_ARTIFACT"
                primary_conf = round(max(0.75, 1.0 - health.occlusion_health), 2)
                primary_explanation = f"Phát hiện ống kính camera bị bám bẩn hoặc có vệt nước che khuất (Độ sạch ống kính: {health.occlusion_health*100:.0f}%). Vết bẩn quang học tạo nguy cơ báo động giả (False Alarm) cao; hệ thống đã loại bỏ báo động nhầm và khuyến nghị vệ sinh thấu kính."
            elif health.overall_health < 0.40:
                primary_decision = "CAMERA_FAULT"
                primary_conf = round(1.0 - health.overall_health, 2)
                primary_explanation = f"Chất lượng quang học của camera suy giảm nghiêm trọng ({health.overall_health*100:.0f}%); hệ thống cảnh báo kiểm tra lại nguồn sáng và ống kính."
            else:
                primary_decision = "NO_PEDESTRIAN"
                primary_conf = 1.0
                primary_explanation = f"Ống kính sạch ({health.occlusion_health*100:.0f}%), ánh sáng ổn định ({health.brightness_health*100:.0f}%). Không có người đi bộ trong vùng quan sát."

        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / max(1.0, latency_ms)

        # Render visual annotations on frame
        annotated = self._render_annotations(frame.copy(), health, tracks, predictions, fps)

        return PipelineFrameResult(
            frame_id=f_id,
            annotated_frame=annotated,
            camera_health=health,
            tracks=tracks,
            predictions=predictions,
            fps=fps,
            latency_ms=latency_ms,
            primary_decision=primary_decision,
            primary_confidence=primary_conf,
            primary_explanation=primary_explanation
        )

    def _render_annotations(
        self,
        frame: np.ndarray,
        health: CameraHealthResult,
        tracks: List[TrackInfo],
        predictions: List[RareCasePrediction],
        fps: float
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        pred_map = {p.track_id: p for p in predictions}

        # Color schemes (BGR):
        # NORMAL: Green (50, 205, 50)
        # REAL_RARE_EVENT: Vibrant Cyan/Blue (255, 180, 0)
        # CAMERA_ARTIFACT: Red/Crimson (0, 70, 240)
        COLOR_MAP = {
            "NORMAL": (50, 205, 50),
            "REAL_RARE_EVENT": (255, 180, 0),
            "CAMERA_ARTIFACT": (30, 60, 240)
        }

        LABEL_MAP = {
            "NORMAL": "BINH THUONG",
            "REAL_RARE_EVENT": "HIEM CO THAT",
            "CAMERA_ARTIFACT": "BAO DONG GIA",
            "TRACK": "DANG BAM VET"
        }

        # 1. Draw Track BBoxes & Trajectory
        for track in tracks:
            pred = pred_map.get(track.track_id)
            pred_key = pred.prediction if pred else "TRACK"
            pred_label = LABEL_MAP.get(pred_key, pred_key)
            color = COLOR_MAP.get(pred_key, (200, 200, 200))

            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory path
            if len(track.history) > 1:
                pts = np.array([[int(px), int(py)] for px, py in track.history], dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 2)

            # Header badge
            label_text = f"ID:{track.track_id} | {pred_label}"
            sub_text = f"Tin cay:{track.confidence:.2f} | Mot:{track.motion_score:.2f}"
            
            # Badge background
            cv2.rectangle(frame, (x1, max(0, y1 - 38)), (x1 + 195, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 4, max(12, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, sub_text, (x1 + 4, max(26, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (15, 15, 15), 1, cv2.LINE_AA)

        # 2. Camera Health HUD Overlay (Top-Left)
        hud_w, hud_h = 245, 112
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (12, 12), (12 + hud_w, 12 + hud_h), (80, 80, 90), 1)

        # Health color
        if health.overall_health > 0.70:
            h_color = (60, 220, 80)
        elif health.overall_health > 0.40:
            h_color = (0, 200, 255)
        else:
            h_color = (50, 50, 255)

        cv2.putText(frame, f"SUC KHOE CAMERA: {int(health.overall_health * 100)}%", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, h_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Do Net: {health.blur_health:.2f} | Sang: {health.brightness_health:.2f}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Ban Lens: {health.occlusion_health:.2f} | Dung: {health.freeze_health:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        
        tags_str = ", ".join(health.status_tags[:2])
        cv2.putText(frame, f"Trang thai: {tags_str}", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Toc do: {fps:.1f} FPS", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 255, 150), 1, cv2.LINE_AA)

        return frame

    def reset(self):
        self.frame_count = 0
        self.tracker.reset()
        self.health_assessor.reset()
