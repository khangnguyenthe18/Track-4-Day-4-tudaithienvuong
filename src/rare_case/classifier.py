from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import joblib
import numpy as np

from .features import RareCaseFeatureVector, FEATURE_NAMES

CLASSES = ["NORMAL", "REAL_RARE_EVENT", "CAMERA_ARTIFACT"]

@dataclass
class RareCasePrediction:
    track_id: int
    frame_id: int
    prediction: str # 'NORMAL', 'REAL_RARE_EVENT', or 'CAMERA_ARTIFACT'
    confidence: float
    probabilities: Dict[str, float]
    camera_health: float
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "camera_health": round(self.camera_health, 3),
            "explanation": self.explanation
        }

class RareCaseClassifier:
    """
    Classifies pedestrian detections into:
    1. NORMAL: Valid pedestrian under normal camera condition
    2. REAL_RARE_EVENT: Genuine rare pedestrian (e.g. night, rain, partial blur) with confirmed motion
    3. CAMERA_ARTIFACT: False trigger from dirty lens, smudge, camera blur, or freeze
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.classes = CLASSES

        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"[RareCaseClassifier] Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"[RareCaseClassifier] Failed to load model ({e}), using expert rule engine.")

    def predict(self, feature_vector: RareCaseFeatureVector) -> RareCasePrediction:
        feats = feature_vector.features
        x = feature_vector.raw_array.reshape(1, -1)

        # 0. Check for Dirty Lens / Lens Smudge Artifact
        if feats.get("occlusion_health", 1.0) < 0.45 and feats.get("motion_score", 0.0) < 0.25:
            return RareCasePrediction(
                track_id=feature_vector.track_id,
                frame_id=feature_vector.frame_id,
                prediction="CAMERA_ARTIFACT",
                confidence=0.96,
                probabilities={"NORMAL": 0.01, "REAL_RARE_EVENT": 0.03, "CAMERA_ARTIFACT": 0.96},
                camera_health=feats["camera_health"],
                explanation=f"Phát hiện vết bẩn hoặc vệt nước đọng trên ống kính (Occlusion: {feats['occlusion_health']*100:.0f}%) kết hợp đối tượng đứng yên tuyệt đối ({feats['motion_score']:.2f}px); hệ thống khẳng định đây là Báo Động Giả (False Alarm) do thấu kính và đã vô hiệu hóa cảnh báo."
            )

        # 1. Deterministic Sensor Freeze Check
        if feats.get("freeze_health", 1.0) < 0.25:
            return RareCasePrediction(
                track_id=feature_vector.track_id,
                frame_id=feature_vector.frame_id,
                prediction="CAMERA_ARTIFACT",
                confidence=0.96,
                probabilities={"NORMAL": 0.01, "REAL_RARE_EVENT": 0.03, "CAMERA_ARTIFACT": 0.96},
                camera_health=feats["camera_health"],
                explanation="Cảm biến camera bị đứng hình (Freeze Delta = 0); hệ thống xác định đây là lỗi phần cứng cảm biến hoặc nghẽn luồng video."
            )

        # 1. If trained ML model is available
        if self.model is not None:
            try:
                probs = self.model.predict_proba(x)[0]
                cls_idx = int(np.argmax(probs))
                pred_label = self.classes[cls_idx]
                prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}
                explanation = self._explain_decision(feats, pred_label)
                return RareCasePrediction(
                    track_id=feature_vector.track_id,
                    frame_id=feature_vector.frame_id,
                    prediction=pred_label,
                    confidence=float(probs[cls_idx]),
                    probabilities=prob_dict,
                    camera_health=feats["camera_health"],
                    explanation=explanation
                )
            except Exception as e:
                pass

        # Expert rule engine fallback & explainable decision logic
        return self._rule_based_predict(feature_vector)

    def _rule_based_predict(self, feature_vector: RareCaseFeatureVector) -> RareCasePrediction:
        f = feature_vector.features
        cam_h = f["camera_health"]
        tracking_s = f["tracking_score"]
        motion_s = f["motion_score"]
        det_conf = f["detection_confidence"]
        freeze_h = f["freeze_health"]
        occl_h = f["occlusion_health"]

        # 1. Camera Artifact Criteria:
        if freeze_h < 0.2:
            pred = "CAMERA_ARTIFACT"
            p_artifact = 0.95
            p_rare = 0.03
            p_norm = 0.02
            explanation = "Phát hiện cảm biến camera bị đứng hình (frame freeze delta = 0); xác định là lỗi phần cứng/truyền dẫn."
        elif occl_h < 0.35 and motion_s < 0.18:
            pred = "CAMERA_ARTIFACT"
            p_artifact = 0.88
            p_rare = 0.08
            p_norm = 0.04
            explanation = f"Vết bẩn/nước đọng trên kính kèm độ dịch chuyển gần như bằng 0 ({motion_s:.2f}px); xác định báo động giả."
        elif cam_h < 0.40 and motion_s < 0.15:
            pred = "CAMERA_ARTIFACT"
            p_artifact = 0.84
            p_rare = 0.11
            p_norm = 0.05
            explanation = f"Chất lượng camera suy giảm nghiêm trọng ({cam_h*100:.0f}%) và vật thể đứng yên; báo động giả."
        # 2. Real Rare Event Criteria:
        elif cam_h < 0.70 and tracking_s >= 0.50 and motion_s >= 0.15:
            pred = "REAL_RARE_EVENT"
            p_rare = 0.86
            p_norm = 0.09
            p_artifact = 0.05
            explanation = f"Môi trường camera suy giảm ({cam_h*100:.0f}%), nhưng bám vết ByteTrack ổn định ({tracking_s*100:.0f}%) và có chuyển động thật ({motion_s:.2f}); xác nhận người đi bộ thật trong điều kiện hiếm."
        # 3. Normal Pedestrian:
        elif cam_h >= 0.65:
            pred = "NORMAL"
            p_norm = 0.89
            p_rare = 0.07
            p_artifact = 0.04
            explanation = f"Camera hoạt động chuẩn ({cam_h*100:.0f}%), phát hiện người đi bộ bình thường."
        else:
            if motion_s > 0.25 and tracking_s > 0.5:
                pred = "REAL_RARE_EVENT"
                p_rare = 0.65
                p_norm = 0.20
                p_artifact = 0.15
                explanation = "Điều kiện trung gian; đối tượng có vận tốc di chuyển gợi ý sự kiện thật."
            else:
                pred = "CAMERA_ARTIFACT"
                p_artifact = 0.62
                p_rare = 0.23
                p_norm = 0.15
                explanation = "Nhận diện không chắc chắn trong điều kiện camera suy giảm; cảnh báo lỗi cảm biến."

        prob_dict = {
            "NORMAL": float(p_norm),
            "REAL_RARE_EVENT": float(p_rare),
            "CAMERA_ARTIFACT": float(p_artifact)
        }

        return RareCasePrediction(
            track_id=feature_vector.track_id,
            frame_id=feature_vector.frame_id,
            prediction=pred,
            confidence=float(prob_dict[pred]),
            probabilities=prob_dict,
            camera_health=cam_h,
            explanation=explanation
        )

    def _explain_decision(self, f: Dict[str, float], pred: str) -> str:
        cam_h = f.get("camera_health", 0.0)
        motion_s = f.get("motion_score", 0.0)
        track_s = f.get("tracking_score", 0.0)
        freeze_h = f.get("freeze_health", 1.0)
        blur_h = f.get("blur_health", 1.0)
        occl_h = f.get("occlusion_health", 1.0)
        
        if occl_h < 0.45 and motion_s < 0.25:
            return f"Phát hiện vết bẩn hoặc vệt nước đọng trên ống kính (Occlusion: {occl_h*100:.0f}%) kết hợp đối tượng đứng yên tuyệt đối ({motion_s:.2f}px); hệ thống khẳng định đây là Báo Động Giả (False Alarm) do thấu kính và đã vô hiệu hóa cảnh báo."
        elif freeze_h < 0.25:
            return "Cảm biến camera bị đứng hình (Freeze Delta = 0); xác định là lỗi phần cứng hoặc rớt gói tín hiệu mạng."
        elif pred == "CAMERA_ARTIFACT":
            if occl_h < 0.45:
                return f"Ống kính bị bám bẩn (Độ sạch: {occl_h*100:.0f}%) kết hợp vật thể đứng yên ({motion_s:.2f}px); hệ thống khẳng định là Báo Động Giả."
            if blur_h < 0.25:
                return f"Ảnh camera bị mờ nghiêm trọng (Độ nét: {blur_h*100:.0f}%) kết hợp vật thể đứng yên ({motion_s:.2f}px); hệ thống loại bỏ báo động giả."
            return f"Sức khỏe camera thấp ({cam_h*100:.0f}%) và chuyển động gần như bằng 0 ({motion_s:.2f}px); hệ thống xác định đây là báo động giả do vết bẩn trên ống kính."
        elif pred == "REAL_RARE_EVENT":
            return f"Môi trường camera suy giảm ({cam_h*100:.0f}%), nhưng thuật toán tracking ByteTrack duy trì liên tục ({track_s*100:.0f}%) với chuyển động thật ({motion_s:.2f}px); xác nhận người đi bộ có thật trong tình huống hiếm."
        else:
            return f"Camera hoạt động tốt ({cam_h*100:.0f}%), người đi bộ di chuyển bình thường theo vạch sang đường."
