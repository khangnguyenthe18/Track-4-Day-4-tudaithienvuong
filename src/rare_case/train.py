import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from .features import FEATURE_NAMES
from .classifier import CLASSES

def generate_synthetic_feature_dataset(n_samples_per_class: int = 400, random_seed: int = 42):
    """
    Generate synthetic feature vectors for the 3 classes based on domain physics:
    - NORMAL: Healthy camera, good detection, moving pedestrian
    - REAL_RARE_EVENT: Degraded camera (low light, rain, blur), but true pedestrian with good motion
    - CAMERA_ARTIFACT: Degraded camera (smudge, freeze, blur), static or erratic false positives
    """
    rng = np.random.default_rng(random_seed)
    X_list = []
    y_list = []

    # 1. NORMAL (class index 0)
    for _ in range(n_samples_per_class):
        det_conf = rng.uniform(0.60, 0.95)
        track_score = rng.uniform(0.70, 1.0)
        motion_score = rng.uniform(0.35, 1.0)
        blur_h = rng.uniform(0.75, 1.0)
        bright_h = rng.uniform(0.70, 1.0)
        occl_h = rng.uniform(0.85, 1.0)
        noise_h = rng.uniform(0.80, 1.0)
        freeze_h = 1.0
        cam_h = (0.3*blur_h + 0.2*bright_h + 0.25*occl_h + 0.15*noise_h + 0.1*freeze_h)

        row = [det_conf, track_score, motion_score, cam_h, blur_h, bright_h, occl_h, noise_h, freeze_h]
        X_list.append(row)
        y_list.append(0)

    # 2. REAL_RARE_EVENT (class index 1)
    # Night, rain, blur, but persistent pedestrian with active motion
    for _ in range(n_samples_per_class):
        det_conf = rng.uniform(0.40, 0.85)
        track_score = rng.uniform(0.65, 0.98)
        motion_score = rng.uniform(0.30, 0.90) # real pedestrian walking
        # At least one degradation is active
        deg_type = rng.choice(["low_light", "blur", "rain_noise"])
        if deg_type == "low_light":
            bright_h = rng.uniform(0.15, 0.50)
            blur_h = rng.uniform(0.60, 0.90)
            noise_h = rng.uniform(0.50, 0.80)
            occl_h = rng.uniform(0.80, 1.0)
        elif deg_type == "blur":
            blur_h = rng.uniform(0.15, 0.45)
            bright_h = rng.uniform(0.60, 0.90)
            noise_h = rng.uniform(0.70, 0.95)
            occl_h = rng.uniform(0.80, 1.0)
        else: # rain/noise
            noise_h = rng.uniform(0.15, 0.45)
            bright_h = rng.uniform(0.40, 0.70)
            blur_h = rng.uniform(0.50, 0.80)
            occl_h = rng.uniform(0.60, 0.85)
        freeze_h = 1.0
        cam_h = (0.3*blur_h + 0.2*bright_h + 0.25*occl_h + 0.15*noise_h + 0.1*freeze_h)

        row = [det_conf, track_score, motion_score, cam_h, blur_h, bright_h, occl_h, noise_h, freeze_h]
        X_list.append(row)
        y_list.append(1)

    # 3. CAMERA_ARTIFACT (class index 2)
    # Lens smudge, water drop, static freeze, or heavy noise inducing false detection
    for _ in range(n_samples_per_class):
        det_conf = rng.uniform(0.35, 0.80)
        track_score = rng.uniform(0.10, 0.85)
        motion_score = rng.uniform(0.00, 0.15) # Static artifact stays stationary!
        artifact_type = rng.choice(["dirty_lens", "frozen", "severe_blur"])
        if artifact_type == "dirty_lens":
            occl_h = rng.uniform(0.10, 0.35)
            blur_h = rng.uniform(0.40, 0.80)
            bright_h = rng.uniform(0.50, 0.80)
            noise_h = rng.uniform(0.60, 0.90)
            freeze_h = 1.0
        elif artifact_type == "frozen":
            freeze_h = rng.uniform(0.0, 0.1)
            occl_h = rng.uniform(0.60, 0.90)
            blur_h = rng.uniform(0.60, 0.90)
            bright_h = rng.uniform(0.60, 0.90)
            noise_h = 1.0
        else: # severe blur causing false ghost
            blur_h = rng.uniform(0.05, 0.25)
            occl_h = rng.uniform(0.50, 0.80)
            bright_h = rng.uniform(0.40, 0.80)
            noise_h = rng.uniform(0.40, 0.80)
            freeze_h = 1.0

        cam_h = (0.3*blur_h + 0.2*bright_h + 0.25*occl_h + 0.15*noise_h + 0.1*freeze_h)
        row = [det_conf, track_score, motion_score, cam_h, blur_h, bright_h, occl_h, noise_h, freeze_h]
        X_list.append(row)
        y_list.append(2)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y

def train_and_save_model(output_path: str = "weights/rare_case_model.pkl") -> RandomForestClassifier:
    print("[Training] Generating synthetic feature training set...")
    X, y = generate_synthetic_feature_dataset(n_samples_per_class=600)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print(f"[Training] Training Random Forest on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Training] Test Accuracy: {acc * 100:.2f}%")
    print("[Training] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    print("[Training] Feature Importances:")
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"  - {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"[Training] Model successfully saved to: {output_path}")
    return clf

if __name__ == "__main__":
    train_and_save_model()
