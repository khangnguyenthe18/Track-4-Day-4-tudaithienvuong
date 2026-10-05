# Project Plan: Rare Pedestrian Detection with Camera Health Awareness

## 1. Tổng quan đề tài

### Tên đề tài
**Rare Pedestrian Detection with Camera Health Awareness**

### Tên tiếng Việt
**Phát hiện người đi bộ trong các tình huống hiếm có xét đến tình trạng sức khỏe của camera**

### Bài toán
Hệ thống camera giao thông thường sử dụng mô hình Computer Vision để phát hiện người đi bộ. Tuy nhiên, trong các điều kiện khó như:

- Ban đêm
- Mưa lớn
- Ảnh bị blur
- Camera mất nét
- Lens bị bẩn hoặc che khuất
- Camera bị freeze
- Nhiễu hoặc chất lượng hình ảnh thấp

mô hình object detection có thể tạo ra **false positive** hoặc **false negative**.

Mục tiêu của project là xây dựng một hệ thống không chỉ phát hiện pedestrian mà còn đánh giá **camera health**, từ đó phân biệt:

1. **Normal** — trạng thái bình thường
2. **Real Rare Event** — sự kiện người đi bộ hiếm nhưng có thật
3. **Camera Artifact / Sensor Fault** — bất thường xuất phát từ camera hoặc chất lượng hình ảnh

---

# 2. Case cụ thể

## Bối cảnh
Một camera cố định được đặt tại giao lộ để giám sát người đi bộ.

Trong phần lớn thời gian, dữ liệu thu được là bình thường. Tuy nhiên, thỉnh thoảng xuất hiện các tình huống hiếm:

- Người đi bộ băng qua đường vào ban đêm
- Người mặc áo mưa tối màu
- Người bị che khuất một phần
- Người xuất hiện trong điều kiện mưa hoặc blur
- Pedestrian detector phát hiện nhầm vùng ảnh do vệt nước hoặc lens bẩn

## Ví dụ 1: Người đi bộ thật

YOLO phát hiện người:

```text
Confidence: 0.72
```

Camera health:

```text
Blur score: 0.65
Brightness score: 0.55
Occlusion score: 0.88
Camera health: 0.74
```

Object tồn tại qua nhiều frame và có chuyển động hợp lý:

```text
Frame 101 -> Person detected
Frame 102 -> Person detected
Frame 103 -> Person detected
Frame 104 -> Person detected
Frame 105 -> Person detected
```

Tracking consistency:

```text
0.91
```

Kết luận:

```text
REAL RARE EVENT
Probability: 0.89
```

---

## Ví dụ 2: Camera artifact

Detector phát hiện:

```text
Person confidence: 0.78
```

Nhưng vùng detection nằm trên một vệt nước cố định trên lens.

Camera health:

```text
Blur score: 0.25
Occlusion score: 0.20
Camera health: 0.31
```

Object gần như đứng yên qua nhiều frame:

```text
Tracking consistency: 0.12
```

Kết luận:

```text
CAMERA ARTIFACT
Probability: 0.85
```

---

# 3. Mục tiêu project

## Mục tiêu chính

Xây dựng hệ thống:

```text
Video
   |
   v
Pedestrian Detection
   |
   +----------------------+
   |                      |
   v                      v
Object Tracking      Camera Health
   |                      |
   +-----------+----------+
               |
               v
          Feature Fusion
               |
               v
        Rare Case Model
               |
      +--------+---------+
      |        |         |
      v        v         v
   Normal   Rare Event  Artifact
```

## Câu hỏi nghiên cứu

1. Camera health có giúp giảm false positive trong pedestrian detection hay không?
2. Tracking có giúp phân biệt pedestrian thật với camera artifact hay không?
3. Mô hình health-aware có tốt hơn detector truyền thống trong điều kiện ảnh xấu hay không?
4. Camera health giảm đến mức nào thì detection trở nên không đáng tin cậy?
5. Loại lỗi camera nào ảnh hưởng lớn nhất đến hiệu năng pedestrian detection?

---

# 4. Kiến trúc hệ thống

## Module 1 — Video Input

Input:

- Video file
- CCTV footage
- Webcam
- Video stream

Output:

```python
frame
frame_id
timestamp
```

---

# 5. Module 2 — Pedestrian Detection

## Mục tiêu

Phát hiện pedestrian trong từng frame.

Có thể sử dụng:

- YOLOv8
- YOLO11
- RT-DETR
- Faster R-CNN

Khuyến nghị ban đầu:

```text
YOLO
```

vì dễ triển khai, tốc độ tốt và có pretrained model.

## Input

```text
Frame
```

## Output

```json
{
    "frame_id": 120,
    "bbox": [230, 120, 340, 480],
    "class": "person",
    "confidence": 0.82
}
```

## Metrics

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

---

# 6. Module 3 — Object Tracking

Detection trên một frame có thể chưa đủ đáng tin.

Tracking được sử dụng để kiểm tra object có tồn tại liên tục qua nhiều frame hay không.

Có thể sử dụng:

- ByteTrack
- DeepSORT
- BoT-SORT

Khuyến nghị:

```text
ByteTrack
```

## Ví dụ

```text
Frame 100:
ID 7 - Person - confidence 0.55

Frame 101:
ID 7 - Person - confidence 0.63

Frame 102:
ID 7 - Person - confidence 0.69

Frame 103:
ID 7 - Person - confidence 0.75
```

Ta có thể tạo một:

```text
Tracking Consistency Score
```

trong khoảng:

```text
0 -> 1
```

Ví dụ:

```text
0.90
```

có nghĩa object tồn tại ổn định qua nhiều frame.

---

# 7. Module 4 — Camera Health Assessment

Đây là một trong những module quan trọng nhất của project.

Ta định nghĩa:

```text
Camera Health Vector
```

như sau:

\[
H = [
H_{blur},
H_{brightness},
H_{occlusion},
H_{noise},
H_{freeze}
]
\]

Mỗi giá trị nằm trong:

\[
[0,1]
\]

Trong đó:

```text
1.0 = camera tốt
0.0 = camera rất kém
```

---

## 7.1 Blur Detection

Có thể sử dụng:

```text
Variance of Laplacian
```

Ví dụ:

```python
blur_score = variance_of_laplacian(frame)
```

Nếu variance thấp:

```text
ảnh bị blur nhiều
```

---

## 7.2 Low-Light Detection

Phân tích:

- Mean brightness
- Histogram
- Contrast

Ví dụ:

```text
Brightness = mean grayscale intensity
```

Nếu:

```text
Brightness < threshold
```

thì đánh dấu:

```text
LOW LIGHT
```

---

## 7.3 Occlusion / Dirty Lens Detection

Các lỗi:

- Nước trên lens
- Bùn
- Vật cản
- Một vùng camera bị che

Có thể phát hiện dựa trên:

- Static region
- Temporal consistency
- Texture change
- Segmentation
- CNN classifier

---

## 7.4 Frozen Frame Detection

So sánh:

\[
D_t = |I_t - I_{t-1}|
\]

Nếu:

\[
D_t \approx 0
\]

trong nhiều frame:

```text
Camera có khả năng bị freeze
```

---

## 7.5 Camera Health Score

Ví dụ:

\[
H =
w_1H_{blur}
+
w_2H_{brightness}
+
w_3H_{occlusion}
+
w_4H_{noise}
+
w_5H_{freeze}
\]

Với:

\[
\sum w_i = 1
\]

Ví dụ:

```text
Blur Health       = 0.30
Brightness Health = 0.60
Occlusion Health  = 0.90
Noise Health      = 0.80
Freeze Health     = 1.00
```

Camera Health cuối:

```text
0.68
```

---

# 8. Feature Fusion

Mỗi detection tạo ra một feature vector:

\[
X =
[
DetectionConfidence,
TrackingScore,
MotionScore,
CameraHealth,
BlurScore,
BrightnessScore,
OcclusionScore
]
\]

Ví dụ:

```python
X = [
    0.78,
    0.91,
    0.82,
    0.74,
    0.65,
    0.55,
    0.88
]
```

---

# 9. Rare Case Classification

Model nhận:

```text
Detection features
+
Tracking features
+
Camera health
```

và dự đoán:

\[
Y \in
\{
Normal,
RealRareEvent,
CameraArtifact
\}
\]

## Model baseline

Có thể thử:

- Logistic Regression
- Random Forest
- XGBoost
- MLP

Khuyến nghị:

```text
Random Forest / XGBoost
```

cho baseline.

Lý do:

- Dễ train
- Dễ explain
- Không cần dataset quá lớn
- Feature importance rõ ràng

---

# 10. Baseline và Proposed Model

## Baseline

```text
YOLO
```

Decision chỉ dựa trên:

```text
Detection Confidence
```

---

## Model 2

```text
YOLO + Tracking
```

Feature:

```text
Detection Confidence
Tracking Consistency
```

---

## Proposed Model

```text
YOLO
+
Tracking
+
Camera Health
+
Rare Case Classifier
```

So sánh ba mô hình này sẽ giúp chứng minh giá trị của sensor-health awareness.

---

# 11. Dataset

## Dataset gốc

Có thể sử dụng các dataset pedestrian/video có sẵn như:

- MOT17
- MOT20
- CrowdHuman
- CityPersons
- BDD100K

## Dataset mở rộng

Từ video gốc, nhóm chủ động tạo các phiên bản degraded:

### Blur

```python
GaussianBlur
MotionBlur
```

### Low Light

```python
brightness * 0.4
```

### Noise

```text
Gaussian Noise
Salt & Pepper Noise
```

### Occlusion

```text
Random mask
Dirty lens overlay
Water-drop overlay
```

### Frozen Frames

Lặp lại cùng một frame nhiều lần.

---

# 12. Dataset Label

Mỗi sample có thể lưu:

```json
{
    "frame_id": 100,
    "pedestrian": true,
    "rare_event": true,
    "camera_condition": "blur",
    "camera_health": 0.45,
    "label": "REAL_RARE_EVENT"
}
```

Hoặc:

```json
{
    "frame_id": 220,
    "pedestrian": false,
    "camera_condition": "dirty_lens",
    "camera_health": 0.25,
    "label": "CAMERA_ARTIFACT"
}
```

---

# 13. Phân chia công việc cho 4 thành viên

---

# Thành viên 1 — Pedestrian Detection Engineer

## Trách nhiệm chính

Xây dựng module pedestrian detection.

## Công việc

### Research

- Nghiên cứu YOLO / DETR
- Chọn model phù hợp
- Nghiên cứu dataset pedestrian

### Development

- Chuẩn bị dataset
- Preprocessing
- Train / Fine-tune YOLO
- Inference video
- Export detection
- Tối ưu threshold

### Evaluation

Đo:

- Precision
- Recall
- mAP
- FPS

## Deliverables

```text
src/detection/
    detector.py
    train.py
    inference.py
    evaluate.py
```

Model:

```text
weights/
    pedestrian_detector.pt
```

Report:

```text
reports/detection_results.md
```

---

# Thành viên 2 — Camera Health Engineer

## Trách nhiệm chính

Thiết kế module đánh giá chất lượng camera.

Đây là người phụ trách trực tiếp phần:

```text
sensor health
```

## Công việc

### Blur Detection

Implement:

```text
Laplacian Variance
```

### Low Light Detection

Implement:

```text
Brightness Score
Contrast Score
```

### Occlusion Detection

Phát hiện:

- lens bị che
- dirty lens
- water droplets
- static artifact

### Freeze Detection

Phân tích difference giữa các frame.

### Health Score

Thiết kế:

```text
Camera Health Score
```

Ví dụ:

```python
camera_health = (
    0.30 * blur_health +
    0.20 * brightness_health +
    0.25 * occlusion_health +
    0.15 * noise_health +
    0.10 * freeze_health
)
```

## Deliverables

```text
src/camera_health/
    blur.py
    brightness.py
    occlusion.py
    freeze.py
    health_score.py
```

Report:

```text
reports/camera_health_analysis.md
```

---

# Thành viên 3 — Tracking & Rare Case Mining Engineer

## Trách nhiệm chính

Kết nối detection theo thời gian và xây dựng rare-case classifier.

## Công việc

### Object Tracking

Implement:

```text
ByteTrack / DeepSORT
```

Output:

```text
Track ID
Bounding box
Trajectory
Tracking consistency
```

### Feature Engineering

Tạo:

```text
tracking_score
motion_score
trajectory_score
detection_persistence
```

### Feature Fusion

Kết hợp:

```text
YOLO features
Tracking features
Camera health features
```

### Rare Case Model

Train:

```text
Random Forest
XGBoost
MLP
```

Classification:

```text
Normal
Real Rare Event
Camera Artifact
```

### Explainability

Phân tích:

```text
Feature Importance
```

để trả lời:

```text
Camera health có thực sự giúp model không?
```

## Deliverables

```text
src/tracking/
    tracker.py
    trajectory.py

src/rare_case/
    features.py
    classifier.py
    train.py
    evaluate.py
```

Models:

```text
weights/
    rare_case_model.pkl
```

---

# Thành viên 4 — Dataset, Integration & System Engineer

## Trách nhiệm chính

Đảm bảo tất cả module chạy được thành một hệ thống hoàn chỉnh.

## Công việc

### Dataset Management

- Thu thập video
- Chia train / validation / test
- Quản lý annotation
- Generate degraded dataset

Implement:

```text
Blur Generator
Low-light Generator
Noise Generator
Occlusion Generator
Frozen-frame Generator
```

### System Integration

Kết nối:

```text
Detector
    ↓
Tracker
    ↓
Camera Health
    ↓
Rare Case Classifier
```

### Dashboard

Xây UI bằng:

```text
Streamlit
```

hoặc:

```text
Gradio
```

Dashboard hiển thị:

```text
Video
Bounding boxes
Track ID
Camera Health
Rare Event Probability
Final Decision
```

### Experiment Management

Tổng hợp kết quả từ tất cả thành viên.

## Deliverables

```text
src/data/
    degradation.py
    dataset.py

src/dashboard/
    app.py

src/pipeline/
    main.py
```

---

# 14. Phân quyền GitHub

Có thể chia branch như sau:

```text
main

dev

feature/detection
feature/camera-health
feature/tracking-rare-case
feature/dataset-integration
```

Tương ứng:

| Thành viên | Branch |
|---|---|
| Member 1 | `feature/detection` |
| Member 2 | `feature/camera-health` |
| Member 3 | `feature/tracking-rare-case` |
| Member 4 | `feature/dataset-integration` |

Mọi code phải merge qua:

```text
Pull Request
```

Không push trực tiếp lên:

```text
main
```

---

# 15. Interface giữa các module

Đây là phần rất quan trọng để tránh mỗi người code một kiểu.

## Detector Output

```python
Detection = {
    "frame_id": int,
    "bbox": [x1, y1, x2, y2],
    "confidence": float,
    "class_id": int
}
```

---

## Tracker Output

```python
Track = {
    "frame_id": int,
    "track_id": int,
    "bbox": [x1, y1, x2, y2],
    "confidence": float,
    "tracking_score": float
}
```

---

## Camera Health Output

```python
CameraHealth = {
    "frame_id": int,
    "blur_health": float,
    "brightness_health": float,
    "occlusion_health": float,
    "noise_health": float,
    "freeze_health": float,
    "overall_health": float
}
```

---

## Rare Case Model Input

```python
RareCaseInput = {
    "detection_confidence": float,
    "tracking_score": float,
    "motion_score": float,
    "camera_health": float,
    "blur_health": float,
    "brightness_health": float,
    "occlusion_health": float
}
```

---

## Final Output

```python
Prediction = {
    "frame_id": int,
    "track_id": int,
    "prediction": str,
    "probability": float,
    "camera_health": float
}
```

Ví dụ:

```json
{
    "frame_id": 150,
    "track_id": 8,
    "prediction": "REAL_RARE_EVENT",
    "probability": 0.91,
    "camera_health": 0.72
}
```

---

# 16. Timeline đề xuất — 10 tuần

## Tuần 1 — Problem Definition

Cả nhóm:

- Chốt scope
- Nghiên cứu paper
- Chọn dataset
- Chọn model
- Setup repository

Output:

```text
Project proposal
Architecture
Dataset list
```

---

## Tuần 2 — Dataset

### Member 1

Chuẩn bị pedestrian dataset.

### Member 2

Nghiên cứu camera degradation.

### Member 3

Setup tracking.

### Member 4

Xây dataset pipeline.

---

## Tuần 3 — Baseline

### Member 1

Chạy YOLO baseline.

### Member 2

Blur + brightness detection.

### Member 3

ByteTrack baseline.

### Member 4

Generate degraded images.

Output:

```text
YOLO baseline results
```

---

## Tuần 4 — Camera Health

### Member 2

Hoàn thiện:

```text
Blur
Low Light
Freeze
Noise
```

### Member 4

Chuẩn hóa dataset degradation.

---

## Tuần 5 — Tracking & Feature Extraction

### Member 3

Hoàn thiện:

```text
Tracking Score
Motion Score
Trajectory Score
```

### Member 1

Tối ưu detector.

---

## Tuần 6 — Rare Case Model

Member 3 train:

```text
Random Forest
XGBoost
```

Input:

```text
Detection
Tracking
Camera Health
```

---

## Tuần 7 — Integration

Member 4 tích hợp:

```text
YOLO
+
Tracker
+
Camera Health
+
Rare Case Model
```

---

## Tuần 8 — Experiments

Chạy ba hệ thống:

### Experiment A

```text
YOLO
```

### Experiment B

```text
YOLO + Tracking
```

### Experiment C

```text
YOLO + Tracking + Camera Health
```

---

## Tuần 9 — Evaluation

Đánh giá trên:

```text
Normal
Blur
Low-light
Noise
Occlusion
Frozen
```

Vẽ:

- Precision
- Recall
- F1
- PR Curve
- Confusion Matrix
- Camera Health vs Detection Accuracy

---

## Tuần 10 — Demo & Report

Hoàn thiện:

- Dashboard
- Demo video
- Report
- Slides
- GitHub
- README

---

# 17. Experiment Design

## Experiment 1 — Normal Condition

```text
Camera Health ≈ 1.0
```

Đánh giá detector bình thường.

---

## Experiment 2 — Blur

Các mức blur:

```text
10%
30%
50%
70%
90%
```

Đo:

```text
Detection Recall
False Positive
Rare Case Accuracy
```

---

## Experiment 3 — Low Light

Brightness:

```text
100%
80%
60%
40%
20%
```

---

## Experiment 4 — Occlusion

Occlusion:

```text
0%
20%
40%
60%
```

---

## Experiment 5 — Combined Degradation

Ví dụ:

```text
Low Light
+
Blur
+
Noise
```

Đây là experiment gần thực tế nhất.

---

# 18. Evaluation Metrics

Không nên chỉ dùng accuracy.

## Pedestrian Detection

```text
Precision
Recall
mAP
```

---

## Rare Event Classification

```text
Precision
Recall
F1 Score
PR-AUC
```

---

## Camera Artifact Detection

Quan trọng nhất:

```text
False Positive Rate
```

Mục tiêu:

```text
Health-aware model
```

phải giảm:

```text
False pedestrian detection
```

khi camera health thấp.

---

# 19. Kết quả kỳ vọng

Ví dụ kết quả cuối:

| Model | Precision | Recall | F1 | False Positive |
|---|---:|---:|---:|---:|
| YOLO | 0.78 | 0.81 | 0.79 | 18% |
| YOLO + Tracking | 0.84 | 0.83 | 0.83 | 11% |
| YOLO + Tracking + Camera Health | **0.91** | **0.86** | **0.88** | **5%** |

Mục tiêu nghiên cứu là chứng minh:

```text
Camera Health Awareness
```

giúp giảm false positive và tăng độ tin cậy của hệ thống.

---

# 20. Demo cuối cùng

Dashboard:

```text
---------------------------------------

Camera Health: 78 / 100

Status:
LOW LIGHT

---------------------------------------

Person ID: 12

YOLO confidence:
0.82

Tracking consistency:
0.91

Rare event probability:
0.89

---------------------------------------

FINAL DECISION

REAL RARE EVENT

---------------------------------------
```

Trường hợp camera lỗi:

```text
---------------------------------------

Camera Health: 24 / 100

Problems:

- Heavy Blur
- Lens Occlusion

---------------------------------------

YOLO confidence:
0.75

Tracking consistency:
0.13

---------------------------------------

FINAL DECISION

CAMERA ARTIFACT

---------------------------------------
```

---

# 21. Folder Structure

```text
rare-case-camera-health/

├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── degraded/
│   └── annotations/
│
├── src/
│   │
│   ├── detection/
│   │   ├── detector.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── trajectory.py
│   │
│   ├── camera_health/
│   │   ├── blur.py
│   │   ├── brightness.py
│   │   ├── noise.py
│   │   ├── occlusion.py
│   │   ├── freeze.py
│   │   └── health_score.py
│   │
│   ├── rare_case/
│   │   ├── features.py
│   │   ├── classifier.py
│   │   └── evaluate.py
│   │
│   ├── pipeline/
│   │   └── main.py
│   │
│   └── dashboard/
│       └── app.py
│
├── weights/
│
├── notebooks/
│
├── experiments/
│
├── reports/
│
├── requirements.txt
│
└── config.yaml
```

---

# 22. Công nghệ đề xuất

## Language

```text
Python
```

## Computer Vision

```text
OpenCV
Ultralytics YOLO
```

## Tracking

```text
ByteTrack
```

## Machine Learning

```text
Scikit-learn
XGBoost
```

## Deep Learning

```text
PyTorch
```

## Dashboard

```text
Streamlit
```

## Experiment Tracking

Có thể dùng:

```text
Weights & Biases
TensorBoard
```

---

# 23. MVP

Nếu thời gian hạn chế, MVP chỉ cần:

```text
YOLO
+
ByteTrack
+
Blur Detection
+
Brightness Detection
+
Random Forest
```

Classification:

```text
Normal
Real Rare Event
Camera Artifact
```

Đây là scope đủ để hoàn thành project.

---

# 24. Phiên bản nâng cao

Sau khi MVP chạy ổn định, có thể thêm:

### Deep Camera Health

Thay rule-based bằng CNN.

### Temporal Model

Sử dụng:

```text
LSTM
Transformer
Temporal CNN
```

### Explainable AI

Sử dụng:

```text
SHAP
```

để giải thích:

```text
Tại sao hệ thống cho rằng đây là camera artifact?
```

### Multi-Camera

So sánh detection giữa nhiều camera.

---

# 25. Research Contribution

Project có thể đóng góp:

1. Xây dựng camera health score cho pedestrian detection.
2. Kết hợp camera health với object detection và tracking.
3. Phân biệt pedestrian thật với camera artifact.
4. Đánh giá robustness của detector dưới camera degradation.
5. Chứng minh health-aware model giảm false positive.

---

# 26. Tiêu chí hoàn thành project

Project được coi là hoàn thành khi:

- [ ] Có pedestrian detector hoạt động
- [ ] Có object tracking
- [ ] Có ít nhất 3 loại camera degradation
- [ ] Có camera health score
- [ ] Có rare-case classifier
- [ ] Có baseline comparison
- [ ] Có evaluation metrics
- [ ] Có dashboard/demo
- [ ] Có report
- [ ] Có GitHub repository với code chạy được

---

# 27. Phân công tóm tắt

| Thành viên | Vai trò | Module chính |
|---|---|---|
| Member 1 | Detection Engineer | YOLO, pedestrian detection |
| Member 2 | Camera Health Engineer | Blur, low-light, occlusion, freeze |
| Member 3 | ML / Tracking Engineer | ByteTrack, feature fusion, rare-case classifier |
| Member 4 | Data / Integration Engineer | Dataset, degradation, pipeline, dashboard |

---

# 28. Mục tiêu cuối cùng

Hệ thống không chỉ trả lời:

> **Có người trong ảnh hay không?**

Mà phải trả lời:

> **Có khả năng cao đây là người thật hay chỉ là detection bất thường do camera đang gặp vấn đề?**

Biểu diễn:

\[
P(\text{Rare Event} \mid \text{Video}, \text{Camera Health})
\]

thay vì:

\[
P(\text{Pedestrian} \mid \text{Image})
\]

Đây chính là trọng tâm của đề tài:

> **Rare-case mining with sensor health trong lĩnh vực Computer Vision.**
