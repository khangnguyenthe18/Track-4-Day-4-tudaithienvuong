import os
import time
import asyncio
import threading
import cv2
import shutil
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, Response, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from ..data.synthetic_generator import SyntheticVideoGenerator, SCENARIOS
from ..pipeline.pipeline import RarePedestrianPipeline, PipelineFrameResult

app = FastAPI(title="Camera Health Aware Pedestrian Detection")

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
VIDEOS_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data", "synthetic_videos")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data", "uploaded_videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Shared state with thread safety
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.generator = SyntheticVideoGenerator(width=854, height=480, fps=20)
        self.pipeline = RarePedestrianPipeline()
        self.source_mode = "synthetic"  # "synthetic" or "uploaded"
        self.current_scenario = "normal_day"
        self.uploaded_video_name = ""
        self.uploaded_video_path = ""
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0
        self.total_frames = 120
        self.is_playing = True
        self.fps = 20.0
        self.latest_result: Optional[PipelineFrameResult] = None
        self.recent_events: List[Dict] = []
        self.last_decision = "NORMAL"

    def reset_pipeline(self):
        self.pipeline.reset()
        self.frame_idx = 0
        with self.lock:
            if self.video_cap and self.video_cap.isOpened():
                try:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass

    def switch_to_synthetic(self, scenario: str):
        with self.lock:
            if self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
                self.video_cap = None
            self.source_mode = "synthetic"
            self.current_scenario = scenario
            self.total_frames = 120
            self.fps = 20.0
        self.reset_pipeline()

    def open_uploaded_video(self, filepath: str, filename: str) -> bool:
        with self.lock:
            if self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
                self.video_cap = None
            
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return False

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            self.video_cap = cap
            self.uploaded_video_path = filepath
            self.uploaded_video_name = filename
            self.source_mode = "uploaded"
            self.total_frames = total if total > 0 else 100
            self.fps = min(max(float(fps) if fps and fps > 0 else 20.0, 10.0), 30.0)
            self.frame_idx = 0

        self.reset_pipeline()
        return True

state = AppState()

def log_event(event_type: str, message: str, level: str = "info"):
    timestamp = time.strftime("%H:%M:%S")
    event = {
        "timestamp": timestamp,
        "type": event_type,
        "message": message,
        "level": level
    }
    state.recent_events.insert(0, event)
    if len(state.recent_events) > 40:
        state.recent_events.pop()

def step_pipeline():
    res = None
    if state.source_mode == "uploaded":
        with state.lock:
            if state.video_cap is None or not state.video_cap.isOpened():
                return state.latest_result
            
            if not state.is_playing and state.latest_result is not None:
                return state.latest_result

            try:
                ret, raw_frame = state.video_cap.read()
                if not ret or raw_frame is None:
                    # Video reached end -> loop back to beginning
                    state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, raw_frame = state.video_cap.read()
                    if not ret or raw_frame is None:
                        return state.latest_result

                state.frame_idx = int(state.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            except Exception as e:
                print(f"[Video Cap Error] {e}")
                return state.latest_result

        # Downscale oversized frames for real-time inference & smooth streaming
        h, w = raw_frame.shape[:2]
        if w > 1024 or h > 576:
            scale = min(1024.0 / w, 576.0 / h)
            raw_frame = cv2.resize(raw_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # On uploaded real video: simulated_bbox is None, running true YOLOv8 inference
        try:
            res = state.pipeline.process_frame(
                raw_frame,
                frame_id=state.frame_idx,
                simulated_bbox=None
            )
            state.latest_result = res
        except Exception as e:
            print(f"[Pipeline Frame Error] {e}")
            return state.latest_result
    else:
        # Synthetic simulation mode
        if state.is_playing:
            state.frame_idx += 1
            if state.frame_idx > state.total_frames * 10:
                state.frame_idx = 1
        elif state.latest_result is not None:
            return state.latest_result

        sim_bbox = state.generator.get_scenario_bbox(
            state.current_scenario,
            state.frame_idx,
            state.total_frames
        )
        raw_frame = state.generator.generate_frame(
            state.current_scenario,
            state.frame_idx,
            total_frames=state.total_frames
        )
        try:
            res = state.pipeline.process_frame(
                raw_frame,
                frame_id=state.frame_idx,
                simulated_bbox=sim_bbox
            )
            state.latest_result = res
        except Exception as e:
            print(f"[Pipeline Sim Error] {e}")
            return state.latest_result

    # Log significant decision changes
    if res and res.primary_decision != state.last_decision and res.primary_decision not in ("NO_PEDESTRIAN", "INITIALIZING"):
        level = "danger" if res.primary_decision == "CAMERA_ARTIFACT" else ("info" if res.primary_decision == "NORMAL" else "warning")
        dec_vn = "BÌNH THƯỜNG" if res.primary_decision == "NORMAL" else ("TÌNH HUỐNG HIẾM CÓ THẬT" if res.primary_decision == "REAL_RARE_EVENT" else "BÁO ĐỘNG GIẢ / LỖI CAMERA")
        log_event("DECISION_ALERT", f"Kết luận: {dec_vn} (Độ tin cậy: {res.primary_confidence*100:.0f}%)", level)
        state.last_decision = res.primary_decision
    return res

async def background_pipeline_loop():
    while True:
        try:
            step_pipeline()
        except Exception as e:
            print(f"[Worker Error] {e}")
        await asyncio.sleep(1.0 / state.fps)

@app.on_event("startup")
async def startup_event():
    # Process initial frame immediately
    step_pipeline()
    asyncio.create_task(background_pipeline_loop())
    log_event("SYSTEM", "Đã khởi tạo thành công luồng AI (YOLOv8 + ByteTrack + Camera Health Engine)", "info")
    log_event("SCENARIO", f"Đang phát: {SCENARIOS[state.current_scenario]['title']}", "info")

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Đang tải giao diện điều khiển...</h1>")

@app.get("/api/scenarios")
def get_scenarios():
    return {
        "current": state.current_scenario,
        "source_mode": state.source_mode,
        "uploaded_video": state.uploaded_video_name,
        "scenarios": SCENARIOS
    }

class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/set-scenario")
def set_scenario(req: ScenarioRequest):
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail="Scenario not found")
    state.switch_to_synthetic(req.scenario)
    step_pipeline()
    info = SCENARIOS[req.scenario]
    log_event("SCENARIO_SWITCH", f"Đã chuyển sang tình huống mô phỏng: {info['title']}", "warning")
    return {"status": "ok", "scenario": req.scenario, "title": info["title"], "source_mode": state.source_mode}

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    allowed_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Định dạng video không được hỗ trợ ({ext}). Các định dạng hỗ trợ: {', '.join(allowed_exts)}")

    # Sanitize filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    if not safe_filename:
        safe_filename = f"video_{int(time.time())}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    success = state.open_uploaded_video(dest_path, safe_filename)
    if not success:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail="Không thể mở hoặc giải mã tệp video đã tải lên.")

    step_pipeline()
    log_event("VIDEO_UPLOAD", f"Đã tải lên và bắt đầu phân tích: {safe_filename} ({state.total_frames} khung hình)", "success")
    return {
        "status": "ok",
        "filename": safe_filename,
        "total_frames": state.total_frames,
        "fps": round(state.fps, 1),
        "source_mode": state.source_mode
    }

@app.get("/api/uploaded-videos")
def get_uploaded_videos():
    videos = []
    if os.path.exists(UPLOAD_DIR):
        for fname in sorted(os.listdir(UPLOAD_DIR)):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath):
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                videos.append({
                    "filename": fname,
                    "size_mb": size_mb,
                    "is_active": (state.source_mode == "uploaded" and state.uploaded_video_name == fname)
                })
    return {"videos": videos, "active": state.uploaded_video_name if state.source_mode == "uploaded" else None}

class SelectVideoRequest(BaseModel):
    filename: str

@app.post("/api/select-uploaded-video")
def select_uploaded_video(req: SelectVideoRequest):
    fpath = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Video không tồn tại.")
    
    success = state.open_uploaded_video(fpath, req.filename)
    if not success:
        raise HTTPException(status_code=400, detail="Không thể mở video đã chọn.")
    
    step_pipeline()
    log_event("VIDEO_SWITCH", f"Đã chuyển sang video thực tế: {req.filename}", "info")
    return {"status": "ok", "filename": req.filename, "total_frames": state.total_frames}

@app.post("/api/toggle-playback")
def toggle_playback():
    state.is_playing = not state.is_playing
    status_str = "Tiếp tục" if state.is_playing else "Tạm dừng"
    log_event("PLAYBACK", f"Đã {status_str} luồng video giám sát", "info")
    return {"is_playing": state.is_playing}

@app.post("/api/reset")
def reset_stream():
    state.reset_pipeline()
    step_pipeline()
    log_event("STREAM", "Đã đặt lại luồng video và bộ bám vết về Frame 0", "info")
    return {"status": "ok"}

@app.get("/api/telemetry")
def get_telemetry():
    res = state.latest_result
    if res is None:
        res = step_pipeline()
    
    data = res.to_dict() if res else {}
    data["scenario"] = state.current_scenario
    data["source_mode"] = state.source_mode
    data["video_name"] = state.uploaded_video_name if state.source_mode == "uploaded" else state.current_scenario
    data["total_frames"] = state.total_frames
    data["frame_idx"] = state.frame_idx
    data["is_playing"] = state.is_playing
    data["events"] = state.recent_events
    return data

async def generate_mjpeg_stream():
    while True:
        res = state.latest_result
        if res is not None and getattr(res, 'annotated_frame', None) is not None:
            ret, jpeg = cv2.imencode('.jpg', res.annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        await asyncio.sleep(1.0 / state.fps)

@app.get("/api/stream")
def video_stream():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/generate-all-videos")
def generate_all_videos():
    results = {}
    for sc_name in SCENARIOS.keys():
        out_path = os.path.join(VIDEOS_DIR, f"{sc_name}.mp4")
        saved = state.generator.generate_video(sc_name, out_path, num_frames=120)
        results[sc_name] = saved
        log_event("VIDEO_EXPORT", f"Đã xuất video mô phỏng: {sc_name}.mp4", "success")
    return {"status": "ok", "videos": results}
