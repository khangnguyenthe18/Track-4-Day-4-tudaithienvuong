from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math
import numpy as np

@dataclass
class TrackInfo:
    track_id: int
    bbox: List[float] # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list) # centers
    frame_history: List[int] = field(default_factory=list)
    persistence: int = 1
    tracking_consistency: float = 1.0
    motion_score: float = 0.0 # normalized motion speed/displacement
    total_displacement: float = 0.0
    is_stationary: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

class TrajectoryManager:
    """
    Manages track historical trajectories, calculates motion displacement,
    tracking consistency, and flags static artifacts.
    """
    def __init__(self, window_size: int = 25, stationary_pixel_thresh: float = 8.0):
        self.window_size = window_size
        self.stationary_pixel_thresh = stationary_pixel_thresh
        self.tracks: Dict[int, TrackInfo] = {}
        self.current_frame = 0

    def update_track(
        self,
        track_id: int,
        bbox: List[float],
        confidence: float,
        frame_id: int,
        class_id: int = 0
    ) -> TrackInfo:
        self.current_frame = frame_id
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        if track_id not in self.tracks:
            track = TrackInfo(
                track_id=track_id,
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                history=[(cx, cy)],
                frame_history=[frame_id],
                persistence=1,
                tracking_consistency=1.0,
                motion_score=0.1,
                total_displacement=0.0,
                is_stationary=False
            )
            self.tracks[track_id] = track
            return track

        track = self.tracks[track_id]
        track.bbox = bbox
        track.confidence = confidence
        track.history.append((cx, cy))
        track.frame_history.append(frame_id)
        track.persistence += 1

        # Keep history within window
        if len(track.history) > self.window_size:
            track.history.pop(0)
            track.frame_history.pop(0)

        # Calculate tracking consistency over recent window
        # (number of frames observed vs expected span)
        if len(track.frame_history) >= 2:
            frame_span = track.frame_history[-1] - track.frame_history[0] + 1
            track.tracking_consistency = min(1.0, len(track.frame_history) / max(1, frame_span))

        # Calculate motion displacement
        if len(track.history) >= 3:
            first_cx, first_cy = track.history[0]
            last_cx, last_cy = track.history[-1]
            net_disp = math.hypot(last_cx - first_cx, last_cy - first_cy)
            track.total_displacement = net_disp

            # Check if stationary across multiple frames
            if track.persistence >= 8 and net_disp < self.stationary_pixel_thresh:
                track.is_stationary = True
                track.motion_score = float(np.clip(net_disp / self.stationary_pixel_thresh * 0.2, 0.0, 0.2))
            else:
                track.is_stationary = False
                # Normalize motion score: 40px displacement over window ~ 1.0
                track.motion_score = float(np.clip(net_disp / 35.0, 0.05, 1.0))
        else:
            track.motion_score = 0.3 # Initial default

        return track

    def purge_lost_tracks(self, current_frame_id: int, max_lost_frames: int = 30):
        lost_ids = [
            t_id for t_id, t in self.tracks.items()
            if (current_frame_id - t.frame_history[-1]) > max_lost_frames
        ]
        for t_id in lost_ids:
            del self.tracks[t_id]
