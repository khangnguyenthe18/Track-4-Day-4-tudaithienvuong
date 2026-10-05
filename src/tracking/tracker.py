import numpy as np
from typing import List, Dict, Tuple, Optional
from .trajectory import TrackInfo, TrajectoryManager

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class PedestrianTracker:
    """
    ByteTrack-style Multi-Object Tracker for Pedestrian Sequences.
    Uses two-stage association (high conf, then low conf) to handle
    partial occlusions, rain, and low-light detection dips.
    """
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.15,
        match_thresh: float = 0.35,
        max_lost_frames: int = 25
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = max_lost_frames
        
        self.next_id = 1
        self.trajectory_manager = TrajectoryManager()
        self.active_tracks: Dict[int, List[float]] = {} # id -> bbox
        self.track_lost_count: Dict[int, int] = {} # id -> frames without match

    def update(
        self,
        detections: List[Dict], # list of dicts with 'bbox', 'confidence', 'class_id'
        frame_id: int
    ) -> List[TrackInfo]:
        """
        Process detections for frame_id.
        detections: [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}, ...]
        Returns: list of updated TrackInfo objects for currently tracked objects.
        """
        # Split detections into high and low confidence
        high_dets = [d for d in detections if d.get('confidence', 0.0) >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.get('confidence', 0.0) < self.high_thresh]

        track_ids = list(self.active_tracks.keys())
        matched_tracks = set()
        matched_high_dets = set()

        # Step 1: Match high confidence detections with active tracks
        if track_ids and high_dets:
            iou_matrix = np.zeros((len(track_ids), len(high_dets)), dtype=float)
            for t_idx, t_id in enumerate(track_ids):
                for d_idx, det in enumerate(high_dets):
                    iou_matrix[t_idx, d_idx] = compute_iou(self.active_tracks[t_id], det['bbox'])

            # Greedy matching for high confidence
            while True:
                max_val = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_val < self.match_thresh:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                t_id = track_ids[t_idx]
                matched_tracks.add(t_id)
                matched_high_dets.add(d_idx)
                
                det = high_dets[d_idx]
                self.active_tracks[t_id] = det['bbox']
                self.track_lost_count[t_id] = 0
                self.trajectory_manager.update_track(
                    track_id=t_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    frame_id=frame_id,
                    class_id=det.get('class_id', 0)
                )

                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Step 2: Match remaining tracks with low confidence detections (ByteTrack second stage)
        remaining_track_ids = [t_id for t_id in track_ids if t_id not in matched_tracks]
        if remaining_track_ids and low_dets:
            low_iou_matrix = np.zeros((len(remaining_track_ids), len(low_dets)), dtype=float)
            for t_idx, t_id in enumerate(remaining_track_ids):
                for d_idx, det in enumerate(low_dets):
                    low_iou_matrix[t_idx, d_idx] = compute_iou(self.active_tracks[t_id], det['bbox'])

            while True:
                max_val = np.max(low_iou_matrix) if low_iou_matrix.size > 0 else 0.0
                if max_val < (self.match_thresh * 0.8): # slightly lower threshold for low conf recovery
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(low_iou_matrix), low_iou_matrix.shape)
                t_id = remaining_track_ids[t_idx]
                matched_tracks.add(t_id)
                
                det = low_dets[d_idx]
                self.active_tracks[t_id] = det['bbox']
                self.track_lost_count[t_id] = 0
                self.trajectory_manager.update_track(
                    track_id=t_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    frame_id=frame_id,
                    class_id=det.get('class_id', 0)
                )

                low_iou_matrix[t_idx, :] = -1.0
                low_iou_matrix[:, d_idx] = -1.0

        # Step 3: Unmatched high confidence detections become new tracks
        for d_idx, det in enumerate(high_dets):
            if d_idx not in matched_high_dets:
                new_id = self.next_id
                self.next_id += 1
                self.active_tracks[new_id] = det['bbox']
                self.track_lost_count[new_id] = 0
                self.trajectory_manager.update_track(
                    track_id=new_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    frame_id=frame_id,
                    class_id=det.get('class_id', 0)
                )
                matched_tracks.add(new_id)

        # Step 4: Age unmatched tracks and purge dead tracks
        unmatched_ids = [t_id for t_id in track_ids if t_id not in matched_tracks]
        for t_id in unmatched_ids:
            self.track_lost_count[t_id] = self.track_lost_count.get(t_id, 0) + 1
            if self.track_lost_count[t_id] > self.max_lost_frames:
                del self.active_tracks[t_id]
                del self.track_lost_count[t_id]

        self.trajectory_manager.purge_lost_tracks(frame_id, self.max_lost_frames)

        # Collect current active TrackInfo
        results = []
        for t_id in self.active_tracks.keys():
            if t_id in self.trajectory_manager.tracks:
                results.append(self.trajectory_manager.tracks[t_id])

        return results

    def reset(self):
        self.next_id = 1
        self.active_tracks.clear()
        self.track_lost_count.clear()
        self.trajectory_manager = TrajectoryManager()
