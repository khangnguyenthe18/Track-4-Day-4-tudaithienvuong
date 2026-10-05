import os
import math
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from .degradation import apply_blur, apply_low_light, apply_noise, apply_dirty_lens

SCENARIOS = {
    "normal_day": {
        "title": "Tình huống 1: Ban ngày tiêu chuẩn (Người đi bộ bình thường)",
        "description": "Điều kiện ánh sáng ban ngày lý tưởng, ống kính sạch sẽ. Người đi bộ qua đường với quỹ đạo di chuyển rõ ràng, bám vết ổn định.",
        "expected_result": "BÌNH THƯỜNG",
        "expected_health": "Rất tốt (> 85%)"
    },
    "night_rain_rare": {
        "title": "Tình huống 2: Ban đêm & Mưa lớn (Tình huống hiếm có thật)",
        "description": "Môi trường đêm tối, mưa rơi làm mờ và nhiễu cảm biến. Người mặc áo tối màu qua đường. ByteTrack phát hiện chuyển động thật dù cảm biến suy giảm.",
        "expected_result": "TÌNH HUỐNG HIẾM CÓ THẬT",
        "expected_health": "Bị suy giảm (~45-55%)"
    },
    "dirty_lens_artifact": {
        "title": "Tình huống 3: Báo động giả do vết bẩn trên thấu kính",
        "description": "Ống kính bị bám vệt nước/bùn bẩn cố định kết hợp ảnh bị mờ. Bounding box kích hoạt nhưng vật thể đứng yên 100% qua nhiều frame.",
        "expected_result": "BÁO ĐỘNG GIẢ (CAMERA ARTIFACT)",
        "expected_health": "Kém (< 35%)"
    },
    "frozen_camera": {
        "title": "Tình huống 4: Sự cố cảm biến (Camera bị đứng hình / Treo)",
        "description": "Luồng video bị treo đứng hình đột ngột từ frame 25. Độ chênh lệch pixel giữa các frame bằng 0, kích hoạt cảnh báo lỗi phần cứng cảm biến.",
        "expected_result": "LỖI CAMERA / ĐỨNG HÌNH",
        "expected_health": "Nguy cấp (< 20%)"
    }
}

class SyntheticVideoGenerator:
    """
    Synthesizes simulated street video sequences to demonstrate pedestrian detection,
    ByteTrack tracking, camera health degradation, and rare-case classification.
    """
    def __init__(self, width: int = 854, height: int = 480, fps: int = 20):
        self.width = width
        self.height = height
        self.fps = fps
        self._cached_backgrounds = {}

    def _draw_road_background(self, is_night: bool = False) -> np.ndarray:
        h, w = self.height, self.width
        bg = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky / Horizon
        if is_night:
            sky_color = (25, 20, 30) # Dark navy night sky
            road_color = (35, 35, 38)
            sidewalk_color = (50, 48, 52)
        else:
            sky_color = (220, 200, 160) # Light daylight sky
            road_color = (95, 95, 100) # Asphalt
            sidewalk_color = (170, 170, 175)

        # Draw sky
        horizon_y = int(h * 0.45)
        bg[0:horizon_y, :] = sky_color

        # Background building silhouettes
        for bx in range(0, w, 90):
            bh = int(horizon_y * (0.3 + 0.4 * ((bx * 7) % 11) / 10.0))
            b_col = (sky_color[0] - 20, sky_color[1] - 20, sky_color[2] - 20)
            cv2.rectangle(bg, (bx, horizon_y - bh), (bx + 80, horizon_y), b_col, -1)

        # Draw Sidewalk
        cv2.rectangle(bg, (0, horizon_y), (w, int(h * 0.58)), sidewalk_color, -1)

        # Draw Road
        cv2.rectangle(bg, (0, int(h * 0.58)), (w, h), road_color, -1)

        # Draw Crosswalk Zebra Stripes
        cross_y1 = int(h * 0.60)
        cross_y2 = int(h * 0.95)
        stripe_color = (220, 220, 225) if not is_night else (140, 140, 145)
        for sx in range(120, w - 100, 70):
            cv2.rectangle(bg, (sx, cross_y1), (sx + 35, cross_y2), stripe_color, -1)

        # If night, draw street lamp lights
        if is_night:
            # Street lamps
            for lx in [200, 650]:
                cv2.line(bg, (lx, horizon_y - 80), (lx, horizon_y + 40), (80, 80, 80), 4)
                cv2.circle(bg, (lx, horizon_y - 80), 10, (150, 240, 255), -1)
                # Light cone overlay
                cone = bg.copy()
                pts = np.array([
                    [lx, horizon_y - 80],
                    [lx - 160, h],
                    [lx + 160, h]
                ])
                cv2.fillPoly(cone, [pts], (40, 70, 90))
                cv2.addWeighted(bg, 0.75, cone, 0.25, 0, bg)

        return bg

    def _draw_pedestrian(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        scale: float = 1.0,
        walk_phase: float = 0.0,
        dark_clothing: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        Draws an articulated pedestrian walking: head, torso, moving arms and legs.
        Returns bounding box [x1, y1, x2, y2].
        """
        head_r = int(14 * scale)
        torso_h = int(50 * scale)
        torso_w = int(24 * scale)
        leg_h = int(45 * scale)

        head_cx = int(x)
        head_cy = int(y - torso_h - leg_h)

        if dark_clothing:
            shirt_color = (40, 35, 45)
            pants_color = (25, 25, 30)
            skin_color = (130, 150, 180)
        else:
            shirt_color = (40, 120, 220) # Bright red/orange jacket
            pants_color = (130, 80, 50)  # Denim blue jeans
            skin_color = (160, 200, 240) # Skin tone

        # Legs with scissor walking animation
        leg_swing = math.sin(walk_phase) * (18 * scale)
        left_foot = (int(x - 6 - leg_swing), int(y))
        right_foot = (int(x + 6 + leg_swing), int(y))
        hip = (int(x), int(y - leg_h))

        cv2.line(frame, hip, left_foot, pants_color, int(8 * scale))
        cv2.line(frame, hip, right_foot, pants_color, int(8 * scale))

        # Torso
        torso_top = (int(x), int(y - leg_h - torso_h))
        cv2.rectangle(
            frame,
            (int(x - torso_w / 2), int(y - leg_h - torso_h)),
            (int(x + torso_w / 2), int(y - leg_h)),
            shirt_color,
            -1
        )

        # Arms swinging opposite to legs
        arm_swing = -math.sin(walk_phase) * (15 * scale)
        cv2.line(
            frame,
            (int(x - torso_w / 2), int(y - leg_h - torso_h + 8)),
            (int(x - torso_w / 2 - 8 - arm_swing), int(y - leg_h - 10)),
            shirt_color,
            int(5 * scale)
        )
        cv2.line(
            frame,
            (int(x + torso_w / 2), int(y - leg_h - torso_h + 8)),
            (int(x + torso_w / 2 + 8 + arm_swing), int(y - leg_h - 10)),
            shirt_color,
            int(5 * scale)
        )

        # Head
        cv2.circle(frame, (head_cx, head_cy), head_r, skin_color, -1)
        # Hair/Hat
        cv2.circle(frame, (head_cx, head_cy - int(head_r * 0.4)), int(head_r * 0.8), (20, 20, 20), -1)

        # Bounding box
        x1 = int(x - torso_w - 10 * scale)
        y1 = int(head_cy - head_r - 2)
        x2 = int(x + torso_w + 10 * scale)
        y2 = int(y + 2)

        return x1, y1, x2, y2

    def generate_frame(self, scenario_name: str, frame_idx: int, total_frames: int = 120) -> np.ndarray:
        """
        Generate a single frame for the selected scenario.
        """
        w, h = self.width, self.height
        is_night = (scenario_name == "night_rain_rare")

        if scenario_name not in self._cached_backgrounds:
            self._cached_backgrounds[scenario_name] = self._draw_road_background(is_night)
        
        bg = self._cached_backgrounds[scenario_name].copy()

        # Motion trajectory across crosswalk
        # Start at x=140, walk across to x=720
        progress = (frame_idx % total_frames) / float(total_frames)

        if scenario_name == "normal_day":
            ped_x = int(140 + progress * 550)
            ped_y = int(h * 0.78)
            walk_phase = frame_idx * 0.35
            self._draw_pedestrian(bg, ped_x, ped_y, scale=1.3, walk_phase=walk_phase, dark_clothing=False)
            return bg

        elif scenario_name == "night_rain_rare":
            ped_x = int(180 + progress * 520)
            ped_y = int(h * 0.80)
            walk_phase = frame_idx * 0.30
            # Dark rain scene with pedestrian in dark clothes
            self._draw_pedestrian(bg, ped_x, ped_y, scale=1.35, walk_phase=walk_phase, dark_clothing=True)
            # Add falling rain streaks
            np.random.seed(frame_idx * 17)
            for _ in range(120):
                rx = np.random.randint(0, w)
                ry = np.random.randint(0, h - 30)
                cv2.line(bg, (rx, ry), (rx - 6, ry + 16), (180, 190, 205), 1)

            # Apply low-light and noise
            dark_frame = apply_low_light(bg, brightness_factor=0.35)
            noisy_frame = apply_noise(dark_frame, noise_sigma=16.0)
            return noisy_frame

        elif scenario_name == "dirty_lens_artifact":
            # Blurred background
            blurred_bg = apply_blur(bg, severity=0.6)
            # Static dirty smudge in center that creates pseudo-pedestrian contour
            # Stationary across all frames (does not move with progress)
            smudge_frame = apply_dirty_lens(
                blurred_bg,
                smudge_centers=[
                    (int(w * 0.48), int(h * 0.65), 52),
                    (int(w * 0.50), int(h * 0.55), 35),
                ]
            )
            # Draw fake stationary silhouette inside the smudge to trigger detector false positive
            fake_x = int(w * 0.49)
            fake_y = int(h * 0.70)
            # Draw static blob
            cv2.ellipse(smudge_frame, (fake_x, fake_y - 30), (22, 45), 0, 0, 360, (50, 45, 40), -1)
            cv2.circle(smudge_frame, (fake_x, fake_y - 75), 14, (60, 55, 50), -1)
            # Soften
            smudge_frame = cv2.GaussianBlur(smudge_frame, (9, 9), 0)
            return smudge_frame

        elif scenario_name == "frozen_camera":
            # Freezes after frame 25
            frozen_idx = min(frame_idx, 25)
            ped_x = int(160 + (frozen_idx / float(total_frames)) * 520)
            ped_y = int(h * 0.78)
            walk_phase = frozen_idx * 0.35
            self._draw_pedestrian(bg, ped_x, ped_y, scale=1.3, walk_phase=walk_phase, dark_clothing=False)
            return bg

        return bg

    def get_scenario_bbox(self, scenario_name: str, frame_idx: int, total_frames: int = 120) -> Optional[List[float]]:
        """
        Returns ground truth [x1, y1, x2, y2] bounding box for synthetic scenarios.
        """
        h, w = self.height, self.width
        progress = (frame_idx % total_frames) / float(total_frames)

        if scenario_name == "normal_day":
            ped_x = int(140 + progress * 550)
            ped_y = int(h * 0.78)
            return [float(ped_x - 30), float(ped_y - 122), float(ped_x + 30), float(ped_y)]
        elif scenario_name == "night_rain_rare":
            ped_x = int(180 + progress * 520)
            ped_y = int(h * 0.80)
            return [float(ped_x - 32), float(ped_y - 126), float(ped_x + 32), float(ped_y)]
        elif scenario_name == "dirty_lens_artifact":
            fake_x = int(w * 0.49)
            fake_y = int(h * 0.70)
            return [float(fake_x - 26), float(fake_y - 105), float(fake_x + 26), float(fake_y)]
        elif scenario_name == "frozen_camera":
            frozen_idx = min(frame_idx, 25)
            ped_x = int(160 + (frozen_idx / float(total_frames)) * 520)
            ped_y = int(h * 0.78)
            return [float(ped_x - 30), float(ped_y - 122), float(ped_x + 30), float(ped_y)]
        return None

    def generate_video(
        self,
        scenario_name: str,
        output_path: str,
        num_frames: int = 120
    ) -> str:
        """
        Generate and save scenario to MP4 video file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Try mp4v or XVID
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        for f_idx in range(num_frames):
            frame = self.generate_frame(scenario_name, f_idx, total_frames=num_frames)
            writer.write(frame)

        writer.release()
        print(f"[SyntheticVideoGenerator] Generated {scenario_name} video: {output_path} ({num_frames} frames)")
        return output_path
