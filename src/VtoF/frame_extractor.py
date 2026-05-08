"""Video sahneleri için orta nokta kare çıkarma (yerel, API yok)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2

logger = logging.getLogger(__name__)


def extract_frame_at(video_path: str, time_seconds: float, out_path: str, max_dimension: int = 720) -> str:
    """time_seconds anındaki kareyi JPEG olarak kaydeder."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_idx = max(0, int(time_seconds * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Kare okunamadı: {video_path} @ {time_seconds}s")
        h, w = frame.shape[:2]
        if max_dimension and max(h, w) > max_dimension:
            scale = max_dimension / float(max(h, w))
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return out_path
    finally:
        cap.release()


def extract_scene_frames(
    video_path: str,
    scenes: List[Dict[str, Any]],
    frames_output_dir: str,
    max_dimension: int = 720,
) -> List[Dict[str, Any]]:
    """
    Her sahne için orta zaman karesini çıkarır.

    scenes: video_analyzer çıktısı (scene_id, start, end, duration, ...)
    """
    out: List[Dict[str, Any]] = []
    for scene in scenes:
        sid = scene["scene_id"]
        t0, t1 = 0.0, 0.0
        try:
            start_tc = scene["start"]
            end_tc = scene["end"]
            if isinstance(start_tc, (int, float)) and isinstance(end_tc, (int, float)):
                t0, t1 = float(start_tc), float(end_tc)
            else:
                from src.video_analyzer import timecode_to_seconds

                t0 = timecode_to_seconds(str(start_tc))
                t1 = timecode_to_seconds(str(end_tc))
            mid = (t0 + t1) / 2.0
        except Exception as e:
            logger.warning("Sahne %s süre hesabı başarısız: %s", sid, e)
            mid = 0.0

        frame_path = os.path.join(frames_output_dir, f"scene_{sid}_mid.jpg")
        try:
            extract_frame_at(video_path, mid, frame_path, max_dimension=max_dimension)
        except Exception as e:
            logger.warning("Sahne %s kare çıkarılamadı: %s", sid, e)
            frame_path = ""

        dur = scene.get("duration")
        if dur is None:
            dur = max(0.0, t1 - t0)
        out.append(
            {
                "scene_id": sid,
                "frame_path": frame_path,
                "duration": float(dur),
            }
        )
    return out
