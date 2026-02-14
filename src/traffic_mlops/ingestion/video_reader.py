from pathlib import Path
from typing import Any

import cv2


def get_video_properties(video_path: str | Path) -> dict[str, Any]:
    if isinstance(video_path, str):
        video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_id = video_path.stem
    cap = cv2.VideoCapture(str(video_path))

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "video_id": video_id,
        "fps": fps,
        "width": width,
        "height": height,
    }
