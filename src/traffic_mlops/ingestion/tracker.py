import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ultralytics import YOLO

from traffic_mlops.ingestion.config import YoloVehicleTrackerConfig
from traffic_mlops.ingestion.video_reader import get_video_properties


_MODEL: YOLO | None = None


def _init_worker(model_name: str, device: str | None = None) -> None:
    """Initialize the YOLO model in a worker process.

    Loads the model into worker-local state for use by _track_one_video.
    Auto-selects CUDA if available when device is None.

    Args:
        model_name: Name of the YOLO model to load.
        device: Device to run inference on (e.g. 'cuda', 'cpu'). Defaults to
            None (auto-detect).
    """
    global _MODEL
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _MODEL = YOLO(model_name)
    _MODEL.to(device)


def _track_one_video(video_path: Path, config: YoloVehicleTrackerConfig) -> None:
    """Run vehicle tracking on a single video and write detections to CSV.

    Uses the worker-initialized YOLO model to track vehicles frame-by-frame,
    then writes bounding boxes and metadata to output_dir/{video_id}.csv.

    Args:
        video_path: Path to the input video file.
        config: Tracker configuration (model params, output dir, etc.).

    Raises:
        RuntimeError: If the worker model was not initialized.
    """
    global _MODEL
    if _MODEL is None:
        raise RuntimeError("Model not initialized in worker")

    video_props = get_video_properties(video_path)

    rows: list[dict[str, Any]] = []

    for frame_idx, result in enumerate(
        _MODEL.track(
            video_path,
            classes=config.vehicle_classes,
            conf=config.confidence,
            persist=True,
            verbose=False,
            show=False,
            stream=True,
        )
    ):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        boxes = boxes.cpu()

        track_ids = boxes.id.tolist()  # type: ignore  # noqa: PGH003
        xyxy = boxes.xyxy.tolist()
        class_ids = boxes.cls.tolist()
        confidences = boxes.conf.tolist()
        t = frame_idx / video_props["fps"]

        for track_id, class_id, (x1, y1, x2, y2), confidence in zip(
            track_ids, class_ids, xyxy, confidences, strict=False
        ):
            rows.append(
                {
                    **video_props,
                    "frame_idx": frame_idx,
                    "t": t,
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(confidence),
                }
            )

    # For empty dataframes, we need to specify the columns
    df = pd.DataFrame(
        rows,
        columns=[
            "video_id",
            "fps",
            "width",
            "height",
            "frame_idx",
            "t",
            "track_id",
            "class_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
        ],
    )
    output_file = config.output_dir / f"{video_props['video_id']}.csv"
    df.to_csv(output_file, index=False)


class YoloVehicleTracker:
    """Process videos in parallel to extract vehicle tracks via YOLO."""

    @staticmethod
    def track(config: YoloVehicleTrackerConfig) -> None:
        """Run vehicle tracking on all MP4 videos in the input directory.

        Spawns a process pool, tracks vehicles in each video, and writes
        per-video CSV files to the configured output directory.

        Args:
            config: Tracker configuration (paths, model, workers, etc.).

        Raises:
            ValueError: If no MP4 files are found in input_dir.
        """
        video_paths = list(config.input_dir.glob("*.mp4"))
        if not video_paths:
            raise ValueError(f"No videos found in {config.input_dir}")

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=config.n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(config.model_name, config.device),
        ) as executor:
            futures = [
                executor.submit(_track_one_video, video_path, config)
                for video_path in video_paths
            ]

            for future in as_completed(futures):
                future.result()
