from pathlib import Path

from pydantic import BaseModel
from pydantic import Field


class YoloVehicleTrackerConfig(BaseModel):
    """Configuration for YOLO-based vehicle tracking over video files.

    Attributes:
        model_name: Name of the YOLO model to use for vehicle tracking.
        input_dir: Directory containing the input video files.
        output_dir: Directory to save the output video files.
        n_workers: Number of workers to use for processing the video files.
        confidence: Confidence threshold for the YOLO model.
        device: Device to use for processing (e.g. 'cuda', 'cpu'). None
            triggers auto-detection.
        vehicle_classes: COCO class IDs to track. Defaults to car, motorcycle,
            bus, truck. See Ultralytics docs for full class mapping.
    """

    model_name: str = Field(
        ...,
        description="Name of the YOLO model to use for vehicle tracking.",
    )
    input_dir: Path = Field(
        ...,
        description="Directory containing the input video files.",
    )
    output_dir: Path = Field(
        ...,
        description="Directory to save the output video files.",
    )
    n_workers: int = Field(
        default=4,
        description="Number of workers to use for processing the video files.",
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence threshold for the YOLO model.",
    )
    device: str | None = Field(
        default=None,
        description="Device to use for processing.",
    )
    vehicle_classes: list[int] = Field(
        default=[2, 3, 5, 7],
        description=(
            "Classes of vehicles to track. The classes are defined in the YOLO model."
            "See https://docs.ultralytics.com/models/detect/#classes for more details."
        ),
    )
