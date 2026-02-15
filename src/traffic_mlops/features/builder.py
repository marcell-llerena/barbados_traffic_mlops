from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TabularFeaturesBuilder:
    """Aggregate vehicle track CSVs into per-video feature vectors.

    Reads track CSVs from input_dir, computes vehicle counts, occupied area,
    and speed statistics per video, then writes a single features CSV to
    output_dir. Uses COCO class IDs: car=2, motorcycle=3, bus=5, truck=7.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> None:
        """Initialize the builder with input and output paths.

        Args:
            input_dir: Directory containing track CSV files (one per video).
            output_dir: Directory to write the aggregated features CSV.
        """
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._class_map = {
            "car": 2,
            "motorcycle": 3,
            "bus": 5,
            "truck": 7,
        }

    def _deduplicate_tracks(self, track_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve conflicting class_id values within each track.

        When a track has multiple class_id values across frames, replaces all
        with the mode (most frequent) for that track. Returns a copy.

        Args:
            track_df: DataFrame with columns track_id, class_id.

        Returns:
            Copy of track_df with class_id unified per track.
        """
        track_df = track_df.copy()
        track_df["class_id"] = track_df.groupby("track_id")["class_id"].transform(
            lambda x: x.mode()[0]
        )
        return track_df

    def _compute_vehicle_counts(self, track_df: pd.DataFrame) -> dict[str, Any]:
        """Compute per-class vehicle counts and ratios from unique tracks.

        One row per track_id; class_id taken as first occurrence. Returns
        counts and ratios for car, motorcycle, bus, truck, plus
        heavy_vehicles_ratio (bus + truck).

        Args:
            track_df: DataFrame with columns track_id, class_id.

        Returns:
            Dict with total_vehicles, *_count, *_ratio, heavy_vehicles_ratio.
        """
        tracks = (
            track_df.groupby("track_id")
            .agg(class_id=("class_id", "first"))
            .reset_index()
        )

        total = len(tracks)
        car_count = (tracks["class_id"] == self._class_map["car"]).sum()
        motorcycle_count = (tracks["class_id"] == self._class_map["motorcycle"]).sum()
        bus_count = (tracks["class_id"] == self._class_map["bus"]).sum()
        truck_count = (tracks["class_id"] == self._class_map["truck"]).sum()

        return {
            "total_vehicles": float(total),
            "car_count": float(car_count),
            "car_ratio": car_count / total if total else 0.0,
            "motorcycle_count": float(motorcycle_count),
            "motorcycle_ratio": motorcycle_count / total if total else 0.0,
            "bus_count": float(bus_count),
            "bus_ratio": bus_count / total if total else 0.0,
            "truck_count": float(truck_count),
            "truck_ratio": truck_count / total if total else 0.0,
            "heavy_vehicles_ratio": (bus_count + truck_count) / total if total else 0.0,
        }

    def _compute_occupied_area(self, track_df: pd.DataFrame) -> dict[str, Any]:
        """Compute bounding-box area statistics normalized by frame size.

        Each detection contributes (x2-x1)*(y2-y1) / (width*height). Returns
        sum, mean, std, max, min of normalized areas across all detections.

        Args:
            track_df: DataFrame with x1, y1, x2, y2, width, height.

        Returns:
            Dict with total_occupied_area, mean_occupied_area, std_occupied_area,
            max_occupied_area, min_occupied_area.
        """
        area_px = (track_df["x2"] - track_df["x1"]) * (track_df["y2"] - track_df["y1"])
        frame_area = track_df["width"].iloc[0] * track_df["height"].iloc[0]
        area_norm = area_px / frame_area

        return {
            "total_occupied_area": float(area_norm.sum()),
            "mean_occupied_area": float(area_norm.mean()) if len(area_norm) else 0.0,
            "std_occupied_area": float(area_norm.std()) if len(area_norm) > 1 else 0.0,
            "max_occupied_area": float(area_norm.max()) if len(area_norm) else 0.0,
            "min_occupied_area": float(area_norm.min()) if len(area_norm) else 0.0,
        }

    def _compute_speed_per_track(self, track_df: pd.DataFrame) -> pd.Series:
        """Compute mean speed per track from bounding-box centroid displacement.

        Speed is displacement (pixels) / dt between consecutive frames.
        Tracks with fewer than two detections are skipped. Returns one value
        per track (mean of frame-to-frame speeds).

        Args:
            track_df: DataFrame with track_id, t, x1, y1, x2, y2.

        Returns:
            Series of mean speeds per track; empty Series if no valid tracks.
        """
        track_df = track_df.sort_values(["track_id", "t"])
        track_df = track_df.copy()
        track_df["cx"] = (track_df["x1"] + track_df["x2"]) / 2
        track_df["cy"] = (track_df["y1"] + track_df["y2"]) / 2

        speeds = []
        for _, grp in track_df.groupby("track_id"):
            grp = grp.sort_values("t")
            if len(grp) < 2:
                continue
            dt = grp["t"].diff().dropna()
            dx = grp["cx"].diff().dropna()
            dy = grp["cy"].diff().dropna()
            displacement = np.sqrt(dx**2 + dy**2)
            speed = displacement / dt
            speed = speed[dt > 0]
            if len(speed) > 0:
                speeds.append(speed.mean())
        return pd.Series(speeds) if speeds else pd.Series(dtype=float)

    def _compute_speed_features(self, track_df: pd.DataFrame) -> dict[str, Any]:
        """Compute speed statistics across all tracks in the video.

        Aggregates mean, std, min, max, percentiles (10, 50, 75, 90), and
        range. Returns zeros for all fields when no valid speeds exist.

        Args:
            track_df: DataFrame with track_id, t, x1, y1, x2, y2.

        Returns:
            Dict with mean_speed, std_speed, min_speed, max_speed,
            p10_speed, p50_speed, p75_speed, p90_speed, speed_range.
        """
        speeds = self._compute_speed_per_track(track_df)

        if len(speeds) == 0:
            return {
                "mean_speed": 0.0,
                "std_speed": 0.0,
                "max_speed": 0.0,
                "min_speed": 0.0,
                "p10_speed": 0.0,
                "p50_speed": 0.0,
                "p75_speed": 0.0,
                "p90_speed": 0.0,
                "speed_range": 0.0,
            }

        return {
            "mean_speed": float(speeds.mean()),
            "std_speed": float(speeds.std()) if len(speeds) > 1 else 0.0,
            "max_speed": float(speeds.max()),
            "min_speed": float(speeds.min()),
            "p10_speed": float(speeds.quantile(0.10)),
            "p50_speed": float(speeds.quantile(0.50)),
            "p75_speed": float(speeds.quantile(0.75)),
            "p90_speed": float(speeds.quantile(0.90)),
            "speed_range": float(speeds.max() - speeds.min()),
        }

    def _build_features_for_file(self, track_file: Path) -> pd.DataFrame:
        """Build a single-row feature vector from one track CSV file.

        Reads the CSV, deduplicates tracks, computes counts, area, and speed
        features, then returns a one-row DataFrame. Empty input yields a row
        with only video_id. Side effect: reads from disk.

        Args:
            track_file: Path to the track CSV file.

        Returns:
            DataFrame with one row: video_id plus all computed features.
        """
        track_df = pd.read_csv(track_file)
        video_id = str(track_file.stem)

        if track_df.empty:
            return pd.DataFrame([{"video_id": video_id}])

        track_df = self._deduplicate_tracks(track_df)
        counts = self._compute_vehicle_counts(track_df)
        area = self._compute_occupied_area(track_df)
        speed = self._compute_speed_features(track_df)

        return pd.DataFrame([{"video_id": video_id, **counts, **area, **speed}])

    def build_features(self) -> None:
        """Process all track CSVs in input_dir and write features to output_dir.

        Glob *.csv from input_dir, build features per file, concatenate,
        impute NaN with 0, and write features_per_minute.csv to output_dir.
        Creates output_dir if it does not exist. Side effect: reads and writes
        files.

        Raises:
            ValueError: If no CSV files are found in input_dir.
            RuntimeError: If processing any file fails.
        """
        track_files = sorted(self._input_dir.glob("*.csv"))
        if not track_files:
            raise ValueError("No track files found")

        rows = []
        for track_file in track_files:
            try:
                df = self._build_features_for_file(track_file)
                rows.append(df)
            except Exception as e:
                raise RuntimeError(f"Failed to process {track_file}") from e

        result = pd.concat(rows, ignore_index=True)

        # Impute missing values by zero
        result = result.fillna(0)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "features_per_minute.csv"
        result.to_csv(out_path, index=False)
