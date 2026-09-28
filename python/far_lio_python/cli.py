"""Run the Python odometry reference on timestamped .npy/.npz LiDAR frames."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .geometry import Pose
from .pipeline import OdometryPipeline, PipelineConfig
from .pointcloud import PointCloud


@dataclass(kw_only=True)
class Arguments:
    input_dir: Path
    output_dir: Path
    config_file: Path | None = None
    initial_map: Path | None = None
    guesses_csv: Path | None = None
    deskew_poses_csv: Path | None = None
    method: Literal["gicp", "icp"] = "gicp"
    max_time_ms: float | None = None
    overwrite: bool = False


def read_poses(path: Path) -> dict[int, Pose]:
    poses: dict[int, Pose] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            stamp = int(row["timestamp_ns"])
            if stamp in poses:
                raise ValueError(f"duplicate timestamp {stamp} in {path}")
            translation = np.array([float(row[name]) for name in ("tx", "ty", "tz")])
            quaternion = np.array([float(row[name]) for name in ("qx", "qy", "qz", "qw")])
            poses[stamp] = Pose(rotation=Rotation.from_quat(quaternion), translation=translation)
    return poses


def read_cloud(path: Path) -> PointCloud:
    if path.suffix == ".npy":
        return PointCloud(xyz=np.load(path, allow_pickle=False))
    with np.load(path, allow_pickle=False) as data:
        if "xyz" not in data:
            raise ValueError(f"{path} is missing the 'xyz' array")
        offsets = data["time_offset_s"] if "time_offset_s" in data else None
        return PointCloud(xyz=data["xyz"], time_offset_s=offsets)


def run_pipeline(args: Arguments) -> Path:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory not found: {input_dir}")
    frames = sorted(
        (p for p in input_dir.iterdir() if p.suffix in {".npy", ".npz"}),
        key=lambda p: int(p.stem),
    )
    if not frames:
        raise ValueError(f"no timestamp_ns.npy or timestamp_ns.npz frames in {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir} (use --overwrite)")

    config = PipelineConfig()
    if args.config_file is not None:
        with args.config_file.expanduser().open() as stream:
            config = PipelineConfig.from_ros_yaml(yaml.safe_load(stream))
    overrides: dict[str, str | float] = {"method": args.method}
    if args.max_time_ms is not None:
        overrides["max_time_ms"] = args.max_time_ms
    config = replace(config, registration=replace(config.registration, **overrides))
    pipeline = OdometryPipeline(config)
    if args.initial_map is not None:
        prior = read_cloud(args.initial_map.expanduser().resolve())
        pipeline.load_map(prior.xyz)
    guesses = read_poses(args.guesses_csv.expanduser()) if args.guesses_csv else {}
    deskew_poses = read_poses(args.deskew_poses_csv.expanduser()) if args.deskew_poses_csv else {}
    history = iter(sorted(deskew_poses.items()))
    next_history = next(history, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = output_dir / "trajectory.csv"
    fields = [
        "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        "vx", "vy", "vz", "wx", "wy", "wz", "status", "correspondences",
        "iterations", "duration_ms", "deskewed", "map_points",
    ]
    with trajectory.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for path in frames:
            stamp_ns = int(path.stem)
            while next_history is not None and next_history[0] <= stamp_ns:
                pipeline.set_deskew_pose(*next_history)
                next_history = next(history, None)
            result = pipeline.register_frame(
                read_cloud(path), stamp_ns, initial_guess=guesses.get(stamp_ns)
            )
            reg = result.registration
            writer.writerow([
                stamp_ns, *result.pose.translation, *result.pose.rotation.as_quat(),
                *result.twist, result.status, reg.correspondences if reg else 0,
                reg.iterations if reg else 0, reg.duration_ms if reg else 0.0,
                result.deskewed, result.map_points,
            ])
            print(f"{stamp_ns}: {result.status}, {result.map_points} map points")
    np.savez_compressed(output_dir / "map.npz", xyz=pipeline.map.arrays(with_covariance=False)[0])
    return trajectory


def main() -> None:
    import tyro

    args = tyro.cli(Arguments)
    print(f"Wrote {run_pipeline(args)}")


if __name__ == "__main__":
    main()
