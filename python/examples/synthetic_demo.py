#!/usr/bin/env python3
"""Generate four LiDAR scans, run the offline CLI pipeline, and report pose error.

Usage from the repository root after ``python -m pip install -e ./python``:
    python python/examples/synthetic_demo.py --output-dir ./python-demo
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from far_lio_python import Pose
from far_lio_python.cli import Arguments, run_pipeline


@dataclass(kw_only=True)
class DemoConfig:
    output_dir: Path = Path("python-demo")


def main() -> None:
    import tyro

    config = tyro.cli(DemoConfig)
    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"demo output exists: {output_dir}; choose another --output-dir")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    world = rng.uniform([-15, -10, -3], [15, 10, 4], (2800, 3))
    truth: dict[int, Pose] = {}
    for frame_id in range(4):
        stamp_ns = 1_000_000_000 + 100_000_000 * frame_id
        pose = Pose(
            rotation=Rotation.from_euler("xyz", [0.005 * frame_id, -0.003 * frame_id, 0.02 * frame_id]),
            translation=np.array([0.35 * frame_id, -0.08 * frame_id, 0.02 * frame_id]),
        )
        truth[stamp_ns] = pose
        xyz = pose.inverse().transform(world) + rng.normal(0, 0.005, world.shape)
        np.save(frames_dir / f"{stamp_ns}.npy", xyz)

    trajectory = run_pipeline(Arguments(input_dir=frames_dir, output_dir=output_dir / "result"))
    with trajectory.open(newline="") as stream:
        for row in csv.DictReader(stream):
            stamp_ns = int(row["timestamp_ns"])
            position = np.array([float(row[name]) for name in ("tx", "ty", "tz")])
            quaternion = np.array([float(row[name]) for name in ("qx", "qy", "qz", "qw")])
            translation_error = np.linalg.norm(position - truth[stamp_ns].translation)
            rotation_error = np.linalg.norm((Rotation.from_quat(quaternion).inv() * truth[stamp_ns].rotation).as_rotvec())
            print(
                f"{stamp_ns}: {row['status']}, translation error {translation_error:.4f} m, "
                f"rotation error {np.degrees(rotation_error):.4f} deg"
            )
    print(f"Trajectory: {trajectory}")
    print(f"Map: {output_dir / 'result' / 'map.npz'}")


if __name__ == "__main__":
    main()
