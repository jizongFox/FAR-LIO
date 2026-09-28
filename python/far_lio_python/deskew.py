"""Linear Lie-algebra motion fit used for optional per-point deskewing."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose
from .pointcloud import PointCloud


@dataclass(kw_only=True)
class PoseHistory:
    max_age_s: float = 0.2
    max_time_diff_ms: float = 30.0
    velocity_threshold_mps: float = 3.0
    _stamps: list[int] = field(default_factory=list, init=False)
    _poses: list[Pose] = field(default_factory=list, init=False)

    def add(self, stamp_ns: int, pose: Pose) -> None:
        if self._stamps and stamp_ns < self._stamps[0]:
            self._stamps.clear()  # Bag replay jumped back in time.
            self._poses.clear()
        index = bisect_left(self._stamps, stamp_ns)
        if index < len(self._stamps) and self._stamps[index] == stamp_ns:
            self._poses[index] = pose
        else:
            self._stamps.insert(index, stamp_ns)
            self._poses.insert(index, pose)
        oldest = stamp_ns - round(self.max_age_s * 1e9)
        while self._stamps and self._stamps[0] < oldest:
            self._stamps.pop(0)
            self._poses.pop(0)

    def _interpolate(self, stamp_ns: int) -> Pose:
        index = min(max(bisect_left(self._stamps, stamp_ns), 1), len(self._stamps) - 1)
        t0, t1 = self._stamps[index - 1 : index + 1]
        return self._poses[index - 1].interpolate(self._poses[index], (stamp_ns - t0) / (t1 - t0))

    def deskew(self, cloud: PointCloud, frame_stamp_ns: int) -> tuple[PointCloud, bool]:
        if cloud.time_offset_s is None or len(self._stamps) < 2:
            return cloud, False
        dt = (self._stamps[-1] - self._stamps[-2]) * 1e-9
        if dt <= 0 or np.linalg.norm(self._poses[-1].translation - self._poses[-2].translation) / dt < self.velocity_threshold_mps:
            return cloud, False
        max_diff_ns = self.max_time_diff_ms * 1e6
        if not any(abs(stamp - (frame_stamp_ns - 100_000_000)) < max_diff_ns for stamp in self._stamps):
            return cloud, False
        if not any(abs(stamp - frame_stamp_ns) < max_diff_ns for stamp in self._stamps):
            return cloud, False
        reference = self._interpolate(frame_stamp_ns).inverse()
        times = (np.asarray(self._stamps, dtype=np.float64) - frame_stamp_ns) * 1e-9
        twists = np.array([reference.compose(pose).log() for pose in self._poses])
        coefficients = np.linalg.lstsq(np.column_stack((np.ones(len(times)), times)), twists, rcond=None)[0]
        corrected = cloud.xyz.copy()
        offsets = np.clip(cloud.time_offset_s, times[0], times[-1])
        for index, offset in enumerate(offsets):
            corrected[index] = Pose.exp(coefficients[0] + offset * coefficients[1]).transform(corrected[index])
        return PointCloud(xyz=corrected, time_offset_s=cloud.time_offset_s), True
