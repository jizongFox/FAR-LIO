"""Point cloud validation, cropping, and first-point voxel sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class PointCloud:
    xyz: NDArray[np.float64]
    time_offset_s: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        object.__setattr__(self, "xyz", xyz.copy())
        if self.time_offset_s is not None:
            offsets = np.asarray(self.time_offset_s, dtype=np.float64)
            if offsets.shape != (len(xyz),):
                raise ValueError("time_offset_s must have shape (N,)")
            object.__setattr__(self, "time_offset_s", offsets.copy())

    def select(self, selector: NDArray[np.bool_] | NDArray[np.int64]) -> PointCloud:
        offsets = None if self.time_offset_s is None else self.time_offset_s[selector]
        return PointCloud(xyz=self.xyz[selector], time_offset_s=offsets)


def preprocess(
    cloud: PointCloud,
    *,
    crop_range: tuple[float, float] | None,
    crop_footprint: tuple[float, float] | None,
) -> PointCloud:
    mask = np.isfinite(cloud.xyz).all(axis=1)
    if cloud.time_offset_s is not None:
        mask &= np.isfinite(cloud.time_offset_s)
    if crop_range is not None:
        radius = np.linalg.norm(cloud.xyz, axis=1)
        mask &= (radius >= crop_range[0]) & (radius <= crop_range[1])
    if crop_footprint is not None:
        longitudinal, lateral = crop_footprint
        mask &= ~((np.abs(cloud.xyz[:, 0]) < longitudinal) & (np.abs(cloud.xyz[:, 1]) < lateral))
    return cloud.select(mask)


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if not len(cloud.xyz):
        return cloud
    keys = np.floor(cloud.xyz / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return cloud.select(np.sort(indices))


def double_downsample(cloud: PointCloud, voxel_size: float) -> tuple[PointCloud, PointCloud]:
    frame_map = voxel_downsample(cloud, voxel_size * 0.5)
    frame_registration = voxel_downsample(frame_map, voxel_size * 1.5)
    return frame_registration, frame_map
