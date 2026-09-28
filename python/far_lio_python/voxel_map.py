"""Bounded voxel map and local covariances for CPU GICP."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .geometry import Pose

_OFFSETS = tuple(product((-1, 0, 1), repeat=3))


def regularize_covariance(covariance: NDArray[np.float64], method: str) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if method == "SVD":
        values = np.array([1e-3, 1.0, 1.0])
    elif method == "MIN_EIGENVALUE":
        values = np.maximum(eigenvalues, 1e-3)
    elif method == "FROBENIUS":
        # Equivalent to (inv(C) / ||inv(C)||_F)^-1 for SPD C.
        values = eigenvalues + 1e-3
        values *= np.linalg.norm(1.0 / values)
    else:
        raise ValueError(f"unknown covariance regularization: {method}")
    return (eigenvectors * values) @ eigenvectors.T


def local_covariances(
    xyz: NDArray[np.float64],
    *,
    voxel_size: float,
    num_neighbors: int,
    method: str,
) -> NDArray[np.float64]:
    """For each point, use its nearest points within its 27 neighboring voxels."""
    n = len(xyz)
    covariances = np.zeros((n, 3, 3), dtype=np.float64)
    if n < num_neighbors:
        return covariances
    voxel_keys = np.floor(xyz / voxel_size).astype(np.int64)
    voxels: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(voxel_keys):
        voxels.setdefault(tuple(key), []).append(index)
    tree = cKDTree(xyz)
    candidate_count = min(n, max(4 * num_neighbors, 32))
    _, candidates = tree.query(xyz, k=candidate_count)
    candidates = np.asarray(candidates).reshape(n, candidate_count)

    for index, key in enumerate(voxel_keys):
        nearby = candidates[index]
        nearby = nearby[np.all(np.abs(voxel_keys[nearby] - key) <= 1, axis=1)]
        if len(nearby) < num_neighbors:
            local = [
                candidate
                for offset in _OFFSETS
                for candidate in voxels.get(tuple(key + offset), ())
            ]
            if len(local) < num_neighbors:
                continue
            local = np.asarray(local, dtype=np.int64)
            distances = np.sum((xyz[local] - xyz[index]) ** 2, axis=1)
            nearby = local[np.argpartition(distances, num_neighbors - 1)[:num_neighbors]]
        else:
            nearby = nearby[:num_neighbors]
        neighbors = xyz[nearby]
        centered = neighbors - neighbors.mean(axis=0)
        covariance = centered.T @ centered / num_neighbors
        covariances[index] = regularize_covariance(covariance, method)
    return covariances


@dataclass(kw_only=True)
class VoxelMap:
    voxel_size: float
    max_distance: float
    max_points_per_voxel: int = 40
    num_neighbors: int = 10
    cov_regularization: str = "FROBENIUS"
    _voxels: dict[tuple[int, int, int], list[NDArray[np.float64]]] = field(default_factory=dict, init=False)
    _xyz: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 3)), init=False)
    _covariances: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 3, 3)), init=False)
    _keys: NDArray[np.int64] = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64), init=False)
    _indices: dict[tuple[int, int, int], list[int]] = field(default_factory=dict, init=False)
    _tree: cKDTree | None = field(default=None, init=False)
    _dirty: bool = field(default=True, init=False)
    _covariance_dirty: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.voxel_size <= 0 or self.max_distance <= 0:
            raise ValueError("voxel_size and max_distance must be positive")
        if self.max_points_per_voxel < 1 or self.num_neighbors < 3:
            raise ValueError("max_points_per_voxel >= 1 and num_neighbors >= 3 are required")
        if self.cov_regularization not in {"SVD", "FROBENIUS", "MIN_EIGENVALUE"}:
            raise ValueError("invalid covariance regularization")

    @property
    def resolution(self) -> float:
        return self.voxel_size / np.sqrt(self.max_points_per_voxel)

    @property
    def empty(self) -> bool:
        return not self._voxels

    @property
    def num_points(self) -> int:
        return sum(map(len, self._voxels.values()))

    def clear(self) -> None:
        self._voxels.clear()
        self._dirty = True

    def add_points(self, xyz: NDArray[np.float64], *, origin: NDArray[np.float64] | None = None) -> int:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("map points must be a finite (N, 3) array")
        added = 0
        resolution2 = self.resolution**2
        for point in xyz:
            key = tuple(np.floor(point / self.voxel_size).astype(np.int64))
            bucket = self._voxels.setdefault(key, [])
            if len(bucket) >= self.max_points_per_voxel:
                continue
            if any(np.sum((point - other) ** 2) < resolution2 for other in bucket):
                continue
            bucket.append(point.copy())
            added += 1
        if origin is not None:
            origin = np.asarray(origin, dtype=np.float64)
            if origin.shape != (3,):
                raise ValueError("origin must have shape (3,)")
            max_distance2 = self.max_distance**2
            for key in list(self._voxels):
                if self._voxels[key] and np.sum((self._voxels[key][0] - origin) ** 2) >= max_distance2:
                    del self._voxels[key]
        self._dirty = True
        return added

    def update(self, xyz: NDArray[np.float64], pose: Pose) -> int:
        return self.add_points(pose.transform(xyz), origin=pose.translation)

    def _refresh(self) -> None:
        if not self._dirty:
            return
        self._xyz = np.asarray([p for bucket in self._voxels.values() for p in bucket], dtype=np.float64).reshape(-1, 3)
        self._keys = np.floor(self._xyz / self.voxel_size).astype(np.int64)
        self._indices = {}
        for index, key in enumerate(self._keys):
            self._indices.setdefault(tuple(key), []).append(index)
        self._tree = cKDTree(self._xyz) if len(self._xyz) else None
        self._covariance_dirty = True
        self._dirty = False

    def arrays(self, *, with_covariance: bool = True) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._refresh()
        if with_covariance and self._covariance_dirty:
            self._covariances = local_covariances(
                self._xyz,
                voxel_size=self.voxel_size,
                num_neighbors=self.num_neighbors,
                method=self.cov_regularization,
            )
            self._covariance_dirty = False
        if not with_covariance:
            return self._xyz.copy(), np.empty((0, 3, 3))
        return self._xyz.copy(), self._covariances.copy()

    def match(
        self, query: NDArray[np.float64], max_distance: float
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return source and target indices, with the same 27-voxel search as the C++ map."""
        self._refresh()
        if self._tree is None or not len(query):
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        distances, indices = self._tree.query(query, distance_upper_bound=max_distance)
        source_keys = np.floor(query / self.voxel_size).astype(np.int64)
        target = np.asarray(indices, dtype=np.int64)
        valid = target < len(self._xyz)
        for i in np.flatnonzero(valid):
            if np.all(np.abs(source_keys[i] - self._keys[target[i]]) <= 1):
                continue
            local = [j for offset in _OFFSETS for j in self._indices.get(tuple(source_keys[i] + offset), ())]
            if not local:
                valid[i] = False
                continue
            local = np.asarray(local, dtype=np.int64)
            squared = np.sum((self._xyz[local] - query[i]) ** 2, axis=1)
            closest = int(np.argmin(squared))
            if squared[closest] >= max_distance**2:
                valid[i] = False
            else:
                target[i] = local[closest]
        # cKDTree uses <= distance_upper_bound; the original uses strict <.
        valid &= np.asarray(distances) < max_distance
        return np.flatnonzero(valid), target[valid]
