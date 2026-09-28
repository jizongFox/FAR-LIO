"""Point-to-point ICP and covariance-weighted GICP with left SE(3) updates."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .geometry import Pose
from .voxel_map import VoxelMap, local_covariances


@dataclass(frozen=True, kw_only=True)
class RegistrationConfig:
    method: Literal["gicp", "icp"] = "gicp"
    solver_type: Literal["GaussNewton", "LevenbergMarquardt"] = "GaussNewton"
    max_iter: int = 80
    max_inner_iter: int = 10
    max_time_ms: float = 0.0  # 0 disables the C++ wall-clock limit for offline CPU use.
    convergence_criterion: float = 5e-3
    damping_factor: float = 1e-9
    damping_scale: float = 2.0
    min_correspondences: int = 6

    def __post_init__(self) -> None:
        if self.max_iter < 1 or self.max_inner_iter < 1 or self.min_correspondences < 3:
            raise ValueError("iteration counts and min_correspondences must be positive")
        if self.convergence_criterion <= 0 or self.damping_factor < 0 or self.max_time_ms < 0:
            raise ValueError("invalid registration thresholds")
        if self.damping_scale <= 1:
            raise ValueError("damping_scale must exceed 1")


@dataclass(frozen=True, kw_only=True)
class RegistrationResult:
    pose: Pose
    converged: bool
    iterations: int
    correspondences: int
    duration_ms: float


def _normal_equations(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    precision: NDArray[np.float64],
    kernel_scale: float,
    method: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    residual = source - target
    squared = np.sum(residual**2, axis=1)
    if method == "gicp":
        weights = 1.0 / (1.0 + squared / kernel_scale**2)  # Cauchy
    else:
        weights = kernel_scale**2 / (kernel_scale**2 + squared) ** 2  # Geman-McClure

    jacobian = np.zeros((len(source), 3, 6))
    jacobian[:, :, :3] = np.eye(3)
    x, y, z = source.T
    jacobian[:, 0, 4] = z
    jacobian[:, 0, 5] = -y
    jacobian[:, 1, 3] = -z
    jacobian[:, 1, 5] = x
    jacobian[:, 2, 3] = y
    jacobian[:, 2, 4] = -x

    weighted = weights[:, None, None] * np.einsum("nij,njk->nik", precision, jacobian)
    hessian = np.einsum("nki,nkj->ij", jacobian, weighted)
    gradient = np.einsum(
        "nki,nk->i", jacobian, weights[:, None] * np.einsum("nij,nj->ni", precision, residual)
    )
    return hessian, gradient


def _weighted_error(
    source: NDArray[np.float64], target: NDArray[np.float64], precision: NDArray[np.float64]
) -> float:
    residual = source - target
    return float(0.5 * np.einsum("ni,nij,nj->", residual, precision, residual))


def register(
    frame: NDArray[np.float64],
    voxel_map: VoxelMap,
    initial_guess: Pose,
    *,
    sigma: float,
    config: RegistrationConfig,
) -> RegistrationResult:
    """Register one frame to the local map; return T_world_frame."""
    start = perf_counter()
    if voxel_map.empty:
        return RegistrationResult(
            pose=initial_guess, converged=True, iterations=0, correspondences=0, duration_ms=0.0
        )
    if not len(frame) or sigma <= 0:
        raise ValueError("frame must be nonempty and sigma positive")

    map_xyz, map_covariance = voxel_map.arrays(with_covariance=config.method == "gicp")
    frame_covariance = None
    if config.method == "gicp":
        # FAR-LIO builds a separate one-scan voxel map for source covariances.
        frame_map = VoxelMap(
            voxel_size=voxel_map.voxel_size,
            max_distance=voxel_map.max_distance,
            max_points_per_voxel=voxel_map.max_points_per_voxel,
            num_neighbors=voxel_map.num_neighbors,
            cov_regularization=voxel_map.cov_regularization,
        )
        frame_map.add_points(frame)
        frame_xyz, frame_covariance = frame_map.arrays()
    else:
        frame_xyz = frame

    pose = initial_guess
    converged = False
    count = 0
    num_iter = 0
    lam = 0.0
    for iteration in range(config.max_iter):
        source_all = pose.transform(frame_xyz)
        source_idx, target_idx = voxel_map.match(source_all, 3.0 * sigma)
        count = len(source_idx)
        if count < config.min_correspondences:
            break
        source = source_all[source_idx]
        target = map_xyz[target_idx]
        if frame_covariance is None:
            precision = np.broadcast_to(np.eye(3), (count, 3, 3))
        else:
            rotation = pose.rotation.as_matrix()
            source_cov = rotation @ frame_covariance[source_idx] @ rotation.T
            target_cov = map_covariance[target_idx]
            matrices = source_cov + target_cov
            no_cov = (
                np.max(np.abs(source_cov), axis=(1, 2)) < 1e-6
            ) | (np.max(np.abs(target_cov), axis=(1, 2)) < 1e-6)
            matrices[no_cov] = np.eye(3)
            try:
                precision = np.linalg.inv(matrices)
            except np.linalg.LinAlgError:
                break

        hessian, gradient = _normal_equations(source, target, precision, sigma / 3.0, config.method)
        if np.linalg.norm(gradient) <= 1e-9 and np.linalg.matrix_rank(hessian) == 6:
            converged = True  # An exactly stationary, fully constrained scan is valid.
            num_iter = iteration + 1
            break
        if config.solver_type == "LevenbergMarquardt" and iteration == 0:
            lam = max(config.damping_factor * np.max(np.abs(np.diag(hessian))), 1e-12)
        accepted = False
        for _ in range(config.max_inner_iter if config.solver_type == "LevenbergMarquardt" else 1):
            damping = lam if config.solver_type == "LevenbergMarquardt" else config.damping_factor
            try:
                step = np.linalg.solve(hessian + damping * np.eye(6), -gradient)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(step).all() or np.linalg.norm(step) <= 1e-12:
                break
            candidate = Pose.exp(step)
            if config.solver_type == "LevenbergMarquardt":
                old_error = _weighted_error(source, target, precision)
                new_error = _weighted_error(candidate.transform(source), target, precision)
                if new_error >= old_error:
                    lam *= config.damping_scale
                    continue
                lam = max(lam / config.damping_scale, 1e-12)
            pose = candidate.compose(pose)
            accepted = True
            break
        if not accepted:
            break
        num_iter = iteration + 1
        if np.linalg.norm(step) < config.convergence_criterion:
            converged = True
            break
        if config.max_time_ms and (perf_counter() - start) * 1e3 >= config.max_time_ms:
            break

    return RegistrationResult(
        pose=pose,
        converged=converged,
        iterations=num_iter,
        correspondences=count,
        duration_ms=(perf_counter() - start) * 1e3,
    )
