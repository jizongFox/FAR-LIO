"""A small, explicit port of the FAR-LIO LiDAR odometry stage order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .deskew import PoseHistory
from .geometry import Pose
from .pointcloud import PointCloud, double_downsample, preprocess, voxel_downsample
from .registration import RegistrationConfig, RegistrationResult, register
from .voxel_map import VoxelMap


@dataclass(frozen=True, kw_only=True)
class PipelineConfig:
    update_map: bool = True
    undistort: bool = True
    preprocess: bool = True
    downsample: bool = False
    buffer_z: bool = False
    crop_range: tuple[float, float] | None = None
    crop_footprint: tuple[float, float] | None = None
    voxel_size: float = 4.0
    max_distance: float = 1000.0
    max_points_per_voxel: int = 40
    num_neighbors: int = 10
    cov_regularization: str = "FROBENIUS"
    initial_threshold: float = 1.0
    min_motion_threshold: float = 0.25
    max_correspondence_range: float = 100.0
    ellipsis_size_s: float = 5.0
    ellipsis_size_d: float = 2.5
    deskew_max_time_diff_ms: float = 30.0
    deskew_velocity_threshold_mps: float = 3.0
    pose_covariance_diagonal: tuple[float, ...] = (0.01, 0.01, 0.01, 0.001, 0.001, 0.001)
    twist_covariance_diagonal: tuple[float, ...] = (0.1, 0.1, 0.1, 0.01, 0.01, 0.01)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)

    def __post_init__(self) -> None:
        if self.crop_range is not None and (
            len(self.crop_range) != 2 or self.crop_range[0] < 0 or self.crop_range[0] > self.crop_range[1]
        ):
            raise ValueError("crop_range must be a nonnegative (min, max) pair")
        if self.crop_footprint is not None and (
            len(self.crop_footprint) != 2 or any(value < 0 for value in self.crop_footprint)
        ):
            raise ValueError("crop_footprint must be a nonnegative (longitudinal, lateral) pair")
        if min(self.initial_threshold, self.max_correspondence_range, self.ellipsis_size_s, self.ellipsis_size_d) <= 0:
            raise ValueError("thresholds and diagnostic ellipse axes must be positive")
        if self.min_motion_threshold < 0 or self.deskew_max_time_diff_ms <= 0 or self.deskew_velocity_threshold_mps < 0:
            raise ValueError("motion and deskew thresholds are invalid")
        for name in ("pose_covariance_diagonal", "twist_covariance_diagonal"):
            diagonal = getattr(self, name)
            if len(diagonal) != 6 or not np.isfinite(diagonal).all() or min(diagonal) < 0:
                raise ValueError(f"{name} must contain six nonnegative values")

    @classmethod
    def from_ros_yaml(cls, parameters: dict[str, Any]) -> PipelineConfig:
        """Read the LidarOdometry ros__parameters block from config/far-lio.yml."""
        if "/core/state" in parameters:
            parameters = parameters["/core/state"]["LidarOdometry"]["ros__parameters"]
        pipeline = parameters.get("pipeline", {})
        map_config = parameters.get("map", {})
        threshold = parameters.get("threshold", {})
        registration = parameters.get("registration", {})
        pre = parameters.get("preprocessing", {})
        covariance = parameters.get("covariance", {})
        distortion = parameters.get("distortion", {})

        def pair(value: Any) -> tuple[float, float] | None:
            return tuple(map(float, value)) if isinstance(value, list) and len(value) == 2 else None

        return cls(
            update_map=pipeline.get("update_map", True),
            undistort=pipeline.get("undistort", True),
            preprocess=pipeline.get("preprocess", True),
            downsample=pipeline.get("downsample", False),
            buffer_z=pipeline.get("buffer_z", False),
            crop_range=pair(pre.get("crop_range")),
            crop_footprint=pair(pre.get("crop_footprint")),
            voxel_size=map_config.get("voxel_size", 4.0),
            max_distance=map_config.get("max_distance", 1000.0),
            cov_regularization=map_config.get("cov_regularization", "FROBENIUS"),
            initial_threshold=threshold.get("initial_threshold", 1.0),
            min_motion_threshold=threshold.get("min_motion_threshold", 0.25),
            max_correspondence_range=threshold.get("max_correspondence_range", 100.0),
            ellipsis_size_s=parameters.get("diagnostic", {}).get("ellipsis_size_s", 5.0),
            ellipsis_size_d=parameters.get("diagnostic", {}).get("ellipsis_size_d", 2.5),
            deskew_max_time_diff_ms=distortion.get("max_time_diff", 30.0),
            deskew_velocity_threshold_mps=distortion.get("vel_threshold", 3.0),
            pose_covariance_diagonal=tuple(
                covariance.get("min_cov_translation", [0.01, 0.01, 0.01])
                + covariance.get("min_cov_orientation", [0.001, 0.001, 0.001])
            ),
            twist_covariance_diagonal=tuple(
                covariance.get("min_cov_linear_twist", [0.1, 0.1, 0.1])
                + covariance.get("min_cov_angular_twist", [0.01, 0.01, 0.01])
            ),
            registration=RegistrationConfig(
                solver_type=registration.get("solver_type", "GaussNewton"),
                max_iter=registration.get("max_iter", 80),
                max_inner_iter=registration.get("max_inner_iter", 10),
                max_time_ms=registration.get("max_time", 0.0),
                convergence_criterion=registration.get("convergence_criterion", 5e-3),
                damping_factor=registration.get("damping_factor", 1e-9),
                damping_scale=registration.get("damping_scale", 2.0),
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Odometry:
    stamp_ns: int
    pose: Pose
    twist: NDArray[np.float64]
    pose_covariance: NDArray[np.float64]
    twist_covariance: NDArray[np.float64]
    status: str
    registration: RegistrationResult | None
    map_points: int
    deskewed: bool = False


class AdaptiveThreshold:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model_sse = config.initial_threshold**2
        self.num_samples = 1
        self.previous_guess: Pose | None = None

    @property
    def sigma(self) -> float:
        return min(np.sqrt(self.model_sse / self.num_samples), 10.0)

    def update(self, guess: Pose, registered: Pose) -> None:
        if self.previous_guess is not None:
            error = guess.inverse().compose(registered)
            angle = np.linalg.norm(error.rotation.as_rotvec())
            model_error = np.linalg.norm(error.translation) + 2 * self.config.max_correspondence_range * np.sin(angle / 2)
            motion = np.linalg.norm(self.previous_guess.inverse().compose(guess).translation)
            if motion > 3 * self.config.min_motion_threshold and model_error > self.config.min_motion_threshold:
                self.model_sse += model_error**2
                self.num_samples += 1
        self.previous_guess = guess


class OdometryPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config = config
        self.map = VoxelMap(
            voxel_size=config.voxel_size,
            max_distance=config.max_distance,
            max_points_per_voxel=config.max_points_per_voxel,
            num_neighbors=config.num_neighbors,
            cov_regularization=config.cov_regularization,
        )
        self.threshold = AdaptiveThreshold(config)
        self.pose_history = PoseHistory(
            max_time_diff_ms=config.deskew_max_time_diff_ms,
            velocity_threshold_mps=config.deskew_velocity_threshold_mps,
        )
        self.previous: Odometry | None = None
        self.previous_previous: Odometry | None = None

    def set_deskew_pose(self, stamp_ns: int, pose: Pose) -> None:
        self.pose_history.add(stamp_ns, pose)

    def load_map(self, xyz: NDArray[np.float64]) -> None:
        if not self.map.empty:
            raise ValueError("load_map requires an empty map")
        self.map.add_points(xyz)

    def _predict(self) -> Pose:
        if self.previous is None:
            return Pose.identity()
        if self.previous_previous is None:
            return self.previous.pose
        # This is the source's constant-displacement predictor (not time-scaled).
        return self.previous.pose.compose(self.previous_previous.pose.inverse()).compose(self.previous.pose)

    def _twist(self, pose: Pose, stamp_ns: int) -> NDArray[np.float64]:
        if self.previous is None:
            return np.zeros(6)
        dt = (stamp_ns - self.previous.stamp_ns) * 1e-9
        if dt <= 1e-6 or dt > 1.0:
            return np.zeros(6)
        twist = self.previous.pose.inverse().compose(pose).log() / dt
        if np.linalg.norm(twist[:3]) > 100 or not np.isfinite(twist[:3]).all():
            twist[:3] = 0
        if np.linalg.norm(twist[3:]) > 10 or not np.isfinite(twist[3:]).all():
            twist[3:] = 0
        return twist

    def register_frame(
        self, cloud: PointCloud | NDArray[np.float64], stamp_ns: int, *, initial_guess: Pose | None = None
    ) -> Odometry:
        if stamp_ns < 0 or (self.previous is not None and stamp_ns <= self.previous.stamp_ns):
            raise ValueError("frame timestamps must be strictly increasing nanoseconds")
        if not isinstance(cloud, PointCloud):
            cloud = PointCloud(xyz=cloud)
        if self.map.empty and not self.config.update_map:
            raise ValueError("localization mode requires load_map() before register_frame()")
        guess = initial_guess if initial_guess is not None else self._predict()
        if self.config.buffer_z and self.previous is not None and self.previous.status == "ok":
            guess = guess.with_z(self.previous.pose.translation[2])
        if not len(cloud.xyz):
            return Odometry(
                stamp_ns=stamp_ns, pose=guess, twist=np.zeros(6), status="empty cloud",
                pose_covariance=np.diag(self.config.pose_covariance_diagonal),
                twist_covariance=np.diag(self.config.twist_covariance_diagonal),
                registration=None, map_points=self.map.num_points,
            )

        if self.config.undistort:
            cloud, deskewed = self.pose_history.deskew(cloud, stamp_ns)
        else:
            deskewed = False
        cloud = preprocess(
            cloud,
            crop_range=self.config.crop_range if self.config.preprocess else None,
            crop_footprint=self.config.crop_footprint if self.config.preprocess else None,
        )
        if not len(cloud.xyz):
            return Odometry(
                stamp_ns=stamp_ns, pose=guess, twist=np.zeros(6), status="empty after preprocessing",
                pose_covariance=np.diag(self.config.pose_covariance_diagonal),
                twist_covariance=np.diag(self.config.twist_covariance_diagonal),
                registration=None, map_points=self.map.num_points, deskewed=deskewed,
            )
        if self.config.downsample:
            frame_registration, frame_map = double_downsample(cloud, self.config.voxel_size)
        else:
            frame_registration = voxel_downsample(cloud, self.map.resolution)
            frame_map = frame_registration

        result = register(
            frame_registration.xyz, self.map, guess,
            sigma=self.threshold.sigma, config=self.config.registration,
        )
        pose = result.pose
        offset = guess.inverse().compose(pose).translation
        inside_ellipse = (offset[0] / self.config.ellipsis_size_s) ** 2 + (offset[1] / self.config.ellipsis_size_d) ** 2 < 1
        status = "ok" if result.converged and inside_ellipse else "registration failed"
        twist = self._twist(pose, stamp_ns) if status == "ok" else np.zeros(6)
        if status == "ok":
            self.threshold.update(guess, pose)
            if self.config.update_map:
                self.map.update(frame_map.xyz, pose)
        odometry = Odometry(
            stamp_ns=stamp_ns, pose=pose, twist=twist, status=status,
            pose_covariance=np.diag(self.config.pose_covariance_diagonal),
            twist_covariance=np.diag(self.config.twist_covariance_diagonal),
            registration=result, map_points=self.map.num_points, deskewed=deskewed,
        )
        if status == "ok":
            self.previous_previous, self.previous = self.previous, odometry
        return odometry
