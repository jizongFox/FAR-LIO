"""SE(3) operations using the Sophus convention (translation, rotation) for twists."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def skew(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _left_jacobian(rotvec: NDArray[np.float64]) -> NDArray[np.float64]:
    theta = np.linalg.norm(rotvec)
    omega = skew(rotvec)
    if theta < 1e-6:
        return np.eye(3) + 0.5 * omega + (1.0 / 6.0) * omega @ omega
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta**2 * omega
        + (theta - np.sin(theta)) / theta**3 * omega @ omega
    )


@dataclass(frozen=True, kw_only=True)
class Pose:
    rotation: Rotation = field(default_factory=Rotation.identity)
    translation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation must be a finite 3-vector")
        object.__setattr__(self, "translation", translation.copy())

    @classmethod
    def identity(cls) -> Pose:
        return cls()

    @classmethod
    def exp(cls, twist: NDArray[np.float64]) -> Pose:
        twist = np.asarray(twist, dtype=np.float64)
        if twist.shape != (6,) or not np.isfinite(twist).all():
            raise ValueError("twist must be a finite 6-vector [translation, rotation]")
        return cls(
            rotation=Rotation.from_rotvec(twist[3:]),
            translation=_left_jacobian(twist[3:]) @ twist[:3],
        )

    def log(self) -> NDArray[np.float64]:
        rotvec = self.rotation.as_rotvec()
        return np.r_[np.linalg.solve(_left_jacobian(rotvec), self.translation), rotvec]

    def compose(self, other: Pose) -> Pose:
        return Pose(
            rotation=self.rotation * other.rotation,
            translation=self.rotation.apply(other.translation) + self.translation,
        )

    def inverse(self) -> Pose:
        rotation = self.rotation.inv()
        return Pose(rotation=rotation, translation=-rotation.apply(self.translation))

    def transform(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.rotation.apply(points) + self.translation

    def interpolate(self, other: Pose, fraction: float) -> Pose:
        delta = self.inverse().compose(other)
        return self.compose(Pose.exp(fraction * delta.log()))

    def with_z(self, z: float) -> Pose:
        translation = self.translation.copy()
        translation[2] = z
        return Pose(rotation=self.rotation, translation=translation)

    def as_matrix(self) -> NDArray[np.float64]:
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation.as_matrix()
        matrix[:3, 3] = self.translation
        return matrix
