from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from far_lio_python import OdometryPipeline, PipelineConfig, Pose
from far_lio_python.cli import Arguments, run_pipeline
from far_lio_python.deskew import PoseHistory
from far_lio_python.pointcloud import PointCloud, preprocess, voxel_downsample
from far_lio_python.registration import RegistrationConfig
from far_lio_python.voxel_map import VoxelMap


def synthetic_points() -> np.ndarray:
    return np.random.default_rng(15).uniform(-4, 4, (600, 3))


class GeometryTests(unittest.TestCase):
    def test_se3_exp_log_and_inverse(self) -> None:
        twist = np.array([1.0, -0.5, 0.3, 0.08, -0.09, 0.1])
        pose = Pose.exp(twist)
        np.testing.assert_allclose(pose.log(), twist, atol=1e-12)
        np.testing.assert_allclose(pose.compose(pose.inverse()).as_matrix(), np.eye(4), atol=1e-12)

    def test_voxel_sampling_uses_floor_and_first_point(self) -> None:
        cloud = PointCloud(xyz=np.array([[-0.1, 0, 0], [-0.8, 0, 0], [0.1, 0, 0]]))
        result = voxel_downsample(cloud, 1.0)
        np.testing.assert_array_equal(result.xyz, cloud.xyz[[0, 2]])

    def test_crop_footprint_and_range(self) -> None:
        cloud = PointCloud(xyz=np.array([[0.1, 0.1, 0], [2.0, 0.1, 0], [20, 0, 0]]))
        result = preprocess(cloud, crop_range=(0.0, 10.0), crop_footprint=(1.0, 0.5))
        np.testing.assert_array_equal(result.xyz, cloud.xyz[[1]])


class OdometryTests(unittest.TestCase):
    def test_gicp_recovers_pose(self) -> None:
        world = synthetic_points()
        truth = Pose(
            rotation=Rotation.from_euler("xyz", [0.02, -0.03, 0.06]),
            translation=np.array([0.4, -0.2, 0.1]),
        )
        frame = truth.inverse().transform(world)
        pipeline = OdometryPipeline(PipelineConfig(voxel_size=1.0, max_distance=20.0))
        pipeline.load_map(world)
        output = pipeline.register_frame(frame, 1_000_000_000)
        self.assertEqual(output.status, "ok")
        self.assertGreater(output.registration.correspondences, 100)
        np.testing.assert_allclose(output.pose.as_matrix(), truth.as_matrix(), atol=0.02)

    def test_icp_recovers_pose(self) -> None:
        world = synthetic_points()
        truth = Pose(translation=np.array([0.1, 0.15, -0.05]))
        pipeline = OdometryPipeline(
            PipelineConfig(
                voxel_size=1.0,
                registration=RegistrationConfig(method="icp"),
            )
        )
        pipeline.load_map(world)
        output = pipeline.register_frame(truth.inverse().transform(world), 1_000_000_000)
        self.assertEqual(output.status, "ok")
        np.testing.assert_allclose(output.pose.translation, truth.translation, atol=0.02)

    def test_gicp_tolerates_noise_and_unmatched_points(self) -> None:
        rng = np.random.default_rng(99)
        world = rng.uniform(-8, 8, (800, 3))
        truth = Pose(
            rotation=Rotation.from_euler("xyz", [0.01, 0.02, -0.05]),
            translation=np.array([0.35, -0.15, 0.08]),
        )
        frame = truth.inverse().transform(world) + rng.normal(0, 0.008, world.shape)
        frame[:80] = rng.uniform(-50, 50, (80, 3))
        pipeline = OdometryPipeline(PipelineConfig(voxel_size=1.5, max_distance=20.0))
        pipeline.load_map(world)
        output = pipeline.register_frame(frame, 1_000_000_000)
        self.assertEqual(output.status, "ok")
        np.testing.assert_allclose(output.pose.as_matrix(), truth.as_matrix(), atol=0.025)

    def test_stationary_frame_and_bad_frame_does_not_corrupt_map(self) -> None:
        world = synthetic_points()
        pipeline = OdometryPipeline(PipelineConfig(voxel_size=1.0, max_distance=20.0))
        initial = pipeline.register_frame(world, 1_000_000_000)
        self.assertEqual(initial.status, "ok")
        stationary = pipeline.register_frame(world, 1_100_000_000)
        self.assertEqual(stationary.status, "ok")
        count = pipeline.map.num_points
        bad = pipeline.register_frame(world + 500, 1_200_000_000)
        self.assertEqual(bad.status, "registration failed")
        self.assertEqual(pipeline.map.num_points, count)
        self.assertEqual(pipeline.previous.stamp_ns, 1_100_000_000)

    def test_deskew_constant_velocity(self) -> None:
        history = PoseHistory()
        history.add(0, Pose.identity())
        history.add(100_000_000, Pose(translation=np.array([0.4, 0, 0])))
        cloud = PointCloud(
            xyz=np.array([[10, 0, 0], [9.6, 0, 0]]),
            time_offset_s=np.array([-0.1, 0.0]),
        )
        corrected, valid = history.deskew(cloud, 100_000_000)
        self.assertTrue(valid)
        np.testing.assert_allclose(corrected.xyz, np.array([[9.6, 0, 0], [9.6, 0, 0]]))

    def test_local_map_prunes_old_voxels(self) -> None:
        voxel_map = VoxelMap(voxel_size=1.0, max_distance=3.0)
        voxel_map.add_points(np.array([[0, 0, 0], [10, 0, 0]]))
        voxel_map.add_points(np.empty((0, 3)), origin=np.array([10, 0, 0]))
        np.testing.assert_array_equal(voxel_map.arrays()[0], np.array([[10, 0, 0]]))


class CliTests(unittest.TestCase):
    def test_ros_yaml_and_offline_output(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        import yaml

        parameters = yaml.safe_load((repository / "config/far-lio.yml").read_text())
        config = PipelineConfig.from_ros_yaml(parameters)
        self.assertEqual(config.voxel_size, 4.0)
        self.assertEqual(config.registration.solver_type, "GaussNewton")
        self.assertIsNone(config.crop_range)  # The source YAML has only one entry.

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            xyz = synthetic_points()
            np.save(frames / "1000000000.npy", xyz)
            np.savez(frames / "1100000000.npz", xyz=xyz)
            trajectory = run_pipeline(Arguments(input_dir=frames, output_dir=root / "out"))
            with trajectory.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
            with np.load(root / "out/map.npz") as data:
                self.assertGreater(len(data["xyz"]), 0)
            with self.assertRaises(FileExistsError):
                run_pipeline(Arguments(input_dir=frames, output_dir=root / "out"))


if __name__ == "__main__":
    unittest.main()
