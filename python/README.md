# FAR-LIO Python reference

This is a CPU Python rewrite of the **LiDAR odometry core** of FAR-LIO. It follows the
`OdometryPipeline::register_frame` stage order: optional deskew, crop, voxel sampling,
adaptive correspondence threshold, Cauchy-weighted GICP (or Geman-McClure ICP),
velocity from SE(3) poses, and a bounded voxel map. It reads the `LidarOdometry`
portion of the repository's ROS YAML file. The modules are plain NumPy/SciPy, so
you can inspect or change the geometry without ROS or CUDA.

The original `StateEstimation` 3D-EKF, ROS 2 communication, CUDA kernels, asynchronous
map replacement, and real-time performance are **not** ported. An externally fused
pose can be passed as `initial_guess` to `OdometryPipeline.register_frame(...)`; without
one, the pipeline uses the original constant-displacement prediction. This is a
research reference for offline runs, not a replacement for the deployed LIO stack.

## Install and run

To run a self-contained four-scan demo with known ground-truth motion:

```bash
python -m pip install -e ./python
python python/examples/synthetic_demo.py --output-dir ./python-demo
```

The script writes generated scans, `result/trajectory.csv`, and `result/map.npz`,
and prints translation/rotation errors against the known poses. Choose a new
`--output-dir` for another run.

For your own scans:

```bash
python -m pip install -e ./python
far-lio-python \
  --input-dir ./frames \
  --output-dir ./python-run \
  --config-file ./config/far-lio.yml \
  --max-time-ms 0
```

Each file in `frames/` is named by its **integer nanosecond timestamp** (for example,
`1234567890000000000.npz`). An `.npy` file holds an `(N, 3)` XYZ array. An `.npz`
file holds `xyz` and optionally `time_offset_s`, a length-`N` vector measured relative
to that frame's timestamp. These points must be in the vehicle/base frame already.
The output contains `trajectory.csv` and `map.npz` (with an `xyz` array). Existing
nonempty output directories require `--overwrite`.

Optional `--guesses-csv` and `--deskew-poses-csv` files have the header:

```text
timestamp_ns,tx,ty,tz,qx,qy,qz,qw
```

Guesses are exact per-frame matches; missing entries use constant-displacement
prediction. Deskew samples are world-from-base poses. Deskew occurs only with point
time offsets, adequate pose history, and sufficient motion, as in the C++ path.
For localization, pass `--initial-map map.npz` and configure
`pipeline.update_map: false` in the YAML. `--method icp` selects point-to-point ICP.

If using the repository YAML, the CPU loop observes its `registration.max_time`
limit in milliseconds. The example overrides it with `--max-time-ms 0`, which
disables the limit for offline use because Python cannot meet the original CUDA
budget. The default Python configuration also has no wall-clock limit.

## API

```python
from far_lio_python import OdometryPipeline, PipelineConfig, Pose

pipeline = OdometryPipeline(PipelineConfig())
odom = pipeline.register_frame(xyz, stamp_ns, initial_guess=Pose.identity())
print(odom.status, odom.pose.as_matrix())
```

The reference rejects a nonconverged registration before inserting points in the
map; the C++ pipeline currently inserts if `update_map` is enabled regardless of
its status. The Python adaptive threshold also updates its previous guess every
frame. These are intentional protections for offline replay and are called out
because they can change a failed-frame trajectory relative to C++.
