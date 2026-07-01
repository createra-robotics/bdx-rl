# BDX Policy Deployment on a ROS Robot

This document is for the engineer or coding agent deploying a policy that has already passed
`sim2sim/scripts/sim2sim_bdx.py`. Treat `sim2sim_bdx.py` as the executable specification for the policy interface.
The robot-side ROS node must reproduce the same observation layout, action scaling, joint order, control rate, and
safety limits before the policy is allowed to command the hardware.

## Scope

The deployment target is a robot computer running ROS. The exact ROS version and actuator interface may differ between
robots, so this document describes the required node behavior and the checks that must be completed. Do not assume topic
names, frame names, motor modes, or safety services. Discover them on the robot and make them ROS parameters.

The policy must not be launched as part of the robot's default startup until it has passed the dry-run and suspended
hardware checks below.

## Files to Transfer

Copy these artifacts from the training/sim2sim machine to the robot computer:

- The exported ONNX policy, for example `sim2sim/onnx/model_2850.onnx`.
- The matching runtime configuration from `sim2sim/configs/bdx.yaml`, or a robot-specific copy of it.
- A record of the exact git commit and any local diff used to train/export the policy.
- A short sim2sim validation note with the tested command range and friction settings.

On the robot computer, keep these files in a versioned deployment directory, for example:

```text
~/bdx_policy_deploy/
  policies/model_2850.onnx
  config/bdx_policy.yaml
  logs/
```

## Runtime Dependencies

Install the ONNX runtime dependency in the same Python environment used by the ROS node:

```bash
python -m pip install onnxruntime numpy
```

Use the ROS workspace style already used on the robot:

- ROS 1: catkin package with a Python node using `rospy`.
- ROS 2: ament/colcon package with a Python node using `rclpy`.

The package should be named something explicit, for example `bdx_policy_deploy`. Its launch file should default to
`dry_run:=true`, where the node computes observations and policy outputs but does not send motor commands.

## Source of Truth: Policy Interface

The policy has a fixed input and output contract:

- Input: one `float32` tensor with shape `(1, 39)`.
- Output: one `float32` tensor with shape `(1, 10)`.
- Policy rate: 50 Hz.
- Low-level control rate: 200 Hz if torque PD is implemented on the robot computer.
- Joint target scale: `target_joint_pos = default_joint_pos + action * 0.5`.
- Hardware deployment action clip: use `action = clip(action, -1.0, 1.0)` before scaling.

Use this joint order everywhere:

```python
JOINT_NAMES = [
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Knee",
    "Right_Knee",
    "Left_Ankle",
    "Right_Ankle",
]
```

Do not rely on the order in incoming `sensor_msgs/JointState`. Build an index map by joint name and reorder every
message into `JOINT_NAMES`.

## Observation Construction

The ROS node must build the policy observation exactly like `sim2sim_bdx.py`:

```python
obs = concat(
    imu_ang_vel * 0.2,          # 3 values
    imu_projected_gravity,      # 3 values
    joint_pos - default_pos,    # 10 values
    joint_vel * 0.05,           # 10 values
    last_action,                # 10 values
    command,                    # 3 values: lin_vel_x, lin_vel_y, ang_vel_z
)
```

The resulting vector must have length 39 and dtype `float32`.

### IMU Requirements

`imu_ang_vel` must be angular velocity in the policy IMU/body frame, in rad/s. If the ROS IMU driver publishes angular
velocity in another frame, transform it into the robot IMU frame used by training before applying the `0.2` scale.

`imu_projected_gravity` is the world gravity direction expressed in the IMU/body frame. Do not use linear acceleration
directly for this term. If the IMU orientation quaternion represents the IMU orientation in the world frame as `R_w_i`,
compute:

```python
gravity_world = [0.0, 0.0, -1.0]
imu_projected_gravity = R_w_i.T @ gravity_world
```

When the robot is upright and level, this should be close to `[0, 0, -1]` in the policy frame. If it is not, fix the
frame transform before testing the policy on hardware.

### Joint State Requirements

For each joint in `JOINT_NAMES`, read:

- position in radians,
- velocity in radians per second.

Then compute:

```python
joint_pos_rel = joint_pos - default_joint_pos
joint_vel_scaled = joint_vel * 0.05
```

The default joint position currently used by `sim2sim/configs/bdx.yaml` is all zeros. If the real robot's calibrated
zero differs from the simulation zero, do not silently change the policy convention. Either fix the hardware zeroing
procedure or create an explicit, reviewed offset in the deployment config and revalidate in sim2sim.

### Command Requirements

The command vector is:

```python
command = [lin_vel_x, lin_vel_y, ang_vel_z]
```

Use a ROS command source such as `/cmd_vel`, but clamp it to the trained range before it enters the observation. Start
hardware tests with `[0.0, 0.0, 0.0]`, then use very small commands only after the standing test is stable.

Recommended initial hardware command clamps:

```yaml
command_limits:
  lin_vel_x: [-0.1, 0.1]
  lin_vel_y: [-0.05, 0.05]
  ang_vel_z: [-0.2, 0.2]
```

Increase these only after tethered testing.

## Action to Motor Command

The policy output is not a torque. It is a normalized joint-position offset.

Every policy tick:

```python
action = onnx_policy(obs)[0]
action = clip(action, -1.0, 1.0)
target_joint_pos = default_joint_pos + action * 0.5
target_joint_pos = clip(target_joint_pos, joint_lower_limits, joint_upper_limits)
last_action = action
```

If the robot exposes position-control motors, send `target_joint_pos` to the low-level controller and configure that
controller to use gains equivalent to the sim2sim configuration.

If the robot exposes torque-control motors, hold the latest `target_joint_pos` and run explicit PD at 200 Hz:

```python
torque = kp * (target_joint_pos - joint_pos) - kd * joint_vel
torque = clip(torque, -effort_limit, effort_limit)
```

The sim2sim BDX defaults are:

```yaml
kp_default: 80.0
kd_default: 10.0
kp_ankle: 40.0
kd_ankle: 2.0
effort_limit_default: 42.0
effort_limit_ankle: 11.9
action_scale: 0.5
```

If deploying a different robot variant, do not guess these values. Read the variant-specific gains and effort limits
from the matching config and repeat sim2sim validation with those values.

## ROS Node Design

The node should have these responsibilities:

1. Load the ONNX policy once at startup.
2. Subscribe to joint states, IMU, command velocity, and safety state.
3. Maintain a name-based joint state buffer ordered by `JOINT_NAMES`.
4. Run policy inference at 50 Hz.
5. Run motor command output at the required low-level rate, or publish position targets at the motor controller's
   expected rate.
6. Enforce all safety gates before publishing any actuator command.
7. Publish diagnostics for observation, action, target position, torque or target command, and safety state.
8. Support `dry_run` mode that performs all computation but publishes no motor command.

Suggested parameters:

```yaml
policy_path: /home/robot/bdx_policy_deploy/policies/model_2850.onnx
dry_run: true
policy_rate_hz: 50.0
control_rate_hz: 200.0
action_clip: 1.0
action_scale: 0.5
cmd_timeout_s: 0.2
sensor_timeout_s: 0.05
max_abs_roll_pitch_rad: 0.7
max_joint_velocity_rad_s: 25.0
torque_ramp_time_s: 2.0
publish_debug: true
```

Topic names must be robot-specific parameters. Do not hard-code them in the policy logic.

## Required Safety Gates

The deployment node must refuse to command motors when any of these checks fail:

- E-stop is active or the robot is not in the expected enable mode.
- `dry_run` is true.
- Joint state message is stale.
- IMU message is stale.
- Command message is stale. On timeout, command must become `[0, 0, 0]`.
- Any observation, action, target, or torque contains NaN or Inf.
- Joint position is outside a configured safe range.
- Joint velocity is outside a configured safe range.
- Base roll or pitch exceeds the configured limit.
- Policy output changes too fast for the configured action-rate limit.
- The requested target position exceeds joint limits.

On any safety fault:

1. Stop publishing walking commands.
2. Switch to a safe hold or motor-disable behavior defined by the robot platform.
3. Publish the fault reason.
4. Require an explicit operator reset before re-enabling policy commands.

The E-stop must bypass this node. A software safety gate is not a substitute for a physical or low-level E-stop.

## Bring-up Procedure

Follow this sequence. Do not skip steps.

1. Export and freeze the policy.
   - Export the checkpoint to ONNX.
   - Record the checkpoint path, git state, and ONNX checksum.
   - Run `sim2sim_bdx.py` with the exact ONNX file and command range intended for robot testing.

2. Build the ROS node in dry-run mode.
   - Subscribe to real robot joint states and IMU.
   - Build the 39D observation.
   - Run ONNX inference.
   - Publish debug messages only, no motor commands.

3. Validate the dry-run observation.
   - Robot upright: `imu_projected_gravity` is close to `[0, 0, -1]`.
   - Joint order matches `JOINT_NAMES`.
   - Joint positions and velocities have the correct sign.
   - With zero command, action values are finite and clipped to `[-1, 1]`.
   - Target joint positions remain inside joint limits.

4. Validate command handling.
   - Send `[0, 0, 0]`, small forward command, and small yaw command.
   - Confirm the command appears in observation indices 36 to 38.
   - Confirm command timeout returns to zero.

5. Suspended or unloaded motor test.
   - Keep the robot supported so it cannot fall.
   - Enable motors with reduced torque/current limits if the hardware supports it.
   - Start with zero command.
   - Verify joint target direction for each motor before allowing ground contact.

6. Ground contact standing test.
   - Use a tether or support.
   - Use zero command only.
   - Start with a torque ramp.
   - Confirm the robot does not fight the operator or drive joints into limits.

7. First walking tests.
   - Start with very small commands, for example `lin_vel_x = 0.03` to `0.05`.
   - Keep lateral and yaw commands at zero initially.
   - Record rosbag logs for every attempt.
   - Stop immediately on foot scuffing, joint limit hits, estimator frame errors, or repeated safety faults.

8. Expand the command envelope.
   - Increase one command dimension at a time.
   - Compare action norm, target joint positions, joint velocities, and motor effort with sim2sim logs.
   - Keep the final command limits lower than training limits until repeated hardware runs are stable.

## Debug Data to Record

Record these topics or equivalent debug streams:

- Raw `/joint_states`.
- Raw IMU.
- Command input.
- Reordered joint positions and velocities.
- Full 39D policy observation.
- Raw ONNX action.
- Clipped action.
- Target joint positions.
- Published motor command or torque.
- Safety state and fault reason.
- Robot enable state and E-stop state.

For every hardware run, save:

```text
date_time/
  policy.onnx
  bdx_policy.yaml
  rosbag
  deployment_git_commit.txt
  notes.md
```

## Common Failure Modes

Wrong joint order:
The policy may look stable in logs but command the wrong leg or joint. Always reorder by joint name.

Wrong IMU frame:
If `imu_projected_gravity` is not close to `[0, 0, -1]` when upright, the policy will behave as if the robot is tilted.

Using acceleration as gravity:
Linear acceleration includes motion and vibration. Use the orientation estimate to compute projected gravity.

Missing action clipping:
Unbounded policy output can create huge target jumps. Clip action to `[-1, 1]` before scaling.

Wrong motor mode:
The policy output is a position offset, not torque. If torque mode is used, explicit PD and torque clipping are required.

Silent hardware zero mismatch:
If motor zero differs from simulation zero, the policy will stand in the wrong pose. Fix calibration or explicitly
document and validate the offset.

## Acceptance Criteria Before Untethered Testing

Do not attempt untethered testing until all of the following are true:

- The same ONNX file passes sim2sim.
- The ROS node has run in dry-run mode on the real robot without NaN, stale sensor, or frame faults.
- Zero-command suspended test verifies all joint target directions.
- Zero-command ground test is stable under support.
- E-stop and software fault handling have been tested.
- Logs show action, target position, and torque within configured limits.
- A human operator can command zero velocity and disable the policy without restarting the robot computer.
