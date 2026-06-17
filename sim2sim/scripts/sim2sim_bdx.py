#!/usr/bin/env python3
"""Run the BDX ONNX policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SIM2SIM_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = SIM2SIM_DIR / "configs" / "bdx.yaml"
DEFAULT_XML = SIM2SIM_DIR / "xmls" / "scene.xml"

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

DEFAULT_JOINT_POS = np.zeros(len(JOINT_NAMES), dtype=np.float32)
ACTION_SCALE = 0.5
SIM_DT = 0.005
POLICY_RATE = 50.0

KP_DEFAULT = 80.0
KD_DEFAULT = 10.0
KP_ANKLE = 40.0
KD_ANKLE = 2.0

EFFORT_LIMIT_DEFAULT = 42.0
EFFORT_LIMIT_ANKLE = 11.9
ARMATURE_DEFAULT = 0.02
ARMATURE_ANKLE = 0.0042

OBS_DIM = 39
SLIDING_FRICTION = 0.8
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.0001


def _cfg(config: dict[str, Any], path: str, default: Any) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _pick(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _resolve_config_path(path_value: str | Path, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        config_path.parent / path,
        SIM2SIM_DIR / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (SIM2SIM_DIR / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping at top level: {config_path}")
    return config


def apply_config(config: dict[str, Any]) -> None:
    global JOINT_NAMES
    global DEFAULT_JOINT_POS
    global ACTION_SCALE
    global KP_DEFAULT, KD_DEFAULT, KP_ANKLE, KD_ANKLE
    global EFFORT_LIMIT_DEFAULT, EFFORT_LIMIT_ANKLE, ARMATURE_DEFAULT, ARMATURE_ANKLE

    JOINT_NAMES = [str(name) for name in _cfg(config, "robot.joint_names", JOINT_NAMES)]
    default_joint_pos = _cfg(config, "robot.default_joint_pos", [0.0] * len(JOINT_NAMES))
    if len(default_joint_pos) != len(JOINT_NAMES):
        raise ValueError(
            f"robot.default_joint_pos has {len(default_joint_pos)} entries, expected {len(JOINT_NAMES)}"
        )
    DEFAULT_JOINT_POS = np.asarray(default_joint_pos, dtype=np.float32)

    ACTION_SCALE = float(_cfg(config, "control.action_scale", ACTION_SCALE))
    KP_DEFAULT = float(_cfg(config, "actuators.kp_default", KP_DEFAULT))
    KD_DEFAULT = float(_cfg(config, "actuators.kd_default", KD_DEFAULT))
    KP_ANKLE = float(_cfg(config, "actuators.kp_ankle", KP_ANKLE))
    KD_ANKLE = float(_cfg(config, "actuators.kd_ankle", KD_ANKLE))
    EFFORT_LIMIT_DEFAULT = float(_cfg(config, "actuators.effort_limit_default", EFFORT_LIMIT_DEFAULT))
    EFFORT_LIMIT_ANKLE = float(_cfg(config, "actuators.effort_limit_ankle", EFFORT_LIMIT_ANKLE))
    ARMATURE_DEFAULT = float(_cfg(config, "actuators.armature_default", ARMATURE_DEFAULT))
    ARMATURE_ANKLE = float(_cfg(config, "actuators.armature_ankle", ARMATURE_ANKLE))


def resolve_runtime_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    xml_value = _pick(args.xml, _cfg(config, "paths.xml", None), DEFAULT_XML)
    onnx_value = _pick(args.onnx, _cfg(config, "paths.onnx", None), None)
    if onnx_value is None:
        raise ValueError("ONNX path must be set either by --onnx or paths.onnx in the YAML config.")

    args.xml = _resolve_config_path(xml_value, args.config)
    args.onnx = _resolve_config_path(onnx_value, args.config)

    command = _pick(args.cmd, _cfg(config, "command", None), [0.3, 0.0, 0.0])
    if len(command) != 3:
        raise ValueError(f"command must contain exactly 3 values, got {command}")
    args.cmd = [float(value) for value in command]

    args.duration = float(_pick(args.duration, _cfg(config, "simulation.duration", None), 0.0))
    args.sim_dt = float(_pick(args.sim_dt, _cfg(config, "simulation.sim_dt", None), SIM_DT))
    args.policy_rate = float(_pick(args.policy_rate, _cfg(config, "simulation.policy_rate", None), POLICY_RATE))
    args.base_height = float(_pick(args.base_height, _cfg(config, "simulation.base_height", None), 0.33))
    args.fall_height = float(_pick(args.fall_height, _cfg(config, "simulation.fall_height", None), 0.18))
    args.reset_on_fall = bool(_pick(args.reset_on_fall, _cfg(config, "simulation.reset_on_fall", None), False))
    args.print_every = float(_pick(args.print_every, _cfg(config, "simulation.print_every", None), 1.0))
    args.real_time = bool(_pick(args.real_time, _cfg(config, "simulation.real_time", None), True))
    viewer_enabled = bool(_cfg(config, "simulation.viewer", True))
    args.no_viewer = bool(_pick(args.no_viewer, not viewer_enabled, False))

    args.action_clip = float(_pick(args.action_clip, _cfg(config, "control.action_clip", None), 100.0))
    args.foot_friction = float(_pick(args.foot_friction, _cfg(config, "contact.foot_friction", None), SLIDING_FRICTION))
    args.floor_friction = float(
        _pick(args.floor_friction, _cfg(config, "contact.floor_friction", None), SLIDING_FRICTION)
    )
    args.torsional_friction = float(
        _pick(args.torsional_friction, _cfg(config, "contact.torsional_friction", None), TORSIONAL_FRICTION)
    )
    args.rolling_friction = float(
        _pick(args.rolling_friction, _cfg(config, "contact.rolling_friction", None), ROLLING_FRICTION)
    )
    return args


class OnnxPolicy:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        actions = self.session.run([self.output_name], {self.input_name: obs.astype(np.float32)})[0]
        return np.asarray(actions, dtype=np.float32)


def _name_to_joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
    return joint_id


def _name_to_actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"Actuator not found in MuJoCo model: {actuator_name}")
    return actuator_id


def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise ValueError(f"Sensor not found in MuJoCo model: {sensor_name}")
    start = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[start : start + dim].copy()


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]) -> np.ndarray:
    return np.array([data.qpos[model.jnt_qposadr[joint_id]] for joint_id in joint_ids], dtype=np.float32)


def _joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]) -> np.ndarray:
    return np.array([data.qvel[model.jnt_dofadr[joint_id]] for joint_id in joint_ids], dtype=np.float32)


def _joint_ranges(model: mujoco.MjModel, joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([model.jnt_range[joint_id, 0] for joint_id in joint_ids], dtype=np.float32)
    upper = np.array([model.jnt_range[joint_id, 1] for joint_id in joint_ids], dtype=np.float32)
    return lower, upper


def _projected_gravity_from_site(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Site not found in MuJoCo model: {site_name}")
    site_xmat = data.site_xmat[site_id].reshape(3, 3)
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return site_xmat.T @ gravity_world


def _free_joint_qpos_addr(model: mujoco.MjModel) -> int | None:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[joint_id])
    return None


def configure_actuators(
    model: mujoco.MjModel,
    joint_ids: list[int],
    actuator_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp_values = np.zeros(len(joint_ids), dtype=np.float32)
    kd_values = np.zeros(len(joint_ids), dtype=np.float32)
    effort_limits = np.zeros(len(joint_ids), dtype=np.float32)

    for joint_name, joint_id, actuator_id in zip(JOINT_NAMES, joint_ids, actuator_ids):
        is_ankle = "Ankle" in joint_name
        kp = KP_ANKLE if is_ankle else KP_DEFAULT
        kd = KD_ANKLE if is_ankle else KD_DEFAULT
        effort_limit = EFFORT_LIMIT_ANKLE if is_ankle else EFFORT_LIMIT_DEFAULT
        armature = ARMATURE_ANKLE if is_ankle else ARMATURE_DEFAULT

        joint_index = JOINT_NAMES.index(joint_name)
        kp_values[joint_index] = kp
        kd_values[joint_index] = kd
        effort_limits[joint_index] = effort_limit

        dof_id = model.jnt_dofadr[joint_id]
        model.dof_damping[dof_id] = 0.0
        model.dof_armature[dof_id] = armature

        # Isaac DelayedPDActuatorCfg computes explicit effort and clips the total PD torque.
        # Configure the compiled MuJoCo position actuators as direct torque motors.
        model.actuator_gainprm[actuator_id] = 0.0
        model.actuator_gainprm[actuator_id, 0] = 1.0
        model.actuator_biasprm[actuator_id] = 0.0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [-effort_limit, effort_limit]
        model.actuator_ctrllimited[actuator_id] = 1
        model.actuator_ctrlrange[actuator_id] = [-effort_limit, effort_limit]

    return kp_values, kd_values, effort_limits


def configure_contact_properties(
    model: mujoco.MjModel,
    foot_friction: float,
    floor_friction: float,
    torsional_friction: float,
    rolling_friction: float,
) -> None:
    foot_friction_values = [foot_friction, torsional_friction, rolling_friction]
    floor_friction_values = [floor_friction, torsional_friction, rolling_friction]
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name == "floor":
            model.geom_friction[geom_id] = floor_friction_values
        elif geom_name is not None and "foot_collision" in geom_name:
            model.geom_friction[geom_id] = foot_friction_values


def reset_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    base_height: float,
) -> None:
    mujoco.mj_resetData(model, data)

    free_addr = _free_joint_qpos_addr(model)
    if free_addr is not None:
        data.qpos[free_addr : free_addr + 3] = [0.0, 0.0, base_height]
        data.qpos[free_addr + 3 : free_addr + 7] = [1.0, 0.0, 0.0, 0.0]

    for joint_id, default_pos in zip(joint_ids, DEFAULT_JOINT_POS):
        data.qpos[model.jnt_qposadr[joint_id]] = default_pos
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0

    mujoco.mj_forward(model, data)


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    imu_ang_vel = _sensor_data(model, data, "imu_ang_vel").astype(np.float32) * 0.2
    imu_projected_gravity = _projected_gravity_from_site(model, data, "imu").astype(np.float32)
    joint_pos = _joint_qpos(model, data, joint_ids) - DEFAULT_JOINT_POS
    joint_vel = _joint_qvel(model, data, joint_ids) * 0.05

    obs = np.concatenate(
        [imu_ang_vel, imu_projected_gravity, joint_pos, joint_vel, last_action, command],
        dtype=np.float32,
    )
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Expected obs shape {(OBS_DIM,)}, got {obs.shape}")
    return obs[None, :]


def compute_policy_target(
    policy: OnnxPolicy,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    command: np.ndarray,
    last_action: np.ndarray,
    action_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    obs = build_observation(model, data, joint_ids, last_action, command)
    action = policy(obs)[0]
    action = np.clip(action, -action_clip, action_clip)

    target_joint_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE
    target_joint_pos = np.clip(target_joint_pos, joint_lower, joint_upper)
    return action.astype(np.float32), target_joint_pos.astype(np.float32)


def apply_pd_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    actuator_ids: list[int],
    target_joint_pos: np.ndarray,
    kp_values: np.ndarray,
    kd_values: np.ndarray,
    effort_limits: np.ndarray,
) -> np.ndarray:
    joint_pos = _joint_qpos(model, data, joint_ids)
    joint_vel = _joint_qvel(model, data, joint_ids)
    torques = kp_values * (target_joint_pos - joint_pos) - kd_values * joint_vel
    torques = np.clip(torques, -effort_limits, effort_limits)
    for actuator_id, torque in zip(actuator_ids, torques):
        data.ctrl[actuator_id] = torque
    return torques.astype(np.float32)


def run(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    model.opt.timestep = args.sim_dt

    joint_ids = [_name_to_joint_id(model, joint_name) for joint_name in JOINT_NAMES]
    actuator_ids = [_name_to_actuator_id(model, f"{joint_name}_servo") for joint_name in JOINT_NAMES]
    joint_lower, joint_upper = _joint_ranges(model, joint_ids)

    kp_values, kd_values, effort_limits = configure_actuators(model, joint_ids, actuator_ids)
    configure_contact_properties(
        model,
        foot_friction=args.foot_friction,
        floor_friction=args.floor_friction,
        torsional_friction=args.torsional_friction,
        rolling_friction=args.rolling_friction,
    )
    reset_robot(model, data, joint_ids, args.base_height)

    policy = OnnxPolicy(args.onnx)
    command = np.array(args.cmd, dtype=np.float32)
    last_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    target_joint_pos = DEFAULT_JOINT_POS.copy()
    last_torque = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    policy_decimation = max(1, int(round((1.0 / args.policy_rate) / args.sim_dt)))
    actual_policy_rate = 1.0 / (policy_decimation * args.sim_dt)

    print(f"[INFO] Loaded MuJoCo XML: {args.xml}")
    print(f"[INFO] Loaded ONNX policy: {args.onnx}")
    print(f"[INFO] Loaded config: {args.config}")
    print(f"[INFO] command=[{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}]")
    print(f"[INFO] sim_dt={args.sim_dt:.4f}, policy_rate={actual_policy_rate:.2f} Hz")
    print("[INFO] actuator_mode=explicit torque PD")
    print(
        f"[INFO] foot_friction=[{args.foot_friction:.3f}, {args.torsional_friction:.4f}, "
        f"{args.rolling_friction:.4f}], floor_friction=[{args.floor_friction:.3f}, "
        f"{args.torsional_friction:.4f}, {args.rolling_friction:.4f}]"
    )

    step_count = 0
    last_print_time = 0.0

    def simulation_step() -> None:
        nonlocal last_action, target_joint_pos, last_torque, step_count, last_print_time
        if step_count % policy_decimation == 0:
            last_action, target_joint_pos = compute_policy_target(
                policy=policy,
                model=model,
                data=data,
                joint_ids=joint_ids,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                command=command,
                last_action=last_action,
                action_clip=args.action_clip,
            )

        last_torque = apply_pd_control(
            model=model,
            data=data,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            target_joint_pos=target_joint_pos,
            kp_values=kp_values,
            kd_values=kd_values,
            effort_limits=effort_limits,
        )
        mujoco.mj_step(model, data)
        step_count += 1

        if args.print_every > 0.0 and data.time - last_print_time >= args.print_every:
            qpos = _joint_qpos(model, data, joint_ids)
            print(
                f"[INFO] t={data.time:.2f} base_z={data.qpos[_free_joint_qpos_addr(model) + 2]:.3f} "
                f"action_norm={np.linalg.norm(last_action):.3f} q_norm={np.linalg.norm(qpos):.3f} "
                f"tau_norm={np.linalg.norm(last_torque):.3f}"
            )
            last_print_time = data.time

        free_addr = _free_joint_qpos_addr(model)
        if args.reset_on_fall and free_addr is not None and data.qpos[free_addr + 2] < args.fall_height:
            reset_robot(model, data, joint_ids, args.base_height)
            last_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
            target_joint_pos = DEFAULT_JOINT_POS.copy()
            last_torque = np.zeros(len(JOINT_NAMES), dtype=np.float32)

    if args.no_viewer:
        while args.duration <= 0.0 or data.time < args.duration:
            simulation_step()
        return

    from mujoco import viewer as mujoco_viewer

    with mujoco_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.3]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 30
        viewer.cam.elevation = -20

        while viewer.is_running() and (args.duration <= 0.0 or data.time < args.duration):
            start_time = time.time()
            simulation_step()
            viewer.sync()
            elapsed = time.time() - start_time
            if args.real_time and elapsed < args.sim_dt:
                time.sleep(args.sim_dt - elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BDX ONNX locomotion policy in MuJoCo.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to the sim2sim YAML config.")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to the exported ONNX policy.")
    parser.add_argument("--xml", type=Path, default=None, help="Path to the MuJoCo scene XML.")
    parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--duration", type=float, default=None, help="Run duration in seconds. 0 means until closed.")
    parser.add_argument("--sim-dt", type=float, default=None, help="MuJoCo physics timestep.")
    parser.add_argument("--policy-rate", type=float, default=None, help="Policy inference rate in Hz.")
    parser.add_argument("--base-height", type=float, default=None, help="Initial floating-base height.")
    parser.add_argument(
        "--foot-friction",
        type=float,
        default=None,
        help="MuJoCo sliding friction for foot collision geoms.",
    )
    parser.add_argument(
        "--floor-friction",
        type=float,
        default=None,
        help="MuJoCo sliding friction for the floor geom.",
    )
    parser.add_argument(
        "--torsional-friction",
        type=float,
        default=None,
        help="MuJoCo torsional friction for foot/floor contacts.",
    )
    parser.add_argument(
        "--rolling-friction",
        type=float,
        default=None,
        help="MuJoCo rolling friction for foot/floor contacts.",
    )
    parser.add_argument(
        "--action-clip",
        type=float,
        default=None,
        help="Clip ONNX actions before scaling to joint targets.",
    )
    parser.add_argument("--fall-height", type=float, default=None, help="Reset threshold for --reset-on-fall.")
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset-on-fall",
        action="store_true",
        default=None,
        help="Reset when the floating base falls below --fall-height.",
    )
    reset_group.add_argument(
        "--no-reset-on-fall",
        action="store_false",
        dest="reset_on_fall",
        help="Do not reset when the floating base falls below --fall-height.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=None,
        help="Print status every N seconds. 0 disables printing.",
    )
    real_time_group = parser.add_mutually_exclusive_group()
    real_time_group.add_argument(
        "--real-time",
        action="store_true",
        default=None,
        help="Sleep to keep simulation close to real time.",
    )
    real_time_group.add_argument(
        "--no-real-time",
        action="store_false",
        dest="real_time",
        help="Run as fast as possible.",
    )
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--viewer",
        action="store_false",
        dest="no_viewer",
        default=None,
        help="Show the MuJoCo viewer.",
    )
    viewer_group.add_argument("--no-viewer", action="store_true", dest="no_viewer", help="Run without the MuJoCo viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    config = load_config(args.config)
    apply_config(config)
    args = resolve_runtime_args(args, config)
    run(args)


if __name__ == "__main__":
    main()
