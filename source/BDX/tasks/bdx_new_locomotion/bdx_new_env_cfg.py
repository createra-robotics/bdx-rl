# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from BDX.robots.bdx import BDX_NEW_CFG  # isort:skip
from BDX.tasks.bdx_locomotion.bdx_env_cfg import BdxrEnvCfg, BdxrEnvCfg_PLAY  # isort:skip

BDX_NEW_BASE_HEIGHT = 0.18717
BDX_NEW_FOOT_BODY_REGEX = ".*ankle_pitch.*"


def _apply_bdx_new_urdf_overrides(env_cfg: BdxrEnvCfg) -> None:
    """Apply body-name and asset overrides required by the bdx-new URDF."""
    env_cfg.scene.robot = BDX_NEW_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

    env_cfg.observations.critic.feet_contact_forces.params["asset_cfg"].body_names = [BDX_NEW_FOOT_BODY_REGEX]

    env_cfg.rewards.air_time.params["sensor_cfg"].body_names = BDX_NEW_FOOT_BODY_REGEX
    env_cfg.rewards.contact_pattern.params["sensor_cfg"].body_names = BDX_NEW_FOOT_BODY_REGEX
    env_cfg.rewards.foot_clearance.params["asset_cfg"].body_names = BDX_NEW_FOOT_BODY_REGEX
    env_cfg.rewards.foot_slip.params["asset_cfg"].body_names = BDX_NEW_FOOT_BODY_REGEX
    env_cfg.rewards.foot_slip.params["sensor_cfg"].body_names = BDX_NEW_FOOT_BODY_REGEX
    env_cfg.rewards.base_height_deviation.params["target_height"] = BDX_NEW_BASE_HEIGHT

    env_cfg.events.physics_material.params["asset_cfg"].body_names = BDX_NEW_FOOT_BODY_REGEX


@configclass
class BdxNewEnvCfg(BdxrEnvCfg):
    """Velocity locomotion task for the bdx-new URDF."""

    def __post_init__(self):
        super().__post_init__()
        _apply_bdx_new_urdf_overrides(self)


@configclass
class BdxNewEnvCfg_PLAY(BdxrEnvCfg_PLAY):
    """Play configuration for the bdx-new velocity locomotion task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _apply_bdx_new_urdf_overrides(self)
