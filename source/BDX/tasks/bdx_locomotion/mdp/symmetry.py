# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to specify left-right symmetry for the BD-X bipedal robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations and actions with left-right symmetry.

    The BD-X robot has bilateral (left-right) symmetry. This function creates
    augmented versions of observations and actions by applying the left-right
    mirror transformation, doubling the batch size.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        # original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # left-right mirror
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(obs["policy"])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # original
        actions_aug[:batch_size] = actions[:]
        # left-right mirror
        actions_aug[batch_size:] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the policy observation tensor.

    Mirrors IMU readings, projected gravity, joints, actions, and velocity commands.
    """
    obs = obs.clone()
    device = obs.device

    # imu_ang_vel (indices 0:3): angular/axial vector under left-right reflection.
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    # imu_projected_gravity (indices 3:6): flip y
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1.0, -1.0, 1.0], device=device)
    # joint_pos (indices 6:16): swap left-right + flip Hip_Yaw/Hip_Roll
    obs[:, 6:16] = _switch_bdx_joints_left_right(obs[:, 6:16])
    # joint_vel (indices 16:26): swap left-right + flip Hip_Yaw/Hip_Roll
    obs[:, 16:26] = _switch_bdx_joints_left_right(obs[:, 16:26])
    # last_actions (indices 26:36): swap left-right + flip Hip_Yaw/Hip_Roll
    obs[:, 26:36] = _switch_bdx_joints_left_right(obs[:, 26:36])
    # velocity_commands (indices 36:39): [lin_vel_x, lin_vel_y, ang_vel_z]
    # flip lin_vel_y, ang_vel_z
    obs[:, 36:39] = obs[:, 36:39] * torch.tensor([1.0, -1.0, -1.0], device=device)

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Apply left-right symmetry to the action tensor."""
    actions = actions.clone()
    actions[:] = _switch_bdx_joints_left_right(actions[:])
    return actions


def _switch_bdx_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left and right leg joints and flip sign of Hip_Yaw and Hip_Roll.

    BD-X joint ordering from Isaac Lab (10 DOF):
        [Left_Hip_Yaw, Right_Hip_Yaw, Left_Hip_Roll, Right_Hip_Roll,
         Left_Hip_Pitch, Right_Hip_Pitch, Left_Knee, Right_Knee, Left_Ankle, Right_Ankle]

    Left indices:  [0, 2, 4, 6, 8]
    Right indices: [1, 3, 5, 7, 9]

    Hip_Yaw (indices 0, 1) and Hip_Roll (indices 2, 3) change sign under mirroring.
    Hip_Pitch, Knee, and Ankle keep their sign.
    """
    joint_data_switched = torch.zeros_like(joint_data)

    # Swap: left <-> right
    left_idx = [0, 2, 4, 6, 8]
    right_idx = [1, 3, 5, 7, 9]
    joint_data_switched[..., left_idx] = joint_data[..., right_idx]
    joint_data_switched[..., right_idx] = joint_data[..., left_idx]

    # Flip sign of Hip_Yaw and Hip_Roll (now at their new positions after swap)
    joint_data_switched[..., [0, 1]] *= -1.0  # Hip_Yaw
    joint_data_switched[..., [2, 3]] *= -1.0  # Hip_Roll

    return joint_data_switched
