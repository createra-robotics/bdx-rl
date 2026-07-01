# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from BDX.tasks.bdx_locomotion.agents.rsl_rl_ppo_cfg import PPORunnerCfg as BdxPPORunnerCfg  # noqa: I202


@configclass
class PPORunnerCfg(BdxPPORunnerCfg):
    experiment_name = "bdx_new_rough"
    max_iterations = 1500 + 1
    clip_actions = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg.use_mirror_loss = True
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 2.0
