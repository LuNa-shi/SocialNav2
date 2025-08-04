#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ---------------------------------------------------------------------------- #
# Hybrid ADAPT3R + ResNet Policy
# ---------------------------------------------------------------------------- #
# This file defines the Hybrid Policy. It works by importing necessary
# components from both the Adapt3R policy and the PointNav ResNet baseline
# policy, and then combines them into a new, more powerful network.
# ---------------------------------------------------------------------------- #

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from gym import spaces
from torch import nn as nn

from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import NetPolicy

if TYPE_CHECKING:
    from omegaconf import DictConfig

# 从你的 Adapt3R 实现中导入核心的3D编码器
try:
    from .adapt3r_policy import Adapt3REncoder, weight_init
except (ImportError, ModuleNotFoundError):
    raise ImportError("Could not import Adapt3REncoder. Make sure adapt3r_policy.py is accessible.")

# 从 baseline 中导入核心的2D网络和策略
try:
    from .resnet_policy import PointNavResNetNet, PointNavResNetPolicy
except (ImportError, ModuleNotFoundError):
    raise ImportError("Could not import PointNavResNetNet. Make sure pointnav_resnet_policy.py is accessible.")


# ############################################################################ #
# 2. 定义混合网络 (Hybrid Network)
#    这个网络继承自 PointNavResNetNet 以复用其权重和逻辑
# ############################################################################ #

class HybridAdapt3RNet(PointNavResNetNet):
    """
    一个混合网络，它在PointNavResNetNet的基础上增加了Adapt3REncoder。
    通过继承，我们可以轻松地加载预训练的 PointNav ResNet 权重。
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        config: "DictConfig", # 接收完整的 habitat_baselines 配置
    ):
        # 从配置中提取 baseline 网络所需的参数
        policy_config = config.rl.policy
        ddppo_config = config.rl.ddppo
        
        # 1. 首先，调用父类 (PointNavResNetNet) 的构造函数。
        #    这将初始化所有的 baseline 组件（2D ResNet, 目标编码器等）。
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.rl.ppo.hidden_size,
            num_recurrent_layers=ddppo_config.num_recurrent_layers,
            rnn_type=ddppo_config.rnn_type,
            backbone=ddppo_config.backbone,
            resnet_baseplanes=ddppo_config.resnet_baseplanes,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            fuse_keys=None,  # 使用 baseline 的默认逻辑
            force_blind_policy=config.force_blind_policy,
            discrete_actions=(policy_config.action_distribution_type == "categorical"),
        )
        
        # 2. 现在，初始化我们新增的 Adapt3R 点云编码器。
        #    我们假设 Adapt3R 的特定配置位于 `rl.ddppo.adapt3r`。
        adapt3r_config = ddppo_config.adapt3r
        self.pointcloud_encoder = Adapt3REncoder(
            observation_space=observation_space, 
            **adapt3r_config.visual_encoder
        )
        

        # 3. 调整 RNN 的输入维度。
        #    原始的 RNN 输入大小已在父类的构造函数中计算完成。
        #    我们只需在它的基础上增加 Adapt3R 编码器的输出维度。
        original_rnn_input_size = self.state_encoder.input_size
        
        new_rnn_input_size = original_rnn_input_size
        if not self.pointcloud_encoder.is_blind:
            # 增加3D感知特征的维度
            new_rnn_input_size += self.pointcloud_encoder.d_out_perception
        if self.pointcloud_encoder.d_out_lowdim > 0:
            # 增加3D编码器处理的低维特征的维度
            new_rnn_input_size += self.pointcloud_encoder.d_out_lowdim
            
        # 4. 基于新的输入维度，重新构建 RNN (state_encoder)。
        self.state_encoder = build_rnn_state_encoder(
            input_size=new_rnn_input_size,
            hidden_size=self._hidden_size,
            rnn_type=ddppo_config.rnn_type,
            num_layers=ddppo_config.num_recurrent_layers,
        )

        print(f"HybridNet initialized. Original RNN input size: {original_rnn_input_size}, New RNN input size: {new_rnn_input_size}")
        self.train()

        # Freeze the visual encoder and visual fc ,in the beginning of training
        self.visual_encoder.eval()
        self.visual_fc.eval()

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.visual_fc.parameters():
            param.requires_grad = False

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 1. Get features from the ResNet baseline part of the model
        x_baseline_parts = []
        aux_loss_state = {}

        # Process visual features from ResNet
        if not self.visual_encoder.is_blind:
            visual_feats = self.visual_fc(self.visual_encoder(observations))
            aux_loss_state["perception_embed_2d"] = visual_feats
            x_baseline_parts.append(visual_feats)
        
        # Process non-visual sensor data (e.g., GPS, Compass) using parent's logic
        # This replicates the logic from PointNavResNetNet.forward()
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_obs = observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
            if goal_obs.shape[1] == 2:
                goal_obs = torch.stack([goal_obs[:, 0], torch.cos(-goal_obs[:, 1]), torch.sin(-goal_obs[:, 1])], -1)
            x_baseline_parts.append(self.tgt_embeding(goal_obs))

        if PointGoalSensor.cls_uuid in observations:
            x_baseline_parts.append(self.pointgoal_embedding(observations[PointGoalSensor.cls_uuid]))
        

        # Process previous action
        if self.discrete_actions:
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions.squeeze(-1) + 1, torch.zeros_like(prev_actions.squeeze(-1)))
            )
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())
        x_baseline_parts.append(prev_actions)
        
        # Concatenate all baseline features
        x_baseline = torch.cat(x_baseline_parts, dim=1)

        # 2. Get features from our new 3D point cloud encoder
        perception_3d, lowdim_3d = self.pointcloud_encoder(observations)
        
        # 3. Concatenate baseline features with 3D features
        final_x_parts = [x_baseline]
        if not self.pointcloud_encoder.is_blind:
            final_x_parts.append(perception_3d)
            aux_loss_state["perception_embed_3d"] = perception_3d
        if lowdim_3d is not None:
            final_x_parts.append(lowdim_3d)

        # 4. Feed the combined features into the RNN
        out_full = torch.cat(final_x_parts, dim=1)
        out_rnn, rnn_hidden_states = self.state_encoder(
            out_full, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out_rnn

        return out_rnn, rnn_hidden_states, aux_loss_state

# ############################################################################ #
# 3. 定义混合策略 (Hybrid Policy)
# ############################################################################ #

@baseline_registry.register_policy
class HybridAdapt3RPolicy(NetPolicy):
    """
    一种混合策略，它使用 HybridAdapt3RNet 作为其网络。
    此类负责从配置文件实例化网络，并将其包装在 NetPolicy 中。
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        policy_config: "DictConfig" = None,
        config: Optional["DictConfig"] = None,
        **kwargs,
    ):
        # 过滤掉不应被策略看到的渲染相机等传感器
        filtered_obs = {
            k: v for k, v in observation_space.spaces.items() 
            if "render_cam" not in k and "panoramic" not in k
        }
        filtered_obs = spaces.Dict(filtered_obs)
        
        super().__init__(
            net=HybridAdapt3RNet(
                observation_space=filtered_obs,
                action_space=action_space,
                config=config,  # 传递完整的 habitat_baselines 配置
            ),
            action_space=action_space,
            policy_config=policy_config,
        )
        # 为 Adapt3R 部分的层应用权重初始化
        self.net.pointcloud_encoder.apply(weight_init)

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.habitat_baselines.rl.policy,
            config=config.habitat_baselines, # 传递 habitat_baselines 部分的配置
            **kwargs
        )