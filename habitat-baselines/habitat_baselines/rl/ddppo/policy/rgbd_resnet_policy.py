#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
RGBD ResNet Policy - Dual Backbone Architecture

This policy extends the PointNavResNetPolicy to support both RGB and depth inputs
using separate ResNet backbones. The RGB backbone uses ImageNet pretrained weights,
while the depth backbone can be initialized from existing checkpoints.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models as torchvision_models

from habitat.tasks.nav.instance_image_nav_task import InstanceImageGoalSensor
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
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions

if TYPE_CHECKING:
    from omegaconf import DictConfig

import torchvision.transforms as T


@baseline_registry.register_policy
class RGBDResNetPolicy(NetPolicy):
    """
    RGBD ResNet Policy with dual backbone architecture.
    
    This policy processes RGB and depth inputs through separate ResNet backbones:
    - RGB backbone: ImageNet pretrained ResNet
    - Depth backbone: Custom ResNet (can load from pretrained checkpoints)
    
    The outputs are concatenated and fed to an RNN state encoder.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        rgb_backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        fuse_keys: Optional[List[str]] = None,
        freeze_depth_backbone: bool = False,
        **kwargs,
    ):
        """
        Initialize RGBD ResNet Policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            hidden_size: RNN hidden size
            num_recurrent_layers: Number of RNN layers
            rnn_type: RNN type (GRU/LSTM)
            resnet_baseplanes: Base planes for depth ResNet
            backbone: Depth backbone architecture
            rgb_backbone: RGB backbone architecture
            normalize_visual_inputs: Whether to normalize inputs
            force_blind_policy: Force blind policy (no visual input)
            policy_config: Policy configuration
            aux_loss_config: Auxiliary loss configuration
            fuse_keys: Keys to fuse in the policy
            freeze_depth_backbone: Whether to freeze the depth backbone
        """
        
        assert backbone in [
            "resnet18",
            "resnet50",
            "resneXt50",
            "se_resnet50",
            "se_resneXt50",
            "se_resneXt101",
        ], f"{backbone} backbone is not recognized."
        
        assert rgb_backbone in [
            "resnet18",
            "resnet50",
        ], f"{rgb_backbone} RGB backbone is not recognized."

        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            RGBDResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                rgb_backbone=rgb_backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                freeze_depth_backbone=freeze_depth_backbone,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        """Create policy from config."""
        # Exclude cameras for rendering from the observation space
        ignore_names = [
            sensor.uuid
            for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
        ]
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = None
        if "agent_name" in kwargs:
            agent_name = kwargs["agent_name"]

        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        # Get RGB backbone from config, default to same as depth backbone
        rgb_backbone = getattr(config.habitat_baselines.rl.ddppo, 'rgb_backbone', 
                              config.habitat_baselines.rl.ddppo.backbone)

        freeze_depth_backbone = getattr(config.habitat_baselines.rl.ddppo, 'freeze_depth_backbone', False)
        # freeze_depth_backbone = True

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            backbone=config.habitat_baselines.rl.ddppo.backbone,
            rgb_backbone=rgb_backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.habitat_baselines.force_blind_policy,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            fuse_keys=None,
            freeze_depth_backbone=freeze_depth_backbone,
        )


class RGBDResNetEncoder(nn.Module):
    """
    RGBD ResNet Encoder with separate RGB and depth backbones.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        rgb_backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        freeze_depth_backbone: bool = False,
    ):
        super().__init__()

        # Separate RGB and depth keys
        self.rgb_keys = [
            k for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and "rgb" in k.lower() and k != ImageGoalSensor.cls_uuid
        ]
        
        self.depth_keys = [
            k for k, v in observation_space.spaces.items()
            if len(v.shape) > 1 and "depth" in k.lower() and k != ImageGoalSensor.cls_uuid
        ]
        
        print(f"RGB keys found: {self.rgb_keys}")
        print(f"Depth keys found: {self.depth_keys}")

        # RGB channels
        self._n_input_rgb_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.rgb_keys
        ) if self.rgb_keys else 0
        
        # Depth channels
        self._n_input_depth_channels = sum(
            observation_space.spaces[k].shape[2] for k in self.depth_keys
        ) if self.depth_keys else 0

        self.key_needs_rescaling = {}
        for k, v in observation_space.spaces.items():
            if k in self.rgb_keys or k in self.depth_keys:
                if v.dtype == np.uint8:
                    self.key_needs_rescaling[k] = 1.0 / v.high.max()
                else:
                    self.key_needs_rescaling[k] = None

        # Initialize RGB backbone (ImageNet pretrained)
        if self._n_input_rgb_channels > 0:
            self.rgb_backbone = self._create_rgb_backbone(rgb_backbone)
            self.imagenet_normalization = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.rgb_backbone = None
            self.rgb_running_mean_and_var = None

        # Initialize depth backbone (custom ResNet)
        if self._n_input_depth_channels > 0:
            self.backbone = make_backbone(
                self._n_input_depth_channels, baseplanes, ngroups
            )
            if freeze_depth_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            if normalize_visual_inputs:
                self.depth_running_mean_and_var = RunningMeanAndVar(self._n_input_depth_channels)
            else:
                self.depth_running_mean_and_var = nn.Sequential()
        else:
            self.backbone = None
            self.depth_running_mean_and_var = None

        if not self.is_blind:
            # Calculate spatial sizes
            if self.rgb_keys:
                spatial_size_h = observation_space.spaces[self.rgb_keys[0]].shape[0] // 2
                spatial_size_w = observation_space.spaces[self.rgb_keys[0]].shape[1] // 2
            elif self.depth_keys:
                spatial_size_h = observation_space.spaces[self.depth_keys[0]].shape[0] // 2
                spatial_size_w = observation_space.spaces[self.depth_keys[0]].shape[1] // 2
            else:
                spatial_size_h = spatial_size_w = spatial_size // 2

            # RGB compression
            if self.rgb_backbone is not None:
                rgb_final_spatial_h = int(np.ceil(spatial_size_h * 0.25))  # ResNet compression factor
                rgb_final_spatial_w = int(np.ceil(spatial_size_w * 0.25))
                rgb_after_compression_flat_size = 2048
                rgb_num_compression_channels = int(
                    round(rgb_after_compression_flat_size / (rgb_final_spatial_h * rgb_final_spatial_w))
                )
                
                self.rgb_compression = nn.Sequential(
                    nn.Conv2d(
                        self._get_rgb_backbone_final_channels(rgb_backbone),
                        rgb_num_compression_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, rgb_num_compression_channels),
                    nn.ReLU(True),
                )
                
                self.rgb_output_shape = (
                    rgb_num_compression_channels,
                    rgb_final_spatial_h,
                    rgb_final_spatial_w,
                )
            else:
                self.rgb_compression = None
                self.rgb_output_shape = None

            # Depth compression
            if self.backbone is not None:
                depth_final_spatial_h = int(np.ceil(spatial_size_h * self.backbone.final_spatial_compress))
                depth_final_spatial_w = int(np.ceil(spatial_size_w * self.backbone.final_spatial_compress))
                depth_after_compression_flat_size = 2048
                depth_num_compression_channels = int(
                    round(depth_after_compression_flat_size / (depth_final_spatial_h * depth_final_spatial_w))
                )
                
                self.compression = nn.Sequential(
                    nn.Conv2d(
                        self.backbone.final_channels,
                        depth_num_compression_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, depth_num_compression_channels),
                    nn.ReLU(True),
                )
                
                self.depth_output_shape = (
                    depth_num_compression_channels,
                    depth_final_spatial_h,
                    depth_final_spatial_w,
                )
            else:
                self.compression = None
                self.depth_output_shape = None

            # Combined output shape
            self.output_shape = self._calculate_combined_output_shape()

    def _create_rgb_backbone(self, backbone_name: str):
        """Create ImageNet pretrained RGB backbone."""
        if backbone_name == "resnet18":
            backbone = torchvision_models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision_models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported RGB backbone: {backbone_name}")
        
        # Remove the final FC layer and global pooling
        return nn.Sequential(*list(backbone.children())[:-2])

    def _get_rgb_backbone_final_channels(self, backbone_name: str) -> int:
        """Get the number of output channels for RGB backbone."""
        if backbone_name == "resnet18":
            return 512
        elif backbone_name == "resnet50":
            return 2048
        else:
            raise ValueError(f"Unsupported RGB backbone: {backbone_name}")

    def _calculate_combined_output_shape(self):
        """Calculate the combined output shape for RGB and depth features."""
        shapes = []
        if self.rgb_output_shape is not None:
            shapes.append(self.rgb_output_shape)
        if self.depth_output_shape is not None:
            shapes.append(self.depth_output_shape)
        
        if not shapes:
            return None
        
        # Concatenate along channel dimension
        total_channels = sum(shape[0] for shape in shapes)
        # Use the spatial dimensions from the first available shape
        spatial_h = shapes[0][1]
        spatial_w = shapes[0][2]
        
        return (total_channels, spatial_h, spatial_w)

    @property
    def is_blind(self):
        return self._n_input_rgb_channels == 0 and self._n_input_depth_channels == 0

    def layer_init(self):
        """Initialize layers."""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through RGBD encoder."""
        if self.is_blind:
            return None

        features = []

        # Process RGB inputs
        if self.rgb_backbone is not None and self.rgb_keys:
            rgb_input = []
            for k in self.rgb_keys:
                obs_k = observations[k]
                # Permute to [BATCH x CHANNEL x HEIGHT x WIDTH]
                obs_k = obs_k.permute(0, 3, 1, 2)
                if self.key_needs_rescaling[k] is not None:
                    obs_k = obs_k.float() * self.key_needs_rescaling[k]
                rgb_input.append(obs_k)
            
            rgb_x = torch.cat(rgb_input, dim=1)
            
            # Ensure input is in correct format for ImageNet normalization (B, C, H, W)
            if rgb_x.shape[1] != 3:  # If not 3 channels
                # If single channel, repeat to make 3 channels
                print(f"- Warning: Unexpected channel dimension {rgb_x.shape[1]}, reshaping...")
                rgb_x = rgb_x.repeat(1, 3, 1, 1)
            
            # Ensure values are in [0,1] before normalization
            if rgb_x.max() > 1.0:
                rgb_x = rgb_x / 255.0
            
            rgb_x = self.imagenet_normalization(rgb_x)
            rgb_x = self.rgb_backbone(rgb_x)
            rgb_x = self.rgb_compression(rgb_x)
            features.append(rgb_x)

        # Process depth inputs
        if self.backbone is not None and self.depth_keys:
            depth_input = []
            for k in self.depth_keys:
                obs_k = observations[k]
                # Permute to [BATCH x CHANNEL x HEIGHT x WIDTH]
                obs_k = obs_k.permute(0, 3, 1, 2)
                if self.key_needs_rescaling[k] is not None:
                    obs_k = obs_k.float() * self.key_needs_rescaling[k]
                depth_input.append(obs_k)
            
            depth_x = torch.cat(depth_input, dim=1)
            depth_x = F.avg_pool2d(depth_x, 2)  # Downsample
            depth_x = self.depth_running_mean_and_var(depth_x)
            depth_x = self.backbone(depth_x)
            depth_x = self.compression(depth_x)
            features.append(depth_x)

        # Concatenate features
        if len(features) == 1:
            return features[0]
        elif len(features) == 2:
            # Ensure spatial dimensions match before concatenation
            rgb_feat, depth_feat = features
            if rgb_feat.shape[2:] != depth_feat.shape[2:]:
                # Resize to match the smaller spatial dimension
                target_h = min(rgb_feat.shape[2], depth_feat.shape[2])
                target_w = min(rgb_feat.shape[3], depth_feat.shape[3])
                rgb_feat = F.adaptive_avg_pool2d(rgb_feat, (target_h, target_w))
                depth_feat = F.adaptive_avg_pool2d(depth_feat, (target_h, target_w))
            
            return torch.cat([rgb_feat, depth_feat], dim=1)
        else:
            raise ValueError("No features to process")


class RGBDResNetNet(Net):
    """
    RGBD ResNet Network that processes RGB and depth through separate backbones.
    """

    PRETRAINED_VISUAL_FEATURES_KEY = "visual_features"
    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone: str,
        rgb_backbone: str,
        resnet_baseplanes: int,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        freeze_depth_backbone: bool = False,
    ):
        super().__init__()
        
        self.discrete_actions = discrete_actions
        self._n_prev_action = 32
        
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(
                num_actions, self._n_prev_action
            )
        
        rnn_input_size = self._n_prev_action

        # Handle sensor fusion (same as original ResNet policy)
        if fuse_keys is None:
            fuse_keys = observation_space.spaces.keys()
            goal_sensor_keys = {
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid,
                ObjectGoalSensor.cls_uuid,
                EpisodicGPSSensor.cls_uuid,
                PointGoalSensor.cls_uuid,
                HeadingSensor.cls_uuid,
                ProximitySensor.cls_uuid,
                EpisodicCompassSensor.cls_uuid,
                ImageGoalSensor.cls_uuid,
                InstanceImageGoalSensor.cls_uuid,
            }
            fuse_keys = [k for k in fuse_keys if k not in goal_sensor_keys]

        self._fuse_keys_1d: List[str] = [
            k for k in fuse_keys 
            if len(observation_space.spaces[k].shape) == 1 
            and k not in ["human_num_sensor", "localization_sensor"]
        ]
        
        if len(self._fuse_keys_1d) != 0:
            rnn_input_size += sum(
                observation_space.spaces[k].shape[0]
                for k in self._fuse_keys_1d
            )

        # Hidden size must be set before building any goal encoders
        self._hidden_size = hidden_size

        # Add sensor embeddings (same as original ResNet policy) and account for dims
        added_sensor_dim = self._add_sensor_embeddings(observation_space)
        rnn_input_size += added_sensor_dim

        # Create RGBD visual encoder
        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = spaces.Dict(
                {
                    k: observation_space.spaces[k]
                    for k in fuse_keys
                    if len(observation_space.spaces[k].shape) == 3
                }
            )

        self.visual_encoder = RGBDResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            rgb_backbone=rgb_backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            freeze_depth_backbone=freeze_depth_backbone,
        )

        if not self.visual_encoder.is_blind:
            # 1. 创建一个符合 observation_space 规范的虚拟输入
            dummy_inputs = {
                key: torch.randn(1, *observation_space.spaces[key].shape)
                for key in self.visual_encoder.rgb_keys + self.visual_encoder.depth_keys
            }
            # 将输入从 HWC 转换为 CHW

            
            # 2. 执行一次虚拟前向传播来获取真实的输出
            with torch.no_grad():
                dummy_output = self.visual_encoder(dummy_inputs)
            
            # 3. 从虚拟输出计算扁平化后的真实维度
            in_dim = int(np.prod(dummy_output.shape))

            # 4. 使用这个100%准确的 in_dim 来创建 visual_fc
            self.visual_fc = nn.Sequential(
                nn.Linear(in_dim, self._hidden_size),
                nn.ReLU(True),
            )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    def _add_sensor_embeddings(self, observation_space: spaces.Dict) -> int:
        """Add sensor embeddings (copied from original ResNet policy).
        Returns the total feature dimension added by these embeddings.
        """
        added_dim = 0
        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            added_dim += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            added_dim += 32

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            added_dim += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            added_dim += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            added_dim += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            added_dim += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            added_dim += 32

        # Handle goal sensors (same as original)
        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observation_space.spaces:
                goal_observation_space = spaces.Dict(
                    {"rgb": observation_space.spaces[uuid]}
                )
                goal_visual_encoder = RGBDResNetEncoder(
                    goal_observation_space,
                    baseplanes=32,
                    ngroups=16,
                    make_backbone=getattr(resnet, "resnet18"),
                    rgb_backbone="resnet18",
                    normalize_visual_inputs=False,
                )
                setattr(self, f"{uuid}_encoder", goal_visual_encoder)

                goal_visual_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(
                        np.prod(goal_visual_encoder.output_shape), self._hidden_size
                    ),
                    nn.ReLU(True),
                )
                setattr(self, f"{uuid}_fc", goal_visual_fc)

                added_dim += self._hidden_size

        return added_dim

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass (similar to original ResNet policy)."""
        x = []
        aux_loss_state = {}
        
        if not self.is_blind:
            if (
                RGBDResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                in observations
            ):
                visual_feats = observations[
                    RGBDResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)
                
            if visual_feats.dim() > 2:
                visual_feats = torch.flatten(visual_feats, 1)

            # Clone to get a normal tensor that can be used in autograd
            visual_feats = visual_feats.clone()

            visual_feats = self.visual_fc(visual_feats)
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # Add 1D sensor features
        if len(self._fuse_keys_1d) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys_1d], dim=-1
            )
            x.append(fuse_states.float())

        # Add sensor embeddings (same logic as original ResNet policy)
        self._add_sensor_features(observations, x)

        # Add previous action embedding
        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        return out, rnn_hidden_states, aux_loss_state

    def _add_sensor_features(self, observations: Dict[str, torch.Tensor], x: List[torch.Tensor]):
        """Add sensor-specific features to the feature list."""
        # Same logic as original ResNet policy
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert goal_observations.shape[1] == 3, "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1]) * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )
            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        # Handle goal sensors
        for uuid in [
            ImageGoalSensor.cls_uuid,
            InstanceImageGoalSensor.cls_uuid,
        ]:
            if uuid in observations:
                goal_image = observations[uuid]
                goal_visual_encoder = getattr(self, f"{uuid}_encoder")
                goal_visual_output = goal_visual_encoder({"rgb": goal_image})
                goal_visual_fc = getattr(self, f"{uuid}_fc")
                x.append(goal_visual_fc(goal_visual_output))