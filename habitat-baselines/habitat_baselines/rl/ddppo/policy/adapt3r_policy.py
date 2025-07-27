#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ---------------------------------------------------------------------------- #
# ADAPT3R Policy for Habitat Integration (Final, Utils-Verified Version)
# ---------------------------------------------------------------------------- #
# This file adapts the Adapt3R encoder logic into a Habitat-compatible
# NetPolicy. It combines all necessary modules into a single file for
# simplicity and clarity. The logic has been verified against the provided
# utils code.
#
# Structure:
# 1. Utility Functions (Replicated from adapt3r.utils)
# 2. Helper Modules (e.g., ResNet, CLIP, PointCloud Extractor parts)
# 3. Core Encoder: Adapt3REncoder
# 4. Habitat Net: Adapt3RNet
# 5. Habitat Policy: Adapt3RPolicy
# ---------------------------------------------------------------------------- #

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.resnet import _resnet, Bottleneck, ResNet
from torchvision import transforms

import numpy as np
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from gym import spaces
from torch import Tensor

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Net, NetPolicy
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.utils.common import get_num_actions

try:
    import clip
    from clip.model import ModifiedResNet
except ImportError:
    clip = None
    ModifiedResNet = object # Dummy class if clip is not installed
    print("Warning: CLIP library not found. CLIP-based backbones will not be available.")

try:
    import dgl.geometry as dgl_geo
except ImportError:
    dgl_geo = None
    print("Warning: DGL library not found. Farthest point sampling will not be available.")


if TYPE_CHECKING:
    from omegaconf import DictConfig

# ############################################################################ #
# 1. 辅助工具函数 (Replicated from provided utils)
# ############################################################################ #

class PointCloudUtils:
    """
    Replicated point cloud utility functions from adapt3r.utils.point_cloud_utils
    """
    @staticmethod
    def depth2fgpcd_batch(depth, cam_params):
        B, ncam, h, w = depth.shape
        
        fx = cam_params[..., 0, 0].view(B, ncam, 1, 1)
        fy = cam_params[..., 1, 1].view(B, ncam, 1, 1)
        cx = cam_params[..., 0, 2].view(B, ncam, 1, 1)
        cy = cam_params[..., 1, 2].view(B, ncam, 1, 1)
        
        pos_y, pos_x = torch.meshgrid(
            torch.arange(h, device=depth.device, dtype=torch.float32), 
            torch.arange(w, device=depth.device, dtype=torch.float32), 
            indexing='ij'
        )
        pos_x = pos_x.expand(B, ncam, -1, -1)
        pos_y = pos_y.expand(B, ncam, -1, -1)

        x_coords = (pos_x - cx) * depth / fx
        y_coords = (pos_y - cy) * depth / fy
        
        pcd_cam = torch.stack([x_coords, y_coords, depth], dim=-1)
        return einops.rearrange(pcd_cam, 'b ncam h w c -> b ncam (h w) c')

    @staticmethod
    def batch_transform_point_cloud(pcd, transform):
        pcd_homo = F.pad(pcd, (0, 1), mode="constant", value=1.0)
        transform = transform.to(dtype=pcd.dtype)
        trans_pcd_homo = torch.einsum('bn...d,bn...id->bn...i', pcd_homo, transform)
        return trans_pcd_homo[..., :-1]

    @staticmethod
    def lift_point_cloud_batch(depths, intrinsics, extrinsics, keepdims=False):
        # depths: [B, ncam, H, W]
        # intrinsics: [B, ncam, 3, 3]
        # extrinsics: [B, ncam, 4, 4]
        B, ncam, H, W = depths.shape

        pcd_cam = PointCloudUtils.depth2fgpcd_batch(depths, intrinsics)
        trans_pcd = PointCloudUtils.batch_transform_point_cloud(pcd_cam, extrinsics)

        if keepdims:
            return einops.rearrange(trans_pcd, 'b ncam (h w) c -> b ncam h w c', h=H, w=W)
        else:
            return trans_pcd # (B, ncam, H*W, 3)

    @staticmethod
    def crop_point_cloud(pcd, boundaries):
        min_b = boundaries[:, 0:1, :]  # B, 1, 3
        max_b = boundaries[:, 1:2, :]  # B, 1, 3
        
        mask = torch.all((pcd >= min_b) & (pcd <= max_b), dim=-1) # B, N
        return mask

class EnvUtils:
    """
    Replicated environment utility functions from adapt3r.envs.utils
    """
    @staticmethod
    def list_cameras(observation_space: spaces.Dict) -> List[str]:
        # Logic adapted from utils to work with gym.spaces.Dict
        keys = observation_space.keys()
        rgb_keys = sorted([k for k in keys if k.startswith('rgb_')])
        depth_keys = sorted([k for k in keys if k.startswith('depth_')])
        
        if rgb_keys:
            return [EnvUtils.image_key_to_camera_name(k) for k in rgb_keys]
        elif depth_keys:
            return [EnvUtils.depth_key_to_camera_name(k) for k in depth_keys]
        return []

    @staticmethod
    def image_key_to_camera_name(image_key: str) -> str:
        # Simplified for habitat, assuming "rgb_[name]" format
        return image_key[4:]

    @staticmethod
    def depth_key_to_camera_name(depth_key: str) -> str:
        return depth_key[6:]

    @staticmethod
    def camera_name_to_image_key(name: str) -> str:
        return f"rgb_{name}"
    
    @staticmethod
    def camera_name_to_depth_key(name: str) -> str:
        return f"depth_{name}"
        
    @staticmethod
    def camera_name_to_intrinsic_key(name: str) -> str:
        return f"intrinsics_{name}"

    @staticmethod
    def camera_name_to_extrinsic_key(name: str) -> str:
        return f"extrinsics_{name}"

class PositionalEncodings:
    """
    Replicated position encoding functions from adapt3r.algos.utils.position_encodings
    """
    class NeRFSinusoidalPosEmb(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            assert dim % 6 == 0, 'dim must be divisible by 6'
        
        @torch.no_grad()
        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            device = x.device
            n_steps = self.dim // 6
            max_freq = n_steps - 1
            freq_bands = torch.pow(torch.tensor(2, device=device), torch.linspace(0, max_freq, steps=n_steps, device=device))
            emb = x.unsqueeze(-1) * freq_bands.view(1, 1, 1, -1)
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return einops.rearrange(emb, '... i j -> ... (i j)')

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

# ############################################################################ #
# 2. 辅助模型和模块 (ResNet, CLIP, PointCloudBaseEncoder, etc.)
# ############################################################################ #

def load_resnet_features(name: str, pretrained: bool = True):
    if name == 'resnet18':
        model = _resnet('resnet18', Bottleneck, [2, 2, 2, 2], pretrained=pretrained, progress=True)
    elif name == 'resnet50':
        model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, progress=True)
    else:
        raise NotImplementedError(f"ResNet variant {name} not supported.")

    class ResNetFeatures(ResNet):
        def forward(self, x: torch.Tensor):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x0 = self.layer1(x)
            x1 = self.layer2(x0)
            x2 = self.layer3(x1)
            x3 = self.layer4(x2)
            return {'layer1': x0, 'layer2': x1, 'layer3': x2, 'layer4': x3}

    model.__class__ = ResNetFeatures
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return model, normalize

def load_clip_features(model="RN50"):
    if clip is None:
        raise ImportError("CLIP not installed, cannot use CLIP backbone.")
    clip_model, clip_transforms = clip.load(model)
    
    class ModifiedResNetFeatures(ModifiedResNet):
        def forward(self, x: torch.Tensor):
            def stem(x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.relu3(self.bn3(self.conv3(x)))
                x = self.avgpool(x)
                return x
            x = x.type(self.conv1.weight.dtype)
            x = stem(x)
            x0 = self.layer1(x)
            x1 = self.layer2(x0)
            x2 = self.layer3(x1)
            x3 = self.layer4(x2)
            return {'layer1': x0, 'layer2': x1, 'layer3': x2, 'layer4': x3}

    clip_model.visual.__class__ = ModifiedResNetFeatures
    normalize = clip_transforms.transforms[-1]
    return clip_model.visual, normalize

class PointCloudBaseEncoder(nn.Module):
    def __init__(self, observation_space, num_points, lowdim_obs_keys, do_crop=True, boundaries=None):
        super().__init__()
        self.observation_space = observation_space
        self.num_points = num_points
        self.lowdim_obs_keys = lowdim_obs_keys if lowdim_obs_keys else []
        self.do_crop = do_crop

        if self.lowdim_obs_keys:
            lowdim_dim = sum(observation_space[k].shape[0] for k in self.lowdim_obs_keys)
            self.lowdim_encoder = nn.Sequential(nn.Linear(lowdim_dim, 128), nn.ReLU())
            self.d_out_lowdim = 128
        else:
            self.lowdim_encoder = nn.Identity()
            self.d_out_lowdim = 0
            
        default_bounds = torch.tensor(((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)), dtype=torch.float32)
        boundaries_tensor = torch.tensor(boundaries, dtype=torch.float32) if boundaries is not None else default_bounds
        self.register_buffer("boundaries", boundaries_tensor.unsqueeze(0)) # Add batch dim for broadcasting
        self.boundaries: torch.Tensor
        
    def _build_point_cloud(self, obs_data: Dict[str, Tensor]) -> Tensor:
        depths, intrinsics, extrinsics = [], [], []
        
        for cam_name in EnvUtils.list_cameras(self.observation_space):
            # Shape from (B,H,W,C) to (B,C,H,W) for processing
            depths.append(obs_data[EnvUtils.camera_name_to_depth_key(cam_name)].permute(0, 3, 1, 2))
            intrinsics.append(obs_data[EnvUtils.camera_name_to_intrinsic_key(cam_name)])
            
            # Use identity if extrinsics are not available
            ext_key = EnvUtils.camera_name_to_extrinsic_key(cam_name)
            if ext_key in obs_data:
                extrinsics.append(obs_data[ext_key])
            else:
                extrinsics.append(torch.eye(4, device=depths[-1].device).unsqueeze(0).expand(depths[-1].shape[0], -1, -1))

        # Stack to (B, N_cam, ...)
        depths = torch.stack(depths, dim=1)
        intrinsics = torch.stack(intrinsics, dim=1)
        extrinsics = torch.stack(extrinsics, dim=1)
        
        return PointCloudUtils.lift_point_cloud_batch(depths, intrinsics, extrinsics)

    def _downsample_point_cloud(self, pcd, rgb_features):
        if dgl_geo is None: raise ImportError("DGL is required for farthest point sampling.")
        
        b, n, d = pcd.shape
        downsample_indices = dgl_geo.farthest_point_sampler(pcd, self.num_points)
        
        downsampled_pcd = torch.gather(pcd, 1, einops.repeat(downsample_indices, "b n -> b n d", d=d))
        downsampled_feats = torch.gather(rgb_features, 1, einops.repeat(downsample_indices, "b n -> b n d", d=rgb_features.shape[-1]))
        return downsampled_pcd, downsampled_feats
        
    def _crop_point_cloud(self, pcd):
        if not self.do_crop: return torch.ones_like(pcd[..., 0]).bool()
        return PointCloudUtils.crop_point_cloud(pcd, self.boundaries.expand(pcd.shape[0], -1, -1))

    def _encode_lowdim(self, obs_data: Dict[str, Tensor]):
        if not self.lowdim_obs_keys: return None
        
        lowdim_tensors = [obs_data[k] for k in self.lowdim_obs_keys]
        return self.lowdim_encoder(torch.cat(lowdim_tensors, dim=-1))

# ############################################################################ #
# 3. Core Encoder: Adapt3REncoder
# ############################################################################ #

class Adapt3REncoder(PointCloudBaseEncoder):
    def __init__(
        self, observation_space, backbone_type: str, hidden_dim: int, num_points: int,
        do_image: bool, do_pos: bool, do_rgb: bool, finetune: bool,
        xyz_proj_type: str, clip_model: str, lowdim_obs_keys: List[str],
        do_crop: bool, boundaries: List[List[float]]
    ):
        super().__init__(observation_space, num_points, lowdim_obs_keys, do_crop, boundaries)
        
        self.do_image, self.do_pos, self.do_rgb = do_image, do_pos, do_rgb
        self.hidden_dim = hidden_dim
        
        if backbone_type in ["resnet18", "resnet50"]:
            self.backbone, self.normalize = load_resnet_features(backbone_type, pretrained=True)
            fpn_in_channels_list = [256, 512, 1024, 2048] if backbone_type == 'resnet50' else [64, 128, 256, 512]
        elif backbone_type == "clip":
            self.backbone, self.normalize = load_clip_features(clip_model)
            fpn_in_channels_list = [256, 512, 1024, 2048] # For RN50
        else:
            raise NotImplementedError(f"Backbone type {backbone_type} not supported")
        
        if finetune: self.backbone.train()
        else: self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad = finetune
        
        self.feature_pyramid = FeaturePyramidNetwork(fpn_in_channels_list, self.hidden_dim)
        self.fpn_output_key = '0'
        
        if xyz_proj_type == "nerf": self.xyz_proj = PositionalEncodings.NeRFSinusoidalPosEmb(self.hidden_dim)
        else: self.xyz_proj = nn.Identity()

        pc_in_dim = (do_image * hidden_dim) + (do_pos * (3 if xyz_proj_type == "none" else hidden_dim)) + (do_rgb * 3)
        self.pointcloud_extractor = nn.Sequential(nn.Linear(pc_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.d_out_perception = hidden_dim

    @property
    def is_blind(self): return not self.do_image

    def forward(self, observations: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        pcds_world = self._build_point_cloud(observations) # (B, N_cam, H*W, 3)
        
        rgb_tensors = [observations[EnvUtils.camera_name_to_image_key(cam)].permute(0, 3, 1, 2) 
                       for cam in EnvUtils.list_cameras(self.observation_space)]
        
        rgb_batch = torch.cat(rgb_tensors, dim=0)
        n_cam = len(rgb_tensors)
        B = rgb_batch.shape[0] // n_cam

        with torch.set_grad_enabled(self.backbone.training):
            rgb_features_dict = self.backbone(self.normalize(rgb_batch))
        
        fpn_features_dict = self.feature_pyramid({f'layer{i+1}': v for i, v in enumerate(rgb_features_dict.values())})
        rgb_features = fpn_features_dict[self.fpn_output_key]

        feat_h, feat_w = rgb_features.shape[-2:]
        pcd_interp = F.interpolate(pcds_world.permute(0, 1, 3, 2), (feat_h * feat_w)).permute(0, 1, 3, 2)
        pcd_interp = einops.rearrange(pcd_interp, 'b ncam (h w) c -> b ncam h w c', h=feat_h, w=feat_w)

        pcd_flat = einops.rearrange(pcd_interp, "b n h w c -> b (n h w) c")
        rgb_features_flat = einops.rearrange(rgb_features, "(b n) c h w -> b (n h w) c", n=n_cam)
        
        mask = self._crop_point_cloud(pcd=pcd_flat)
        pcd_flat, rgb_features_flat = pcd_flat * mask.unsqueeze(-1), rgb_features_flat * mask.unsqueeze(-1)
        
        pcd_down, feats_down = self._downsample_point_cloud(pcd_flat, rgb_features_flat)
        
        pcd_pos_emb = self.xyz_proj(pcd_down)
        
        cat_cloud = []
        if self.do_pos: cat_cloud.append(pcd_pos_emb)
        if self.do_image: cat_cloud.append(feats_down)
        
        final_cloud_features = torch.cat(cat_cloud, dim=-1)
        perception_out = torch.max(self.pointcloud_extractor(final_cloud_features), dim=1)[0]
        
        lowdim_out = self._encode_lowdim(observations)

        return perception_out, lowdim_out

# ############################################################################ #
# 4. Habitat Net: Adapt3RNet
# ############################################################################ #

class Adapt3RNet(Net):
    def __init__(self, observation_space: spaces.Dict, action_space, config):
        super().__init__()
        
        self.visual_encoder = Adapt3REncoder(observation_space, **config.visual_encoder)
        self._hidden_size = config.hidden_size
        
        is_discrete = isinstance(action_space, spaces.Discrete)
        action_dim = action_space.n if is_discrete else action_space.shape[0]
        self.prev_action_embedding = nn.Embedding(action_dim + 1, 32) if is_discrete else nn.Linear(action_dim, 32)
        
        rnn_input_size = 32 + self.visual_encoder.d_out_perception
        if self.visual_encoder.d_out_lowdim > 0:
            rnn_input_size += self.visual_encoder.d_out_lowdim
        
        self.state_encoder = build_rnn_state_encoder(rnn_input_size, self._hidden_size, config.rnn_type, config.num_recurrent_layers)
        
        self.train()

    @property
    def output_size(self): return self._hidden_size
    @property
    def is_blind(self): return self.visual_encoder.is_blind
    @property
    def num_recurrent_layers(self): return self.state_encoder.num_recurrent_layers
    @property
    def recurrent_hidden_size(self): return self._hidden_size
    @property
    def perception_embedding_size(self): return self.visual_encoder.d_out_perception

    def forward(self, observations, rnn_hxs, prev_actions, masks, **kwargs):
        perception_feats, lowdim_feats = self.visual_encoder(observations)
        
        x = [perception_feats]
        if lowdim_feats is not None: x.append(lowdim_feats)
            
        if isinstance(self.prev_action_embedding, nn.Embedding):
            prev_actions = self.prev_action_embedding(torch.where(masks.view(-1), prev_actions.squeeze(-1) + 1, torch.zeros_like(prev_actions.squeeze(-1))))
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())
        x.append(prev_actions)
        
        out, rnn_hxs = self.state_encoder(torch.cat(x, dim=1), rnn_hxs, masks)
        return out, rnn_hxs, {}

# ############################################################################ #
# 5. Habitat Policy: Adapt3RPolicy
# ############################################################################ #

@baseline_registry.register_policy
class Adapt3RPolicy(NetPolicy):
    def __init__(self, observation_space, action_space, policy_config, **kwargs):
        super().__init__(
            net=Adapt3RNet(observation_space, action_space, policy_config),
            action_space=action_space,
            policy_config=policy_config,
        )
        self.net.apply(weight_init)

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        # Good practice: filter out rendering sensors from obs space
        ignore_names = [s.uuid for s in config.habitat_baselines.eval.extra_sim_sensors.values()]
        filtered_obs = spaces.Dict(OrderedDict([(k, v) for k, v in observation_space.items() if k not in ignore_names]))
        
        return cls(filtered_obs, action_space, config.habitat_baselines.rl.policy)