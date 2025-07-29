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
from omegaconf import OmegaConf

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

# # Import camera sensors to ensure they are registered
# try:
#     from habitat_baselines.rl.ddppo.policy.habitat_camera_sensors import (
#         CameraIntrinsicsSensor, CameraExtrinsicsSensor
#     )
# except ImportError:
#     print("Warning: Camera sensors not found. Using default values for intrinsics/extrinsics.")


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
            # 修复rearrange模式，从(h w)转换回h,w维度
            return einops.rearrange(trans_pcd, 'b ncam (h w) c -> b ncam h w c', h=H, w=W)
        else:
            return trans_pcd # (B, ncam, H*W, 3)

    @staticmethod
    def crop_point_cloud(pcd, boundaries):
        min_b = boundaries[:, 0:1, :]  # B, 1, 3
        max_b = boundaries[:, 1:2, :]  # B, 1, 3
        
        mask = torch.all((pcd >= min_b) & (pcd <= max_b), dim=-1) # B, N
        return mask

#TODO: use the real obs key
class EnvUtils:
    """
    Utilities to map camera names to observation keys,
    updated to match the final observation dictionary format.
    """
    @staticmethod
    def list_cameras(observation_space: spaces.Dict) -> List[str]:
        """
        Extracts camera names (e.g., 'jaw') from observation keys.
        Example key: 'articulated_agent_jaw_rgb' -> 'jaw'
        """
        cam_names = set()
        for k in observation_space.keys():
            # Match keys like 'articulated_agent_jaw_rgb' or 'head_rgb'
            if k.endswith("_rgb"):
                parts = k.split('_')
                # The camera name is the part before '_rgb'
                cam_name = parts[-2]
                cam_names.add(cam_name)
        return sorted(list(cam_names))

    @staticmethod
    def camera_name_to_image_key(name: str) -> str:
        """
        Generates the RGB image key. Matches keys like:
        'articulated_agent_jaw_rgb'
        'head_rgb'
        """
        # Heuristic: if name is 'jaw' or 'arm', it's part of articulated agent
        if name in ["jaw", "arm"]:
            return f"articulated_agent_{name}_rgb"
        else:
            return f"{name}_rgb" # For keys like 'head_rgb'

    @staticmethod
    def camera_name_to_depth_key(name: str) -> str:
        """
        Generates the depth image key. Matches keys like:
        'articulated_agent_jaw_depth'
        'head_depth'
        """
        if name in ["jaw", "arm"]:
            return f"articulated_agent_{name}_depth"
        else:
            return f"{name}_depth"
        
    @staticmethod
    def camera_name_to_intrinsic_key(name: str) -> str:
        """
        Generates the intrinsics key.
        NOTE: This key is NOT present in the provided observation dict.
        The code must handle its absence.
        """
        return f"{name}_intrinsics"

    @staticmethod
    def camera_name_to_extrinsic_key(name: str) -> str:
        """
        Generates the extrinsics key.
        NOTE: This key is NOT present in the provided observation dict.
        The code must handle its absence.
        """
        return f"{name}_extrinsics"



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
    # Skip initialization for CLIP models to preserve pretrained weights
    # and avoid half precision issues
    if hasattr(m, '__class__') and 'CLIP' in str(type(m)):
        return
    if hasattr(m, 'weight') and m.weight.dtype == torch.float16:
        return  # Skip half precision weights
    
    if isinstance(m, nn.Linear):
        try:
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data') and m.bias is not None:
                m.bias.data.fill_(0.0)
        except RuntimeError:
            # Fallback for problematic layers
            pass
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        try:
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data, gain)
            if hasattr(m.bias, 'data') and m.bias is not None:
                m.bias.data.fill_(0.0)
        except RuntimeError:
            # Fallback for problematic layers
            pass

# ############################################################################ #
# 2. 辅助模型和模块 (ResNet, CLIP, PointCloudBaseEncoder, etc.)
# ############################################################################ #


def load_resnet_features(name: str, pretrained: bool = True):
    try:
        # Try new torchvision API (v0.13+)
        if name == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
        elif name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet50(weights=weights)
        else:
            raise NotImplementedError(f"ResNet variant {name} not supported.")
    except ImportError:
        # Fallback to old API for older torchvision versions
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

        print(f"observation_space: {observation_space}")
        print(f"lowdim_obs_keys: {self.lowdim_obs_keys}")

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
            depth_tensor = obs_data[EnvUtils.camera_name_to_depth_key(cam_name)]
            if len(depth_tensor.shape) == 4:  # (B,H,W,C)
                depth_tensor = depth_tensor.permute(0, 3, 1, 2)
            depths.append(depth_tensor)
            
            # Get intrinsics or use default values
            intrinsic_key = EnvUtils.camera_name_to_intrinsic_key(cam_name)
            if intrinsic_key in obs_data:
                intrinsics.append(obs_data[intrinsic_key])
            else:
                # Create default intrinsics based on depth image size
                B, C, H, W = depth_tensor.shape
                default_intrinsics = torch.eye(3, device=depth_tensor.device).unsqueeze(0).expand(B, -1, -1)
                default_intrinsics[:, 0, 0] = W * 0.8  # fx
                default_intrinsics[:, 1, 1] = H * 0.8  # fy  
                default_intrinsics[:, 0, 2] = W / 2    # cx
                default_intrinsics[:, 1, 2] = H / 2    # cy
                intrinsics.append(default_intrinsics)
            
            # Get extrinsics or use identity matrix
            ext_key = EnvUtils.camera_name_to_extrinsic_key(cam_name)
            if ext_key in obs_data:
                extrinsics.append(obs_data[ext_key])
            else:
                B = depth_tensor.shape[0]
                identity_matrix = torch.eye(4, device=depth_tensor.device).unsqueeze(0).expand(B, -1, -1)
                extrinsics.append(identity_matrix)

        # Stack to (B, N_cam, ...)
        depths = torch.stack(depths, dim=1)
        if len(depths.shape) == 5 and depths.shape[2] == 1:  # Remove channel dimension if single channel
            depths = depths.squeeze(2)
        intrinsics = torch.stack(intrinsics, dim=1)
        extrinsics = torch.stack(extrinsics, dim=1)
        
        return PointCloudUtils.lift_point_cloud_batch(depths, intrinsics, extrinsics)

    def _downsample_point_cloud(self, pcd, rgb_features):
        b, n, d = pcd.shape
        
        if dgl_geo is not None:
            # Use DGL farthest point sampling if available
            downsample_indices = dgl_geo.farthest_point_sampler(pcd, self.num_points)
            downsampled_pcd = torch.gather(pcd, 1, einops.repeat(downsample_indices, "b n -> b n d", d=d))
            downsampled_feats = torch.gather(rgb_features, 1, einops.repeat(downsample_indices, "b n -> b n d", d=rgb_features.shape[-1]))
        else:
            # Fallback to random sampling when DGL is not available
            if n <= self.num_points:
                # If we have fewer points than needed, just pad with the last point
                padding_needed = self.num_points - n
                if padding_needed > 0:
                    last_point = pcd[:, -1:, :].expand(-1, padding_needed, -1)
                    last_feat = rgb_features[:, -1:, :].expand(-1, padding_needed, -1)
                    downsampled_pcd = torch.cat([pcd, last_point], dim=1)
                    downsampled_feats = torch.cat([rgb_features, last_feat], dim=1)
                else:
                    downsampled_pcd = pcd
                    downsampled_feats = rgb_features
            else:
                # Random sampling as fallback
                indices = torch.randperm(n, device=pcd.device)[:self.num_points]
                indices = indices.sort()[0]  # Sort to maintain some order
                indices = indices.unsqueeze(0).expand(b, -1)  # (b, num_points)
                
                downsampled_pcd = torch.gather(pcd, 1, einops.repeat(indices, "b n -> b n d", d=d))
                downsampled_feats = torch.gather(rgb_features, 1, einops.repeat(indices, "b n -> b n d", d=rgb_features.shape[-1]))
        
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
        self.backbone_type = backbone_type
        
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
        
        rgb_tensors_uint8 = [
            observations[EnvUtils.camera_name_to_image_key(cam)] 
            for cam in EnvUtils.list_cameras(self.observation_space)
        ]
        
        # 2. 将它们转换成 float 类型并缩放到 [0, 1]
        rgb_tensors_float = []
        for tensor_uint8 in rgb_tensors_uint8:
            tensor_float = tensor_uint8.float() / 255.0
            rgb_tensors_float.append(tensor_float.permute(0, 3, 1, 2))
        
        # 清理不再需要的临时变量
        del rgb_tensors_uint8
        torch.cuda.empty_cache()
        
        rgb_batch = torch.cat(rgb_tensors_float, dim=0)
        n_cam = len(rgb_tensors_float)
        B = rgb_batch.shape[0] // n_cam

        # 清理不再需要的临时变量
        del rgb_tensors_float
        torch.cuda.empty_cache()

        with torch.set_grad_enabled(self.backbone.training):
            rgb_features_dict = self.backbone(self.normalize(rgb_batch))
            # 清理大型输入张量
            del rgb_batch
            torch.cuda.empty_cache()
        
        # Convert CLIP features to float32 for compatibility with FPN
        if self.backbone_type == 'clip':
            rgb_features_dict = {k: v.float() for k, v in rgb_features_dict.items()}
        
        fpn_features_dict = self.feature_pyramid({f'layer{i+1}': v for i, v in enumerate(rgb_features_dict.values())})
        # 清理特征字典
        del rgb_features_dict
        torch.cuda.empty_cache()
        
        # Use available key if expected key doesn't exist
        if self.fpn_output_key in fpn_features_dict:
            rgb_features = fpn_features_dict[self.fpn_output_key]
        else:
            available_keys = list(fpn_features_dict.keys())
            fallback_key = available_keys[0] if available_keys else 'layer1'
            rgb_features = fpn_features_dict[fallback_key]
        
        # 清理FPN特征字典
        del fpn_features_dict
        torch.cuda.empty_cache()

        feat_h, feat_w = rgb_features.shape[-2:]
        
        # Simple subsampling instead of interpolation to avoid shape issues
        B, N_cam, orig_points, C = pcds_world.shape
        target_points = feat_h * feat_w
        
        # Calculate stride for subsampling
        stride = orig_points // target_points if orig_points >= target_points else 1
        indices = torch.arange(0, orig_points, stride, device=pcds_world.device)[:target_points]
        
        # Subsample point cloud
        pcd_interp = pcds_world[:, :, indices, :]  # (B, N_cam, target_points, 3)
        
        # Pad if we don't have enough points
        if pcd_interp.shape[2] < target_points:
            padding_needed = target_points - pcd_interp.shape[2]
            last_point = pcd_interp[:, :, -1:, :].expand(-1, -1, padding_needed, -1)
            pcd_interp = torch.cat([pcd_interp, last_point], dim=2)
        
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

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None, **kwargs):
        perception_feats, lowdim_feats = self.visual_encoder(observations)
        
        x = [perception_feats]
        if lowdim_feats is not None: 
            x.append(lowdim_feats)
            
        if isinstance(self.prev_action_embedding, nn.Embedding):
            prev_actions = self.prev_action_embedding(
                torch.where(
                    masks.view(-1), 
                    prev_actions.squeeze(-1) + 1, 
                    torch.zeros_like(prev_actions.squeeze(-1))
                )
            )
        else:
            prev_actions = self.prev_action_embedding(masks * prev_actions.float())
        x.append(prev_actions)
        
        # 合并特征并清理中间变量
        cat_features = torch.cat(x, dim=1)
        del x, perception_feats
        if lowdim_feats is not None:
            del lowdim_feats
        torch.cuda.empty_cache()
        
        # 将 rnn_build_seq_info 传递给 state_encoder
        out, rnn_hidden_states = self.state_encoder(
            cat_features, rnn_hidden_states, masks, rnn_build_seq_info
        )
        
        # 清理最后的中间变量
        del cat_features
        torch.cuda.empty_cache()

        # 确保所有参数都参与计算
        dummy_loss = 0
        for param in self.parameters():
            if param.requires_grad:
                dummy_loss = dummy_loss + param.mean() * 0
        
        return out + dummy_loss.detach(), rnn_hidden_states, {}

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
        
        # Build policy configuration from config sections
        policy_config = OmegaConf.create({
            'name': 'Adapt3RPolicy',
            'action_distribution_type': config.habitat_baselines.rl.policy.agent_0.get('action_distribution_type', 'categorical'),
            'hidden_size': config.habitat_baselines.rl.ppo.hidden_size,
            'rnn_type': config.habitat_baselines.rl.ddppo.rnn_type,
            'num_recurrent_layers': config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            'visual_encoder': config.habitat_baselines.rl.ddppo.adapt3r.visual_encoder
        })
        
        return cls(filtered_obs, action_space, policy_config)





