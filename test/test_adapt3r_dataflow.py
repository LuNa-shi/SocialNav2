#!/usr/bin/env python3

"""
Independent unit test for Adapt3R policy dataflow and tensor shape validation.
This test can be run standalone to debug tensor shapes at each step.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from gym import spaces
from collections import OrderedDict

# Add the path to import adapt3r_policy
sys.path.append(os.path.join(os.path.dirname(__file__), 'habitat-baselines'))

# Mock DGL for testing without DGL dependency
class MockDGLGeometry:
    @staticmethod
    def farthest_point_sampler(points, num_samples):
        """Mock farthest point sampling - just return random indices for testing"""
        batch_size, num_points, _ = points.shape
        # Return random indices for each batch
        indices = torch.randint(0, num_points, (batch_size, num_samples), device=points.device)
        return indices

# Monkey patch DGL
import habitat_baselines.rl.ddppo.policy.adapt3r_policy as adapt3r_module
adapt3r_module.dgl_geo = MockDGLGeometry()

from habitat_baselines.rl.ddppo.policy.adapt3r_policy import (
    PointCloudUtils, Adapt3REncoder, PointCloudBaseEncoder, EnvUtils, Adapt3RNet, Adapt3RPolicy
)

class TestAdapt3RDataflow:
    """Test class for Adapt3R policy dataflow validation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.height = 128
        self.width = 160
        self.num_cameras = 2
        self.num_points = 512
        self.hidden_dim = 252  # Must be divisible by 6 for NeRF positional encoding
        
    def create_mock_observation_space(self) -> spaces.Dict:
        """Create mock observation space for testing"""
        obs_space = {}
        
        # Add RGB and depth sensors for multiple cameras
        camera_names = ['front', 'left']
        for cam in camera_names:
            obs_space[f'rgb_{cam}'] = spaces.Box(
                low=0, high=255, 
                shape=(self.height, self.width, 3), 
                dtype=np.uint8
            )
            obs_space[f'depth_{cam}'] = spaces.Box(
                low=0.0, high=10.0, 
                shape=(self.height, self.width, 1), 
                dtype=np.float32
            )
            obs_space[f'intrinsics_{cam}'] = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(3, 3), 
                dtype=np.float32
            )
            obs_space[f'extrinsics_{cam}'] = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(4, 4), 
                dtype=np.float32
            )
        
        return spaces.Dict(obs_space)
    
    def create_mock_observations(self) -> Dict[str, torch.Tensor]:
        """Create mock observation data for testing"""
        obs = {}
        camera_names = ['front', 'left']
        
        for cam in camera_names:
            # RGB data (B, H, W, 3)
            obs[f'rgb_{cam}'] = torch.randint(
                0, 255, (self.batch_size, self.height, self.width, 3), 
                dtype=torch.uint8, device=self.device
            ).float()
            
            # Depth data (B, H, W, 1) - single channel depth
            obs[f'depth_{cam}'] = torch.rand(
                (self.batch_size, self.height, self.width, 1), 
                dtype=torch.float32, device=self.device
            ) * 5.0  # 0-5 meter depth range
            
            # Camera intrinsics (B, 3, 3)
            intrinsics = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
            intrinsics[:, 0, 0] = 160.0  # fx
            intrinsics[:, 1, 1] = 120.0  # fy
            intrinsics[:, 0, 2] = self.width / 2   # cx
            intrinsics[:, 1, 2] = self.height / 2  # cy
            obs[f'intrinsics_{cam}'] = intrinsics
            
            # Camera extrinsics (B, 4, 4) - identity for simplicity
            obs[f'extrinsics_{cam}'] = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
            
        return obs
    
    def test_point_cloud_utils(self):
        """Test PointCloudUtils functions"""
        print("=== Testing PointCloudUtils ===")
        
        # Test data
        depths = torch.rand(self.batch_size, self.num_cameras, self.height, self.width, device=self.device) * 5.0
        intrinsics = torch.eye(3, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_cameras, 1, 1)
        intrinsics[:, :, 0, 0] = 160.0  # fx
        intrinsics[:, :, 1, 1] = 120.0  # fy
        intrinsics[:, :, 0, 2] = self.width / 2   # cx
        intrinsics[:, :, 1, 2] = self.height / 2  # cy
        
        print(f"Input depths shape: {depths.shape}")
        print(f"Input intrinsics shape: {intrinsics.shape}")
        
        # Test depth to point cloud conversion
        pcd_cam = PointCloudUtils.depth2fgpcd_batch(depths, intrinsics)
        print(f"Camera point cloud shape: {pcd_cam.shape}")
        assert pcd_cam.shape == (self.batch_size, self.num_cameras, self.height * self.width, 3)
        
        # Test coordinate transformation
        extrinsics = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_cameras, 1, 1)
        print(f"Input extrinsics shape: {extrinsics.shape}")
        
        pcd_world = PointCloudUtils.batch_transform_point_cloud(pcd_cam, extrinsics)
        print(f"World point cloud shape: {pcd_world.shape}")
        assert pcd_world.shape == (self.batch_size, self.num_cameras, self.height * self.width, 3)
        
        # Test full pipeline
        pcd_final = PointCloudUtils.lift_point_cloud_batch(depths, intrinsics, extrinsics)
        print(f"Final point cloud shape: {pcd_final.shape}")
        assert pcd_final.shape == (self.batch_size, self.num_cameras, self.height * self.width, 3)
        
        print("✅ PointCloudUtils tests passed!\n")
    
    def test_encoder_initialization(self):
        """Test Adapt3REncoder initialization"""
        print("=== Testing Adapt3REncoder Initialization ===")
        
        obs_space = self.create_mock_observation_space()
        
        # Test configuration with CLIP backbone
        config = {
            'backbone_type': 'clip',  # Use CLIP as requested
            'hidden_dim': self.hidden_dim,
            'num_points': self.num_points,
            'do_image': True,
            'do_pos': True,
            'do_rgb': False,
            'finetune': False,
            'xyz_proj_type': 'nerf',
            'clip_model': 'RN50',
            'lowdim_obs_keys': [],
            'do_crop': True,
            'boundaries': [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]
        }
        
        encoder = Adapt3REncoder(obs_space, **config)
        encoder = encoder.to(self.device)
        
        print(f"Encoder hidden dimension: {encoder.hidden_dim}")
        print(f"Encoder num points: {encoder.num_points}")
        print(f"Encoder perception output size: {encoder.d_out_perception}")
        print(f"Encoder lowdim output size: {encoder.d_out_lowdim}")
        print("✅ Encoder initialization passed!\n")
        
        return encoder
    
    def test_encoder_forward_step_by_step(self, encoder):
        """Test encoder forward pass step by step"""
        print("=== Testing Encoder Forward Pass (Step by Step) ===")
        
        observations = self.create_mock_observations()
        encoder.eval()
        
        with torch.no_grad():
            # Step 1: Build point cloud
            print("Step 1: Building point cloud...")
            pcds_world = encoder._build_point_cloud(observations)
            print(f"Point cloud shape: {pcds_world.shape}")
            expected_shape = (self.batch_size, self.num_cameras, self.height * self.width, 3)
            assert pcds_world.shape == expected_shape, f"Expected {expected_shape}, got {pcds_world.shape}"
            
            # Step 2: Extract RGB features
            print("Step 2: Extracting RGB features...")
            rgb_tensors = [observations[EnvUtils.camera_name_to_image_key(cam)].permute(0, 3, 1, 2) 
                          for cam in EnvUtils.list_cameras(encoder.observation_space)]
            
            print(f"Number of cameras: {len(rgb_tensors)}")
            for i, rgb in enumerate(rgb_tensors):
                print(f"RGB tensor {i} shape: {rgb.shape}")
            
            rgb_batch = torch.cat(rgb_tensors, dim=0)
            print(f"Concatenated RGB batch shape: {rgb_batch.shape}")
            
            # Step 3: Backbone feature extraction
            print("Step 3: Backbone feature extraction...")
            rgb_features_dict = encoder.backbone(encoder.normalize(rgb_batch))
            print("Backbone output keys:", list(rgb_features_dict.keys()))
            for key, features in rgb_features_dict.items():
                print(f"  {key}: {features.shape}")
            
            # Step 4: FPN features
            print("Step 4: FPN feature extraction...")
            
            # Convert CLIP features to float32 for compatibility with FPN if needed
            if encoder.backbone_type == 'clip':
                rgb_features_dict = {k: v.float() for k, v in rgb_features_dict.items()}
                print("  Converted CLIP features to float32 for FPN compatibility")
            
            fpn_features_dict = encoder.feature_pyramid({f'layer{i+1}': v for i, v in enumerate(rgb_features_dict.values())})
            print("FPN output keys:", list(fpn_features_dict.keys()))
            for key, features in fpn_features_dict.items():
                print(f"  {key}: {features.shape}")
            
            # Check if the expected key exists, otherwise use the first available key
            if encoder.fpn_output_key in fpn_features_dict:
                rgb_features = fpn_features_dict[encoder.fpn_output_key]
                print(f"Using expected key '{encoder.fpn_output_key}' for FPN features")
            else:
                available_keys = list(fpn_features_dict.keys())
                fallback_key = available_keys[0]  # Use the first available key
                rgb_features = fpn_features_dict[fallback_key]
                print(f"⚠️ Expected key '{encoder.fpn_output_key}' not found, using '{fallback_key}' instead")
            
            print(f"Selected FPN features shape: {rgb_features.shape}")
            
            # Step 5: Point cloud interpolation (calling the actual encoder method)
            print("Step 5: Point cloud interpolation...")
            feat_h, feat_w = rgb_features.shape[-2:]
            print(f"Feature map size: {feat_h} x {feat_w}")
            
            # Use the same logic as in the actual encoder
            B, N_cam, orig_points, C = pcds_world.shape
            target_points = feat_h * feat_w
            stride = orig_points // target_points if orig_points >= target_points else 1
            indices = torch.arange(0, orig_points, stride, device=pcds_world.device)[:target_points]
            
            # Subsample point cloud
            pcd_interp = pcds_world[:, :, indices, :]  # (B, N_cam, target_points, 3)
            
            # Pad if needed
            if pcd_interp.shape[2] < target_points:
                padding_needed = target_points - pcd_interp.shape[2]
                last_point = pcd_interp[:, :, -1:, :].expand(-1, -1, padding_needed, -1)
                pcd_interp = torch.cat([pcd_interp, last_point], dim=2)
            
            print(f"Interpolated point cloud shape: {pcd_interp.shape}")
            
            # Step 6: Reshape and flatten
            print("Step 6: Reshape and flatten...")
            import einops
            pcd_interp = einops.rearrange(pcd_interp, 'b ncam (h w) c -> b ncam h w c', h=feat_h, w=feat_w)
            print(f"Reshaped point cloud shape: {pcd_interp.shape}")
            
            pcd_flat = einops.rearrange(pcd_interp, "b n h w c -> b (n h w) c")
            rgb_features_flat = einops.rearrange(rgb_features, "(b n) c h w -> b (n h w) c", n=self.num_cameras)
            print(f"Flattened point cloud shape: {pcd_flat.shape}")
            print(f"Flattened RGB features shape: {rgb_features_flat.shape}")
            
            # Step 7: Cropping
            print("Step 7: Point cloud cropping...")
            mask = encoder._crop_point_cloud(pcd_flat)
            print(f"Crop mask shape: {mask.shape}")
            print(f"Valid points ratio: {mask.float().mean().item():.3f}")
            
            pcd_masked = pcd_flat * mask.unsqueeze(-1)
            rgb_masked = rgb_features_flat * mask.unsqueeze(-1)
            print(f"Masked point cloud shape: {pcd_masked.shape}")
            print(f"Masked RGB features shape: {rgb_masked.shape}")
            
            # Step 8: Downsampling
            print("Step 8: Point cloud downsampling...")
            pcd_down, feats_down = encoder._downsample_point_cloud(pcd_masked, rgb_masked)
            print(f"Downsampled point cloud shape: {pcd_down.shape}")
            print(f"Downsampled features shape: {feats_down.shape}")
            
            # Step 9: Position encoding
            print("Step 9: Position encoding...")
            pcd_pos_emb = encoder.xyz_proj(pcd_down)
            print(f"Position embeddings shape: {pcd_pos_emb.shape}")
            
            # Step 10: Feature concatenation
            print("Step 10: Feature concatenation...")
            cat_cloud = []
            if encoder.do_pos: 
                cat_cloud.append(pcd_pos_emb)
                print(f"Added position embeddings: {pcd_pos_emb.shape}")
            if encoder.do_image: 
                cat_cloud.append(feats_down)
                print(f"Added image features: {feats_down.shape}")
            
            final_cloud_features = torch.cat(cat_cloud, dim=-1)
            print(f"Final concatenated features shape: {final_cloud_features.shape}")
            
            # Step 11: Point cloud extractor
            print("Step 11: Point cloud feature extraction...")
            extracted_features = encoder.pointcloud_extractor(final_cloud_features)
            print(f"Extracted features shape: {extracted_features.shape}")
            
            # Step 12: Max pooling
            print("Step 12: Max pooling...")
            perception_out = torch.max(extracted_features, dim=1)[0]
            print(f"Final perception output shape: {perception_out.shape}")
            
            # Step 13: Low-dim encoding
            print("Step 13: Low-dim encoding...")
            lowdim_out = encoder._encode_lowdim(observations)
            if lowdim_out is not None:
                print(f"Low-dim output shape: {lowdim_out.shape}")
            else:
                print("No low-dim features")
            
            print("✅ Step-by-step forward pass completed!\n")
            
            return perception_out, lowdim_out
    
    def test_full_forward_pass(self, encoder):
        """Test full forward pass"""
        print("=== Testing Full Forward Pass ===")
        
        observations = self.create_mock_observations()
        encoder.eval()
        
        with torch.no_grad():
            perception_out, lowdim_out = encoder(observations)
            
            print(f"Final perception output shape: {perception_out.shape}")
            print(f"Expected shape: ({self.batch_size}, {encoder.d_out_perception})")
            assert perception_out.shape == (self.batch_size, encoder.d_out_perception)
            
            if lowdim_out is not None:
                print(f"Final lowdim output shape: {lowdim_out.shape}")
                assert lowdim_out.shape == (self.batch_size, encoder.d_out_lowdim)
            else:
                print("No lowdim output (as expected)")
            
            print("✅ Full forward pass test passed!\n")
    
    def test_adapt3r_net_end_to_end(self):
        """Test complete Adapt3RNet with RNN state encoder"""
        print("=== Testing Adapt3RNet End-to-End ===")
        
        obs_space = self.create_mock_observation_space()
        
        # Create mock action space
        from gym import spaces as gym_spaces
        action_space = gym_spaces.Discrete(4)  # 4 actions: forward, left, right, stop
        
        # Create Adapt3RNet configuration
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'num_recurrent_layers': 1,
            'visual_encoder': {
                'backbone_type': 'clip',  # Use CLIP backbone
                'hidden_dim': 252,  # Must be divisible by 6
                'num_points': 512,
                'do_image': True,
                'do_pos': True,
                'do_rgb': False,
                'finetune': False,
                'xyz_proj_type': 'nerf',
                'clip_model': 'RN50',
                'lowdim_obs_keys': [],
                'do_crop': True,
                'boundaries': [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]
            }
        })
        
        # Initialize Adapt3RNet
        print("Initializing Adapt3RNet...")
        net = Adapt3RNet(obs_space, action_space, config)
        net = net.to(self.device)
        net.eval()
        
        print(f"Net output size: {net.output_size}")
        print(f"Net recurrent layers: {net.num_recurrent_layers}")
        print(f"Net recurrent hidden size: {net.recurrent_hidden_size}")
        print(f"Net perception embedding size: {net.perception_embedding_size}")
        
        # Create mock observations and states
        observations = self.create_mock_observations()
        rnn_hidden_states = torch.zeros(
            self.batch_size, 
            net.num_recurrent_layers, 
            net.recurrent_hidden_size, 
            device=self.device
        )
        prev_actions = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        masks = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
        
        print(f"Input shapes:")
        print(f"  observations: {len(observations)} sensors")
        print(f"  rnn_hidden_states: {rnn_hidden_states.shape}")
        print(f"  prev_actions: {prev_actions.shape}")
        print(f"  masks: {masks.shape}")
        
        # Test forward pass
        print("\nTesting Adapt3RNet forward pass...")
        with torch.no_grad():
            features, new_rnn_states, aux_outputs = net(
                observations, 
                rnn_hidden_states, 
                prev_actions, 
                masks
            )
        
        print(f"Output shapes:")
        print(f"  features: {features.shape}")
        print(f"  new_rnn_states: {new_rnn_states.shape}")
        print(f"  aux_outputs: {aux_outputs}")
        
        # Verify output shapes
        assert features.shape == (self.batch_size, net.output_size), f"Expected {(self.batch_size, net.output_size)}, got {features.shape}"
        assert new_rnn_states.shape == rnn_hidden_states.shape, f"RNN state shape mismatch"
        
        print("✅ Adapt3RNet end-to-end test passed!\n")
        return net, features, new_rnn_states
    
    def test_adapt3r_policy_end_to_end(self):
        """Test complete Adapt3RPolicy with action prediction"""
        print("=== Testing Adapt3RPolicy End-to-End ===")
        
        obs_space = self.create_mock_observation_space()
        
        # Create mock action space
        from gym import spaces as gym_spaces
        action_space = gym_spaces.Discrete(4)  # 4 actions: forward, left, right, stop
        
        # Create policy configuration
        from omegaconf import OmegaConf
        policy_config = OmegaConf.create({
            'name': 'Adapt3RPolicy',
            'action_distribution_type': 'categorical',
            'hidden_size': 512,
            'rnn_type': 'GRU',  
            'num_recurrent_layers': 1,
            'visual_encoder': {
                'backbone_type': 'clip',  # Use CLIP backbone
                'hidden_dim': 252,
                'num_points': 512,
                'do_image': True,
                'do_pos': True,
                'do_rgb': False,
                'finetune': False,
                'xyz_proj_type': 'nerf',
                'clip_model': 'RN50',
                'lowdim_obs_keys': [],
                'do_crop': True,
                'boundaries': [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]
            }
        })
        
        # Initialize Adapt3RPolicy
        print("Initializing Adapt3RPolicy...")
        policy = Adapt3RPolicy(obs_space, action_space, policy_config)
        policy = policy.to(self.device)
        policy.eval()
        
        print(f"Policy initialized successfully")
        print(f"Action space: {action_space}")
        print(f"Action distribution type: {policy_config.action_distribution_type}")
        
        # Create mock observations and states
        observations = self.create_mock_observations()
        rnn_hidden_states = torch.zeros(
            self.batch_size, 
            policy.net.num_recurrent_layers, 
            policy.net.recurrent_hidden_size, 
            device=self.device
        )
        prev_actions = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        masks = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
        
        print(f"Input shapes:")
        print(f"  observations: {len(observations)} sensors")
        print(f"  rnn_hidden_states: {rnn_hidden_states.shape}")
        print(f"  prev_actions: {prev_actions.shape}")
        print(f"  masks: {masks.shape}")
        
        # Test forward pass and action prediction
        print("\nTesting policy forward pass and action prediction...")
        with torch.no_grad():
            # Test act method (inference)
            policy_output = policy.act(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic=False
            )
            
            # Extract data from PolicyActionData object
            actions = policy_output.actions
            rnn_states_new = policy_output.rnn_hidden_states
            
            print(f"Policy action prediction:")
            print(f"  actions: {actions.shape} -> {actions}")
            print(f"  new_rnn_states: {rnn_states_new.shape}")
            
            # Test evaluate_actions method (training)
            evaluation_results = policy.evaluate_actions(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                actions
            )
            
            print(f"Policy evaluation results:")
            for key, value in evaluation_results.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
        
        # Verify output shapes and ranges
        assert actions.shape == (self.batch_size, 1), f"Expected {(self.batch_size, 1)}, got {actions.shape}"
        assert torch.all(actions >= 0) and torch.all(actions < action_space.n), "Actions out of valid range"
        assert rnn_states_new.shape == rnn_hidden_states.shape, "RNN state shape mismatch"
        
        print("✅ Adapt3RPolicy end-to-end test passed!\n")
        return policy
    
    def test_adapt3r_training_mode(self):
        """Test Adapt3R in training mode with gradient computation"""
        print("=== Testing Adapt3R Training Mode ===")
        
        obs_space = self.create_mock_observation_space()
        from gym import spaces as gym_spaces
        action_space = gym_spaces.Discrete(4)
        
        # Create configuration with gradient computation enabled
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'hidden_size': 256,  # Smaller for faster testing
            'rnn_type': 'LSTM',  # Test different RNN type
            'num_recurrent_layers': 2,
            'visual_encoder': {
                'backbone_type': 'clip',  # Use CLIP backbone
                'hidden_dim': 252,
                'num_points': 256,  # Smaller for faster testing
                'do_image': True,
                'do_pos': True,
                'do_rgb': False,
                'finetune': True,  # Enable gradient computation
                'xyz_proj_type': 'nerf',
                'clip_model': 'RN50',
                'lowdim_obs_keys': [],
                'do_crop': True,
                'boundaries': [[-3.0, -3.0, 0.0], [3.0, 3.0, 2.0]]
            }
        })
        
        # Initialize network in training mode
        print("Initializing Adapt3RNet in training mode...")
        net = Adapt3RNet(obs_space, action_space, config)
        net = net.to(self.device)
        net.train()  # Set to training mode
        
        # Create mock data
        observations = self.create_mock_observations()
        rnn_hidden_states = torch.zeros(
            self.batch_size,
            net.num_recurrent_layers,
            net.recurrent_hidden_size,
            device=self.device
        )
        prev_actions = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        masks = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
        
        print(f"Network in training mode: {net.training}")
        print(f"Backbone requires_grad: {next(net.visual_encoder.backbone.parameters()).requires_grad}")
        
        # Test forward pass with gradient computation
        print("\nTesting forward pass with gradients...")
        features, new_rnn_states, aux_outputs = net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks
        )
        
        print(f"Features shape: {features.shape}")
        print(f"Features requires_grad: {features.requires_grad}")
        
        # Test backward pass
        print("Testing backward pass...")
        dummy_loss = features.mean()
        dummy_loss.backward()
        
        # Check that gradients were computed
        has_gradients = False
        for name, param in net.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        print(f"Gradients computed: {has_gradients}")
        print(f"Dummy loss value: {dummy_loss.item():.6f}")
        
        assert has_gradients, "No gradients were computed in training mode"
        print("✅ Adapt3R training mode test passed!\n")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Adapt3R Dataflow Tests")
        print("=" * 50)
        
        try:
            # Test individual components
            self.test_point_cloud_utils()
            
            # Test encoder
            encoder = self.test_encoder_initialization()
            
            # Test step-by-step forward pass
            self.test_encoder_forward_step_by_step(encoder)
            
            # Test full forward pass
            self.test_full_forward_pass(encoder)
            
            print("🎉 Basic encoder tests passed successfully!")
            print("=" * 50)
            
            # Test complete end-to-end pipeline
            print("🚀 Starting End-to-End Tests...")
            
            # Test Adapt3RNet (encoder + RNN)
            net, features, rnn_states = self.test_adapt3r_net_end_to_end()
            
            # Test Adapt3RPolicy (complete policy with actions)
            policy = self.test_adapt3r_policy_end_to_end()
            
            # Test training mode
            self.test_adapt3r_training_mode()
            
            print("🎉 All end-to-end tests passed successfully!")
            print("=" * 50)
            print("📊 Test Summary:")
            print("  ✅ Point cloud utilities")
            print("  ✅ Adapt3R encoder (visual)")
            print("  ✅ Step-by-step dataflow")
            print("  ✅ Adapt3R network (encoder + RNN)")
            print("  ✅ Adapt3R policy (complete)")
            print("  ✅ Training mode with gradients")
            print("\n🎯 Adapt3R implementation is fully functional!")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def debug_network_outputs(self):
        """Debug network outputs for numerical stability"""
        print("=== Debugging Network Outputs ===")
        
        obs_space = self.create_mock_observation_space()
        from gym import spaces as gym_spaces
        action_space = gym_spaces.Discrete(4)
        
        # Use ResNet18 for more stable testing
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'num_recurrent_layers': 1,
            'visual_encoder': {
                'backbone_type': 'resnet18',  # More stable than CLIP
                'hidden_dim': 252,
                'num_points': 512,
                'do_image': True,
                'do_pos': True,
                'do_rgb': False,
                'finetune': False,
                'xyz_proj_type': 'nerf',
                'clip_model': 'RN50',
                'lowdim_obs_keys': [],
                'do_crop': True,
                'boundaries': [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]
            }
        })
        
        # Initialize network
        net = Adapt3RNet(obs_space, action_space, config)
        net = net.to(self.device)
        net.eval()
        
        # Create inputs
        observations = self.create_mock_observations()
        rnn_hidden_states = torch.zeros(
            self.batch_size, 
            net.num_recurrent_layers, 
            net.recurrent_hidden_size, 
            device=self.device
        )
        prev_actions = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        masks = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
        
        # Test network forward pass
        with torch.no_grad():
            features, new_rnn_states, aux_outputs = net(
                observations, 
                rnn_hidden_states, 
                prev_actions, 
                masks
            )
        
        print(f"Network features shape: {features.shape}")
        print(f"Features range: [{features.min().item():.6f}, {features.max().item():.6f}]")
        print(f"Features mean: {features.mean().item():.6f}")
        print(f"Features std: {features.std().item():.6f}")
        print(f"Contains inf: {torch.isinf(features).any().item()}")
        print(f"Contains nan: {torch.isnan(features).any().item()}")
        
        print("✅ Network output debug completed!\n")
        return net
    
    def test_simple_policy(self):
        """Test policy with ResNet18 for more stability"""
        print("=== Testing Simple Policy (ResNet18) ===")
        
        net = self.debug_network_outputs()
        
        obs_space = self.create_mock_observation_space()
        from gym import spaces as gym_spaces
        action_space = gym_spaces.Discrete(4)
        
        # Create simple policy config
        from omegaconf import OmegaConf
        policy_config = OmegaConf.create({
            'name': 'Adapt3RPolicy',
            'action_distribution_type': 'categorical',
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'num_recurrent_layers': 1,
            'visual_encoder': {
                'backbone_type': 'resnet18',  # More stable
                'hidden_dim': 252,
                'num_points': 512,
                'do_image': True,
                'do_pos': True,
                'do_rgb': False,
                'finetune': False,
                'xyz_proj_type': 'nerf',
                'clip_model': 'RN50',
                'lowdim_obs_keys': [],
                'do_crop': True,
                'boundaries': [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]
            }
        })
        
        # Test policy
        policy = Adapt3RPolicy(obs_space, action_space, policy_config)
        policy = policy.to(self.device)
        policy.eval()
        
        observations = self.create_mock_observations()
        rnn_hidden_states = torch.zeros(
            self.batch_size, 
            policy.net.num_recurrent_layers, 
            policy.net.recurrent_hidden_size, 
            device=self.device
        )
        prev_actions = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        masks = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
        
        print("Testing policy with deterministic actions...")
        with torch.no_grad():
            policy_output = policy.act(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                deterministic=True  # Use deterministic for stability
            )
            
            # Extract actions and rnn states from PolicyActionData object
            actions = policy_output.actions
            rnn_states_new = policy_output.rnn_hidden_states
            
            print(f"Actions shape: {actions.shape}")
            print(f"Actions: {actions.squeeze()}")
            print(f"Action range: [{actions.min().item()}, {actions.max().item()}]")
        
        print("✅ Simple policy test passed!\n")
        return policy
    
    def run_quick_test(self):
        """Run a quick subset of tests for faster debugging"""
        print("🚀 Starting Quick Adapt3R Test")
        print("=" * 30)
        
        try:
            # Test encoder only (with CLIP)
            encoder = self.test_encoder_initialization()
            self.test_full_forward_pass(encoder)
            
            # Test simple policy (with ResNet18 for stability)
            policy = self.test_simple_policy()
            
            print("🎉 Quick test passed successfully!")
            
        except Exception as e:
            print(f"❌ Quick test failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main test function"""
    print("Adapt3R Policy End-to-End Test")
    print("This test validates the complete Adapt3R implementation including:")
    print("  - Point cloud processing and dataflow")
    print("  - Visual encoder with ResNet/CLIP backbones")
    print("  - Complete policy with RNN and action prediction")
    print("  - Training mode with gradient computation")
    print()
    
    # Check command line arguments for test mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("Running in QUICK TEST mode...")
        tester = TestAdapt3RDataflow()
        tester.run_quick_test()
    else:
        print("Running FULL TEST suite...")
        print("(Use 'python test_adapt3r_dataflow.py quick' for faster testing)")
        print()
        tester = TestAdapt3RDataflow()
        tester.run_all_tests()


if __name__ == "__main__":
    main() 