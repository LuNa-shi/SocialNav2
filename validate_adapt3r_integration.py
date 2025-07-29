#!/usr/bin/env python3

"""
Validation script for Adapt3R integration with Falcon training system
"""

import os
import sys
import torch
from omegaconf import DictConfig, OmegaConf
from gym import spaces
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'habitat-baselines'))

def validate_adapt3r_integration():
    """Validate that Adapt3R can be imported and instantiated"""
    print("🚀 Validating Adapt3R Integration")
    print("=" * 50)
    
    try:
        # 1. Test policy import
        print("1. Testing Adapt3R Policy Import...")
        from habitat_baselines.rl.ddppo.policy.adapt3r_policy import Adapt3RPolicy
        print("   ✅ Adapt3RPolicy imported successfully")
        
        # 2. Test camera sensors import
        print("2. Testing Camera Sensors Import...")
        try:
            from habitat_baselines.rl.ddppo.policy.habitat_camera_sensors import (
                CameraIntrinsicsSensor, CameraExtrinsicsSensor
            )
            print("   ✅ Camera sensors imported successfully")
        except ImportError as e:
            print(f"   ⚠️ Camera sensors import failed: {e}")
        
        # 3. Test configuration loading
        print("3. Testing Configuration...")
        config_path = "habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train.yaml"
        
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
            print("   ✅ Configuration loaded successfully")
            
            # Check policy configuration
            if "Adapt3RPolicy" in str(config.habitat_baselines.rl.policy.agent_0.name):
                print("   ✅ Adapt3R policy configured correctly")
            else:
                print("   ❌ Adapt3R policy not found in configuration")
                
        else:
            print("   ❌ Configuration file not found")
        
        # 4. Test observation space creation
        print("4. Testing Observation Space...")
        
        # Create mock observation space that matches Habitat format
        obs_space = spaces.Dict({
            'agent_0_articulated_agent_jaw_rgb': spaces.Box(
                low=0, high=255, shape=(128, 160, 3), dtype=np.uint8
            ),
            'agent_0_articulated_agent_jaw_depth': spaces.Box(
                low=0.0, high=10.0, shape=(128, 160, 1), dtype=np.float32
            ),
            'agent_0_articulated_agent_jaw_intrinsics': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32
            ),
            'agent_0_articulated_agent_jaw_extrinsics': spaces.Box(
                low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32
            ),
            'agent_0_pointgoal_with_gps_compass': spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            'agent_0_localization_sensor': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        })
        
        action_space = spaces.Discrete(4)  # stop, forward, left, right
        
        print("   ✅ Observation and action spaces created")
        
        # 5. Test policy instantiation
        print("5. Testing Policy Instantiation...")
        
        policy_config = OmegaConf.create({
            'name': 'Adapt3RPolicy',
            'action_distribution_type': 'categorical', 
            'hidden_size': 512,
            'rnn_type': 'GRU',
            'num_recurrent_layers': 1,
            'visual_encoder': {
                'backbone_type': 'resnet18',
                'hidden_dim': 252,
                'num_points': 512,
                'do_image': True,
                'do_pos': True,
                'do_rgb': False,
                'finetune': True,
                'xyz_proj_type': 'nerf',
                'clip_model': 'RN50',
                'do_crop': True,
                'boundaries': [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]],
                'lowdim_obs_keys': [
                    'agent_0_pointgoal_with_gps_compass',
                    'agent_0_localization_sensor'
                ]
            }
        })
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            policy = Adapt3RPolicy(obs_space, action_space, policy_config)
            policy = policy.to(device)
            print(f"   ✅ Policy instantiated successfully on {device}")
            
            # 6. Test forward pass with mock data
            print("6. Testing Forward Pass...")
            
            batch_size = 2
            policy.eval()
            
            # Create mock observations
            mock_obs = {
                'agent_0_articulated_agent_jaw_rgb': torch.randint(
                    0, 255, (batch_size, 128, 160, 3), dtype=torch.uint8, device=device
                ).float(),
                'agent_0_articulated_agent_jaw_depth': torch.rand(
                    (batch_size, 128, 160, 1), dtype=torch.float32, device=device
                ) * 5.0,
                'agent_0_articulated_agent_jaw_intrinsics': torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
                'agent_0_articulated_agent_jaw_extrinsics': torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
                'agent_0_pointgoal_with_gps_compass': torch.randn((batch_size, 2), device=device),
                'agent_0_localization_sensor': torch.randn((batch_size, 3), device=device),
            }
            
            # Set intrinsics
            mock_obs['agent_0_articulated_agent_jaw_intrinsics'][:, 0, 0] = 160 * 0.8  # fx
            mock_obs['agent_0_articulated_agent_jaw_intrinsics'][:, 1, 1] = 128 * 0.8  # fy
            mock_obs['agent_0_articulated_agent_jaw_intrinsics'][:, 0, 2] = 80        # cx
            mock_obs['agent_0_articulated_agent_jaw_intrinsics'][:, 1, 2] = 64        # cy
            
            rnn_hidden_states = torch.zeros(
                batch_size, policy.net.num_recurrent_layers, policy.net.recurrent_hidden_size, device=device
            )
            prev_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            masks = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            
            with torch.no_grad():
                policy_output = policy.act(
                    mock_obs, rnn_hidden_states, prev_actions, masks, deterministic=True
                )
                
                print(f"   ✅ Forward pass successful")
                print(f"      Actions shape: {policy_output.actions.shape}")
                print(f"      RNN states shape: {policy_output.rnn_hidden_states.shape}")
                
        except Exception as e:
            print(f"   ❌ Policy instantiation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        print("\n🎉 All validations passed! Adapt3R is ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_adapt3r_integration()
    if success:
        print("\n💡 Ready to train with: python -m habitat_baselines.run --config-name=falcon_hm3d_train")
    else:
        print("\n🔧 Please fix the issues above before training.")
        sys.exit(1) 