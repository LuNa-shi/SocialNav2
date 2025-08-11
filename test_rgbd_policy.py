#!/usr/bin/env python3

"""
Test script for RGBD ResNet Policy integration.

This script tests the basic functionality of the RGBDResNetPolicy:
1. Policy instantiation
2. Forward pass
3. Checkpoint loading compatibility
"""

import torch
import numpy as np
from gym import spaces
from habitat_baselines.rl.ddppo.policy.rgbd_resnet_policy import RGBDResNetPolicy


def create_test_observation_space():
    """Create a test observation space with RGB and depth sensors."""
    return spaces.Dict({
        # RGB sensor
        "agent_0_articulated_agent_jaw_rgb": spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        ),
        # Depth sensor
        "agent_0_articulated_agent_jaw_depth": spaces.Box(
            low=0.0, high=10.0, shape=(256, 256, 1), dtype=np.float32
        ),
        # Navigation sensor
        "agent_0_pointgoal_with_gps_compass": spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        ),
    })


def create_test_observations(batch_size=1):
    """Create test observations."""
    return {
        "agent_0_articulated_agent_jaw_rgb": torch.randint(
            0, 256, (batch_size, 256, 256, 3), dtype=torch.uint8
        ),
        "agent_0_articulated_agent_jaw_depth": torch.rand(
            (batch_size, 256, 256, 1), dtype=torch.float32
        ) * 10.0,
        "agent_0_pointgoal_with_gps_compass": torch.randn(
            (batch_size, 2), dtype=torch.float32
        ),
    }


def test_policy_instantiation():
    """Test policy instantiation."""
    print("Testing policy instantiation...")
    
    observation_space = create_test_observation_space()
    action_space = spaces.Discrete(4)
    
    policy = RGBDResNetPolicy(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="GRU",
        backbone="resnet18",
        rgb_backbone="resnet18",
        normalize_visual_inputs=True,
    )
    
    print(f"✓ Policy created successfully")
    print(f"  - RGB keys: {policy.net.visual_encoder.rgb_keys}")
    print(f"  - Depth keys: {policy.net.visual_encoder.depth_keys}")
    print(f"  - Is blind: {policy.net.is_blind}")
    print(f"  - Output size: {policy.net.output_size}")
    
    return policy


def test_forward_pass(policy):
    """Test forward pass."""
    print("\nTesting forward pass...")
    
    batch_size = 2
    observations = create_test_observations(batch_size)
    
    # Create initial RNN states
    rnn_hidden_states = torch.zeros(
        policy.net.num_recurrent_layers,
        batch_size,
        policy.net.recurrent_hidden_size
    )
    
    # Create previous actions and masks
    prev_actions = torch.zeros((batch_size, 1), dtype=torch.long)
    masks = torch.ones((batch_size, 1), dtype=torch.bool)
    
    # Forward pass
    try:
        features, new_rnn_states, aux_loss_state = policy.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        print(f"✓ Forward pass successful")
        print(f"  - Features shape: {features.shape}")
        print(f"  - RNN states shape: {new_rnn_states.shape}")
        print(f"  - Aux loss keys: {list(aux_loss_state.keys())}")
        
        # Test action distribution
        action_distribution = policy.action_distribution(features)
        actions = action_distribution.sample()
        
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - Action values: {actions}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_checkpoint_loading(policy):
    """Test checkpoint loading compatibility."""
    print("\nTesting checkpoint loading compatibility...")
    
    checkpoint_path = "/root/swx/track2/Falcon/pretrained_model/falcon_noaux_25.pth"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  - Checkpoint keys: {len(checkpoint.keys())}")
        
        # Check for compatible keys
        policy_state_dict = policy.state_dict()
        compatible_keys = []
        incompatible_keys = []
        
        for key in checkpoint.keys():
            # Map checkpoint keys to policy keys
            # Checkpoint uses 'net.conv.xxx' format
            policy_key = key
            if policy_key in policy_state_dict:
                if checkpoint[key].shape == policy_state_dict[policy_key].shape:
                    compatible_keys.append(key)
                else:
                    incompatible_keys.append(f"{key}: {checkpoint[key].shape} vs {policy_state_dict[policy_key].shape}")
            else:
                # Try to find partial matches for depth backbone
                if "visual_encoder" in key:
                    depth_key = key.replace("visual_encoder", "visual_encoder.depth_backbone")
                    if depth_key in policy_state_dict:
                        if checkpoint[key].shape == policy_state_dict[depth_key].shape:
                            compatible_keys.append(f"{key} -> {depth_key}")
        
        print(f"  - Compatible keys: {len(compatible_keys)}")
        if len(compatible_keys) > 0:
            print(f"    Examples: {compatible_keys[:5]}")
        
        print(f"  - Incompatible keys: {len(incompatible_keys)}")
        if len(incompatible_keys) > 0:
            print(f"    Examples: {incompatible_keys[:3]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint loading failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("RGBD ResNet Policy Integration Test")
    print("=" * 60)
    
    # Test 1: Policy instantiation
    try:
        policy = test_policy_instantiation()
    except Exception as e:
        print(f"✗ Policy instantiation failed: {e}")
        return
    
    # Test 2: Forward pass
    forward_success = test_forward_pass(policy)
    if not forward_success:
        return
    
    # Test 3: Checkpoint loading
    test_checkpoint_loading(policy)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()