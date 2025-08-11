# RGBD ResNet Policy vs Original ResNet Policy

## Overview

The **RGBDResNetPolicy** extends the original **PointNavResNetPolicy** to support dual-modality visual input (RGB + Depth) through separate ResNet backbones, while maintaining compatibility with existing pretrained checkpoints.

## Key Differences

### 1. Architecture Changes

| Component | Original ResNet Policy | RGBD ResNet Policy |
|-----------|----------------------|-------------------|
| **Visual Encoder** | Single ResNet backbone | Dual backbones (RGB + Depth) |
| **RGB Processing** | Combined with depth in single encoder | Separate ImageNet pretrained ResNet |
| **Depth Processing** | Custom ResNet with depth channels | Custom ResNet (same as original) |
| **Feature Fusion** | Single feature stream | Concatenated RGB + depth features |

### 2. Backbone Configuration

```python
# Original: Single backbone for all visual inputs
ResNetEncoder(
    observation_space,
    make_backbone=getattr(resnet, backbone)  # e.g., resnet18/resnet50
)

# RGBD: Dual backbones
RGBDResNetEncoder(
    observation_space,
    make_backbone=getattr(resnet, backbone),    # Depth backbone
    rgb_backbone="resnet18"                     # RGB backbone (ImageNet pretrained)
)
```

### 3. Input Processing

| Input Type | Original | RGBD |
|------------|----------|------|
| **RGB sensors** | Processed by single encoder | Processed by dedicated RGB backbone |
| **Depth sensors** | Processed by single encoder | Processed by dedicated depth backbone |
| **Feature fusion** | All features from one encoder | RGB features + depth features concatenated |

### 4. Pretrained Weight Loading

| Weights | Original | RGBD |
|---------|----------|------|
| **RGB backbone** | Part of single checkpoint | ImageNet pretrained (automatic) |
| **Depth backbone** | From checkpoint `net.xxx` | From checkpoint `net.xxx` (compatible) |
| **RNN/Policy** | From checkpoint | From checkpoint (compatible) |

## Configuration Changes

### Original Configuration
```yaml
rl:
  policy:
    agent_0:
      name: "PointNavResNetPolicy"
  
ddppo:
  backbone: "resnet50"  # Single backbone
  pretrained_weights: "pretrained_model/falcon_noaux_25.pth"
```

### RGBD Configuration
```yaml
rl:
  policy:
    agent_0:
      name: "RGBDResNetPolicy"
  
ddppo:
  backbone: "resnet50"      # Depth backbone
  rgb_backbone: "resnet18"  # RGB backbone (NEW)
  pretrained_weights: "pretrained_model/falcon_noaux_25.pth"

gym:
  obs_keys:
    - agent_0_articulated_agent_jaw_rgb    # RGB input (NEW)
    - agent_0_articulated_agent_jaw_depth  # Depth input (existing)
    - agent_0_pointgoal_with_gps_compass   # Navigation sensors
```

## Sensor Requirements

### Original Policy (Depth-only)
```yaml
gym:
  obs_keys:
    - agent_0_articulated_agent_jaw_depth
    - agent_0_pointgoal_with_gps_compass
```

### RGBD Policy (RGB + Depth)
```yaml
gym:
  obs_keys:
    - agent_0_articulated_agent_jaw_rgb      # Required for RGB backbone
    - agent_0_articulated_agent_jaw_depth    # Required for depth backbone
    - agent_0_pointgoal_with_gps_compass
```

## Benefits

1. **Richer Visual Representation**: RGB provides color/texture information, depth provides geometric structure
2. **Pretrained RGB Features**: Leverages ImageNet pretrained weights for better RGB understanding
3. **Backward Compatibility**: Depth backbone can load existing checkpoint weights
4. **Modular Design**: RGB and depth backbones can be configured independently

## Performance Considerations

| Aspect | Original | RGBD | Impact |
|--------|----------|------|--------|
| **Memory Usage** | 1x | ~1.8x | Higher due to dual backbones |
| **Computation** | 1x | ~1.8x | Dual forward passes |
| **Feature Dimensionality** | Single stream | Concatenated (higher) | More parameters in FC layers |

## Checkpoint Compatibility

```python
# Original checkpoint structure
falcon_noaux_25.pth:
  net.visual_encoder.xxx     → depth_backbone.xxx
  net.state_encoder.xxx      → state_encoder.xxx (compatible)
  actor_critic.xxx           → compatible

# RGBD policy loading
RGBDResNetPolicy:
  - RGB backbone: ImageNet weights (automatic)
  - Depth backbone: net.visual_encoder.xxx → visual_encoder.depth_backbone.xxx
  - RNN/Policy: Direct mapping (compatible)
```

## Usage Example

```bash
# Train with RGBD policy
python -m habitat_baselines.run --config-name=rgbd_hm3d_train

# Key config changes needed:
# 1. policy.name: "RGBDResNetPolicy" 
# 2. ddppo.rgb_backbone: "resnet18"
# 3. Add RGB sensor to obs_keys
```

## Migration Guide

To convert from original ResNet to RGBD:

1. **Change policy name**: `PointNavResNetPolicy` → `RGBDResNetPolicy`
2. **Add RGB backbone config**: `rgb_backbone: "resnet18"`
3. **Add RGB sensor**: Include RGB sensor in `obs_keys`
4. **Checkpoint loading**: Existing checkpoints work for depth backbone
5. **Memory allocation**: Increase GPU memory allocation (~80% more)

## Summary

The RGBD ResNet Policy enhances the original ResNet policy by:
- Adding dedicated RGB processing with ImageNet pretrained weights
- Maintaining depth processing compatibility with existing checkpoints  
- Providing richer multi-modal visual features for navigation
- Preserving the same RNN and policy architecture for easy migration