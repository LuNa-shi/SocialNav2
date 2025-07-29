# Adapt3R Policy 集成指南

## 概述

本指南说明如何在 Falcon 训练系统中使用 Adapt3R 点云策略。Adapt3R 是一个结合视觉和点云信息进行导航决策的深度强化学习策略。

## 主要修改

### 1. 策略配置 (`falcon_hm3d_train.yaml`)

```yaml
rl:
  policy:
    agent_0:
      name: "Adapt3RPolicy"  # 替换原有的 PointNavResNetPolicy
      action_distribution_type: "categorical"
      hidden_size: 512
      rnn_type: "GRU" 
      num_recurrent_layers: 1
      
      visual_encoder:
        backbone_type: "resnet18"     # 可选: resnet18, resnet50, clip
        hidden_dim: 252               # 必须被6整除 (NeRF 位置编码要求)
        num_points: 512               # 点云下采样点数
        do_image: True                # 使用视觉特征
        do_pos: True                  # 使用位置编码
        do_rgb: False                 # 不使用原始RGB值
        finetune: True                # 训练视觉编码器
        xyz_proj_type: "nerf"         # NeRF 位置编码
        do_crop: True                 # 点云裁剪
        boundaries: [[-5.0, -5.0, 0.0], [5.0, 5.0, 3.0]]  # 裁剪范围
        lowdim_obs_keys: 
          - "agent_0_pointgoal_with_gps_compass"
          - "agent_0_localization_sensor"
```

### 2. 观测空间配置

```yaml
habitat:
  gym:
    obs_keys: 
      # Adapt3R 必需的传感器
      - agent_0_articulated_agent_jaw_rgb         # RGB 相机
      - agent_0_articulated_agent_jaw_depth       # 深度相机
      - agent_0_articulated_agent_jaw_intrinsics  # 相机内参
      - agent_0_articulated_agent_jaw_extrinsics  # 相机外参
      # 导航传感器
      - agent_0_pointgoal_with_gps_compass
      - agent_0_localization_sensor
      - agent_0_human_num_sensor  
      - agent_0_oracle_humanoid_future_trajectory
```

### 3. 预训练权重配置

```yaml
ddppo:
  # Adapt3R: 无预训练权重
  pretrained: False
  pretrained_encoder: False  
  train_encoder: True
  reset_critic: True
```

## 新增组件

### 1. Adapt3R Policy (`adapt3r_policy.py`)
- **PointCloudUtils**: 点云处理工具
- **EnvUtils**: 环境传感器工具 (适配 Habitat 命名)
- **Adapt3REncoder**: 视觉-点云编码器
- **Adapt3RNet**: 网络架构 (编码器 + RNN)
- **Adapt3RPolicy**: 完整策略

### 2. 相机传感器 (`habitat_camera_sensors.py`)
- **CameraIntrinsicsSensor**: 相机内参传感器
- **CameraExtrinsicsSensor**: 相机外参传感器

## 使用方法

### 1. 验证集成

运行验证脚本检查所有组件是否正确配置：

```bash
cd Falcon
python validate_adapt3r_integration.py
```

### 2. 开始训练

如果验证通过，使用以下命令开始训练：

```bash
cd Falcon
python -m habitat_baselines.run --config-name=falcon_hm3d_train
```

### 3. 分布式训练

对于多GPU训练：

```bash
cd Falcon
python -m habitat_baselines.run --config-name=falcon_hm3d_train \
    habitat_baselines.rl.ddppo.force_distributed=True \
    habitat_baselines.num_environments=16
```

## 配置参数详解

### Adapt3R 特定参数

- **backbone_type**: 视觉特征提取器类型
  - `"resnet18"`: 轻量级，训练快
  - `"resnet50"`: 更强表现，计算量大  
  - `"clip"`: 预训练视觉-语言模型

- **hidden_dim**: 隐藏层维度 (必须被6整除)
- **num_points**: 点云下采样点数 (影响计算效率)
- **do_crop**: 是否裁剪点云 (建议启用)
- **boundaries**: 点云裁剪边界 `[[x_min, y_min, z_min], [x_max, y_max, z_max]]`

### 训练参数建议

- **学习率**: `2.5e-4` (默认)
- **批量大小**: `num_environments * num_steps` 
- **环境数**: 8-16 (根据GPU内存调整)
- **回合长度**: 128 steps (默认)

## 故障排除

### 常见问题

1. **内存不足**:
   - 减少 `num_environments`
   - 减少 `num_points`
   - 使用 `backbone_type: "resnet18"`

2. **训练速度慢**:
   - 使用更少的点云点数
   - 禁用点云裁剪 (`do_crop: False`)
   - 使用较小的图像分辨率

3. **数值不稳定**:
   - 使用 `deterministic=True` 进行调试
   - 检查观测值是否包含 NaN/Inf
   - 减少学习率

### 调试技巧

1. **可视化点云**:
   ```python
   # 在 Adapt3REncoder.forward 中添加
   print(f"Point cloud shape: {pcds_world.shape}")
   print(f"Point cloud range: [{pcds_world.min():.3f}, {pcds_world.max():.3f}]")
   ```

2. **检查传感器数据**:
   ```python
   # 在环境步骤中添加
   for key, value in observations.items():
       print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")
   ```

## 性能优化

### GPU 内存优化
- 使用混合精度训练 (如果支持)
- 分批处理大型点云
- 使用 gradient checkpointing

### 计算效率
- 预计算相机内参和外参
- 使用更高效的点云下采样方法
- 并行化多相机处理

## 扩展功能

### 添加新传感器
1. 在 `obs_keys` 中添加传感器名称
2. 在 `lowdim_obs_keys` 中包含低维传感器
3. 相应调整观测空间配置

### 自定义点云处理
1. 修改 `PointCloudUtils` 类
2. 调整裁剪边界和下采样策略
3. 实现自定义位置编码

### 多相机支持
1. 添加额外相机传感器
2. Adapt3R 自动处理多相机融合
3. 调整相机命名约定

---

**注意**: 此集成基于 Adapt3R 论文实现，适配了 Habitat 环境的传感器命名和数据格式。训练过程中请监控GPU内存使用和训练稳定性。 