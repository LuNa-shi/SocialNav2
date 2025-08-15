# RGB和深度骨干网络学习率缩放功能

## 概述

这个功能允许为RGBD ResNet Policy中的RGB和深度骨干网络设置较低的学习率，而其他层（如分类器、RNN等）使用标准学习率。这对于保护预训练权重和稳定训练非常有用。

## 功能特点

1. **自动参数识别**: 自动识别RGB和深度骨干网络参数
2. **灵活的学习率缩放**: 可配置的缩放因子
3. **兼容现有代码**: 向后兼容，不影响现有功能
4. **详细日志**: 显示参数分组信息

## 使用方法

### 1. 配置文件设置

在您的配置文件中添加 `backbone_lr_scale` 参数：

```yaml
habitat_baselines:
  rl:
    ppo:
      lr: 3e-4  # 标准学习率
      backbone_lr_scale: 0.1  # 骨干网络学习率缩放因子
      # 其他PPO参数...
```

### 2. 缩放因子说明

- `backbone_lr_scale: 0.1`: 骨干网络学习率为标准学习率的10%
- `backbone_lr_scale: 0.05`: 骨干网络学习率为标准学习率的5%
- `backbone_lr_scale: 0.01`: 骨干网络学习率为标准学习率的1%
- `backbone_lr_scale: 0.2`: 骨干网络学习率为标准学习率的20%

### 3. 参数识别规则

系统会自动识别以下参数作为骨干网络参数：
- 包含 `rgb_backbone` 的参数（RGB骨干网络）
- 包含 `backbone` 的参数（深度骨干网络）
- 排除包含 `compression`、`fc`、`embedding`、`state_encoder` 的参数

## 代码修改

### PPO类修改

在 `habitat-baselines/habitat_baselines/rl/ppo/ppo.py` 中：

1. 添加了 `backbone_lr_scale` 参数
2. 修改了 `_create_optimizer` 方法来支持参数分组
3. 更新了 `from_config` 方法来读取配置

### RGBD ResNet Policy

在 `habitat-baselines/habitat_baselines/rl/ddppo/policy/rgbd_resnet_policy.py` 中：

1. 添加了 `freeze_depth_backbone` 功能
2. 支持冻结深度骨干网络参数

## 训练日志

训练时会显示参数分组信息：

```
Backbone params: 12345678, Other params: 98765432
Total Number of params to train: 111111110
```

## 使用建议

### 1. 初始训练
```yaml
backbone_lr_scale: 0.1  # 使用较低学习率保护预训练权重
```

### 2. 微调阶段
```yaml
backbone_lr_scale: 0.05  # 进一步降低学习率
```

### 3. 冻结深度骨干网络
```yaml
habitat_baselines:
  rl:
    ddppo:
      freeze_depth_backbone: True  # 完全冻结深度骨干网络
```

### 4. 完全解冻
```yaml
backbone_lr_scale: 1.0  # 所有参数使用相同学习率
```

## 测试

运行测试脚本来验证参数分组：

```bash
python test_backbone_lr.py
```

## 注意事项

1. 确保您的配置文件中包含 `backbone_lr_scale` 参数
2. 如果未设置，默认值为 0.1
3. 参数识别基于参数名称，确保骨干网络参数名称正确
4. 冻结的深度骨干网络参数不会参与训练

## 故障排除

### 问题1: 骨干网络参数未被识别
- 检查参数名称是否包含 `rgb_backbone` 或 `backbone`
- 确保参数名称不包含排除关键词

### 问题2: 学习率设置无效
- 检查配置文件中的 `backbone_lr_scale` 值
- 确保值在合理范围内（建议 0.01-1.0）

### 问题3: 训练不稳定
- 尝试降低 `backbone_lr_scale` 值
- 检查是否同时使用了 `freeze_depth_backbone: True` 