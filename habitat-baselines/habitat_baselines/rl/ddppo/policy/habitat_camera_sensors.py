#!/usr/bin/env python3

import numpy as np
import torch
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.core.utils import not_none_validator
from hydra.core.config_store import ConfigStore

# Try different import paths for quaternion to matrix conversion
try:
    from habitat_sim.utils.common import quat_to_mat4
except ImportError:
    try:
        from habitat_sim.utils import quat_to_mat4
    except ImportError:
        # Fallback implementation
        def quat_to_mat4(quaternion):
            """Convert quaternion to 4x4 transformation matrix"""
            try:
                # Try to get components directly
                w, x, y, z = quaternion
            except (ValueError, TypeError):
                # If quaternion is a quaternion.quaternion object
                try:
                    w = float(quaternion.w)
                    x = float(quaternion.x)
                    y = float(quaternion.y)
                    z = float(quaternion.z)
                except AttributeError:
                    # If it's a numpy array
                    try:
                        w, x, y, z = quaternion.components
                    except AttributeError:
                        raise TypeError("Unsupported quaternion type")
            
            # Normalize quaternion
            norm = np.sqrt(w*w + x*x + y*y + z*z)
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
            
            # Convert to rotation matrix
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            matrix = np.eye(4, dtype=np.float32)
            matrix[0, 0] = 1 - 2*(yy + zz)
            matrix[0, 1] = 2*(xy - wz)
            matrix[0, 2] = 2*(xz + wy)
            matrix[1, 0] = 2*(xy + wz)
            matrix[1, 1] = 1 - 2*(xx + zz)
            matrix[1, 2] = 2*(yz - wx)
            matrix[2, 0] = 2*(xz - wy)
            matrix[2, 1] = 2*(yz + wx)
            matrix[2, 2] = 1 - 2*(xx + yy)
            
            return matrix


# ###############################################################
# 1. 定义配置结构体 (严格模仿你提供的文件)
# ###############################################################

@dataclass
class CameraIntrinsicsSensorConfig(LabSensorConfig):
    type: str = "CameraIntrinsicsSensor"
    # 我们需要的参数
    WIDTH: int = 256
    HEIGHT: int = 256
    HFOV: float = 90.0
    # 这个传感器不需要知道自己的摄像头名字，因为它计算的是通用的几何属性

@dataclass
class CameraExtrinsicsSensorConfig(LabSensorConfig):
    type: str = "CameraExtrinsicsSensor"
    # 这个传感器目前不需要额外参数，但定义一个类型是好习惯


# ###############################################################
# 2. 定义传感器类
# ###############################################################

@registry.register_sensor(name="CameraIntrinsicsSensor")
class CameraIntrinsicsSensor(Sensor):
    # 这个传感器不再需要 cls_uuid，因为它的身份由 type 决定
    cls_uuid = "camera_intrinsics"
    
    def __init__(self, sim, config: CameraIntrinsicsSensorConfig, **kwargs):
        self._sim = sim
        super().__init__(config=config)
        
    def _get_uuid(self, *args, **kwargs):
        # lab_sensors 的 UUID 是在配置文件中定义的键
        # 这里返回什么不重要，但不能为空
        return "camera_intrinsics"
        
    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR
        
    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32)
    
    def get_observation(self, *args, **kwargs):
        # 直接使用传入的config参数
        height = self.config.HEIGHT
        width = self.config.WIDTH
        hfov = self.config.HFOV
        
        # 基于HFOV的精确计算
        fx = width / (2.0 * np.tan(np.deg2rad(hfov / 2.0)))
        fy = fx # 假设正方形像素
        cx = width / 2.0
        cy = height / 2.0
        
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

@registry.register_sensor(name="CameraExtrinsicsSensor")
class CameraExtrinsicsSensor(Sensor):
    cls_uuid = "camera_extrinsics"

    def __init__(self, sim, config: CameraExtrinsicsSensorConfig, **kwargs):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return "camera_extrinsics"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32)

    def get_observation(self, *args, **kwargs):
        # 这个实现获取的是agent基座的位姿，对于移动机器人通常是合理的
        agent_state = self._sim.get_agent_state()
        position = agent_state.position
        rotation = agent_state.rotation
        
        T_world_agent = np.eye(4, dtype=np.float32)
        rotation_matrix = quat_to_mat4(rotation)
        T_world_agent[:3, :3] = rotation_matrix[:3, :3]
        T_world_agent[:3, 3] = position
        return T_world_agent

# ###############################################################
# 3. 注册配置到 ConfigStore (严格模仿你提供的文件)
# ###############################################################
cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.camera_intrinsics",
    group="habitat/task/lab_sensors",
    name="camera_intrinsics", # 使用一个简洁的名字
    node=CameraIntrinsicsSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.camera_extrinsics",
    group="habitat/task/lab_sensors",
    name="camera_extrinsics",
    node=CameraExtrinsicsSensorConfig,
)


# def print_registered_configs():
#     print("\n" + "="*80)
#     print("HYDRA CONFIG STORE - DEBUG DUMP (habitat/task/lab_sensors)")
#     print("="*80)
#     try:
#         cs_instance = ConfigStore.instance()
#         # 列出指定组下的所有已注册配置
#         registered_list = cs_instance.list("habitat/task/lab_sensors")
        
#         if not registered_list:
#             print("  [!] No configs found in group 'habitat/task/lab_sensors'.")
#         else:
#             for item in registered_list:
#                 print(f"  - Registered config name: {item}")
                
#     except Exception as e:
#         print(f"  [!] Error while accessing ConfigStore: {e}")
#     print("="*80 + "\n")

# # 在模块加载时就打印出来
# print_registered_configs()