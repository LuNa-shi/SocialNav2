#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .resnet_policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from .fixed_policy import (  # noqa: F401.
    FixedPolicy,
)
from .orca_policy import (  # noqa: F401.
    ORCAPolicy,
)
from .astar_policy import (  # noqa: F401.
    ASTARPolicy,
)

from .habitat_camera_sensors import (  # noqa: F401.
    CameraIntrinsicsSensor,
    CameraExtrinsicsSensor,
)

from .rgbd_resnet_policy import (  # noqa: F401.
    RGBDResNetNet,
    RGBDResNetPolicy,
)

from . import habitat_camera_sensors

# Ensure RGBD policy is imported so it registers with the baseline registry
from .rgbd_resnet_policy import RGBDResNetPolicy  # noqa: F401

# print('--------------------------------')
# print("CameraIntrinsicsSensor", CameraIntrinsicsSensor)
# print("CameraExtrinsicsSensor", CameraExtrinsicsSensor)
