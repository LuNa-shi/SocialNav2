import torch
import torch.nn.functional as F
import numpy as np
import einops
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

class PointCloudUtils:
    """
    Replicated point cloud utility functions from adapt3r_policy.py.
    """
    @staticmethod
    def depth2fgpcd_batch(depth, cam_params):
        B, ncam, h, w = depth.shape
        fx = cam_params[..., 0, 0].view(B, ncam, 1, 1)
        fy = cam_params[..., 1, 1].view(B, ncam, 1, 1)
        cx = cam_params[..., 0, 2].view(B, ncam, 1, 1)
        cy = cam_params[..., 1, 2].view(B, ncam, 1, 1)
        pos_y, pos_x = torch.meshgrid(torch.arange(h, device=depth.device, dtype=torch.float32), torch.arange(w, device=depth.device, dtype=torch.float32), indexing='ij')
        pos_x = pos_x.expand(B, ncam, -1, -1)
        pos_y = pos_y.expand(B, ncam, -1, -1)
        x_coords = (pos_x - cx) * depth / fx
        y_coords = (pos_y - cy) * depth / fy
        pcd_cam = torch.stack([x_coords, y_coords, depth], dim=-1)
        return einops.rearrange(pcd_cam, 'b ncam h w c -> b ncam (h w) c')

    @staticmethod
    def batch_transform_point_cloud(pcd, transform):
        pcd_homo = F.pad(pcd, (0, 1), mode='constant', value=1.0)
        transform = transform.to(dtype=pcd.dtype)
        trans_pcd_homo = torch.einsum('bn...d,bn...id->bn...i', pcd_homo, transform)
        return trans_pcd_homo[..., :-1]

    @staticmethod
    def lift_point_cloud_batch(depths, intrinsics, extrinsics):
        pcd_cam = PointCloudUtils.depth2fgpcd_batch(depths, intrinsics)
        trans_pcd = PointCloudUtils.batch_transform_point_cloud(pcd_cam, extrinsics)
        return trans_pcd

def generate_depth_map(shape_type='ball'):
    """Generates a depth map of a specified shape ('ball' or 'cone')."""
    H, W = 128, 128
    B, NCAM = 1, 1
    radius = H / 3
    center_x, center_y = W / 2, H / 2
    shape_base_depth = 2.0
    background_depth = 10.0 # Increased for clearer separation

    x_grid, y_grid = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x_grid - center_y)**2 + (y_grid - center_x)**2)
    depth_numpy = np.full((H, W), background_depth, dtype=np.float32)

    mask = dist_from_center <= radius

    if shape_type == 'ball':
        z_offset = np.sqrt(radius**2 - dist_from_center[mask]**2)
        depth_numpy[mask] = shape_base_depth + z_offset
    elif shape_type == 'cone':
        depth_numpy[mask] = shape_base_depth + dist_from_center[mask]
    
    depth_tensor = torch.from_numpy(depth_numpy).float().view(B, NCAM, H, W)

    fx, fy = 64, 64
    cx, cy = W / 2, H / 2
    intrinsics = torch.eye(3).repeat(B, NCAM, 1, 1)
    intrinsics[:, :, 0, 0] = fx
    intrinsics[:, :, 1, 1] = fy
    intrinsics[:, :, 0, 2] = cx
    intrinsics[:, :, 1, 2] = cy
    extrinsics = torch.eye(4).repeat(B, NCAM, 1, 1)
    
    return depth_tensor, intrinsics, extrinsics, depth_numpy

def main():
    """Main function to run the visualizations."""
    # 1. Generate Data
    ball_data = generate_depth_map(shape_type='ball')
    cone_data = generate_depth_map(shape_type='cone')
    
    # 2. Visualize Input Depth Images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(ball_data[3], cmap='plasma')
    ax1.set_title('Depth Image of a Ball')
    ax2.imshow(cone_data[3], cmap='plasma')
    ax2.set_title('Depth Image of a Cone (圆锥)')
    plt.show()

    # 3. Generate Point Clouds
    pcd_ball_full = PointCloudUtils.lift_point_cloud_batch(ball_data[0], ball_data[1], ball_data[2]).squeeze().numpy()
    pcd_cone_full = PointCloudUtils.lift_point_cloud_batch(cone_data[0], cone_data[1], cone_data[2]).squeeze().numpy()

    # 4. Create color map from original pixel coordinates
    H, W = ball_data[3].shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    
    # Flatten masks and colors
    ball_mask = (ball_data[3] < 9.9).flatten()
    cone_mask = (cone_data[3] < 9.9).flatten()
    colors_flat = x_coords.flatten()

    # 5. Filter points and colors to show only the shapes
    pcd_ball = pcd_ball_full[ball_mask]
    colors_ball = colors_flat[ball_mask]
    
    pcd_cone = pcd_cone_full[cone_mask]
    colors_cone = colors_flat[cone_mask]

    # 6. Interactive 3D Visualization with Plotly
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('3D Ball (Colored by Image X-Coord)', '3D Cone (Colored by Image X-Coord)')
    )

    fig.add_trace(go.Scatter3d(x=pcd_ball[:, 0], y=pcd_ball[:, 1], z=pcd_ball[:, 2], mode='markers', marker=dict(size=2, color=colors_ball, colorscale='Viridis')), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=pcd_cone[:, 0], y=pcd_cone[:, 1], z=pcd_cone[:, 2], mode='markers', marker=dict(size=2, color=colors_cone, colorscale='Viridis')), row=1, col=2)

    fig.update_layout(
        title_text='Interactive Side-by-Side Comparison',
        height=700,
        scene1=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectratio=dict(x=1, y=1, z=1)),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectratio=dict(x=1, y=1, z=1))
    )
    
    fig.show()

if __name__ == '__main__':
    main()
