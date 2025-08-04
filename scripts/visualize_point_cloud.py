import torch
import torch.nn.functional as F
import numpy as np
import einops
import plotly.graph_objects as go
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

def generate_ball_depth_map():
    """Generates a depth map of a ball."""
    H, W = 128, 128
    B, NCAM = 1, 1
    radius = H / 4
    center_x, center_y = W / 2, H / 2
    ball_depth_start = 2.0
    background_depth = 5.0

    x, y = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x - center_y)**2 + (y - center_x)**2)
    depth_numpy = np.full((H, W), background_depth, dtype=np.float32)

    ball_mask = dist_from_center <= radius
    z_offset = np.sqrt(radius**2 - dist_from_center[ball_mask]**2)
    depth_numpy[ball_mask] = ball_depth_start + (radius - z_offset)

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
    """Main function to run the visualization."""
    # 1. Generate Data
    depth_tensor, intrinsics, extrinsics, depth_numpy = generate_ball_depth_map()

    # 2. Visualize Input Depth Image
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_numpy, cmap='plasma')
    plt.title('Depth Image of a Ball')
    plt.colorbar(label='Depth')
    plt.show()

    # 3. Generate Point Cloud
    point_cloud = PointCloudUtils.lift_point_cloud_batch(depth_tensor, intrinsics, extrinsics)
    pcd_numpy = point_cloud.squeeze().detach().numpy()

    # 4. Interactive 3D Visualization using Plotly
    num_points_to_viz = 4096
    if pcd_numpy.shape[0] > num_points_to_viz:
        sample_indices = np.random.choice(pcd_numpy.shape[0], num_points_to_viz, replace=False)
        pcd_sample = pcd_numpy[sample_indices]
    else:
        pcd_sample = pcd_numpy

    fig = go.Figure(data=[
        go.Scatter3d(
            x=pcd_sample[:, 0],
            y=pcd_sample[:, 1],
            z=pcd_sample[:, 2],
            mode='markers',
            marker=dict(size=2, color=pcd_sample[:, 2], colorscale='Plasma', opacity=0.8)
        )
    ])

    fig.update_layout(
        title='Interactive 3D Point Cloud of a Ball',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(r=0, b=0, l=0, t=40)
    )

    fig.show()

if __name__ == '__main__':
    main()
