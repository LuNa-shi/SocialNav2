Adapt3R Policy Dataflow Test
This test validates tensor shapes and dataflow in the Adapt3R policy.

🚀 Starting Adapt3R Dataflow Tests
==================================================
=== Testing PointCloudUtils ===
Input depths shape: torch.Size([2, 2, 128, 160])
Input intrinsics shape: torch.Size([2, 2, 3, 3])
Camera point cloud shape: torch.Size([2, 2, 20480, 3])
Input extrinsics shape: torch.Size([2, 2, 4, 4])
World point cloud shape: torch.Size([2, 2, 20480, 3])
Final point cloud shape: torch.Size([2, 2, 20480, 3])
✅ PointCloudUtils tests passed!

=== Testing Adapt3REncoder Initialization ===
Encoder hidden dimension: 252
Encoder num points: 512
Encoder perception output size: 252
Encoder lowdim output size: 0
✅ Encoder initialization passed!

=== Testing Encoder Forward Pass (Step by Step) ===
Step 1: Building point cloud...
Point cloud shape: torch.Size([2, 2, 20480, 3])
Step 2: Extracting RGB features...
Number of cameras: 2
RGB tensor 0 shape: torch.Size([2, 3, 128, 160])
RGB tensor 1 shape: torch.Size([2, 3, 128, 160])
Concatenated RGB batch shape: torch.Size([4, 3, 128, 160])
Step 3: Backbone feature extraction...
Backbone output keys: ['layer1', 'layer2', 'layer3', 'layer4']
  layer1: torch.Size([4, 64, 32, 40])
  layer2: torch.Size([4, 128, 16, 20])
  layer3: torch.Size([4, 256, 8, 10])
  layer4: torch.Size([4, 512, 4, 5])
Step 4: FPN feature extraction...
FPN output keys: ['layer1', 'layer2', 'layer3', 'layer4']
  layer1: torch.Size([4, 252, 32, 40])
  layer2: torch.Size([4, 252, 16, 20])
  layer3: torch.Size([4, 252, 8, 10])
  layer4: torch.Size([4, 252, 4, 5])
⚠️ Expected key '0' not found, using 'layer1' instead
Selected FPN features shape: torch.Size([4, 252, 32, 40])
Step 5: Point cloud interpolation...
Feature map size: 32 x 40
Interpolated point cloud shape: torch.Size([2, 2, 1280, 3])
Step 6: Reshape and flatten...
Reshaped point cloud shape: torch.Size([2, 2, 32, 40, 3])
Flattened point cloud shape: torch.Size([2, 2560, 3])
Flattened RGB features shape: torch.Size([2, 2560, 252])
Step 7: Point cloud cropping...
Crop mask shape: torch.Size([2, 2560])
Valid points ratio: 0.599
Masked point cloud shape: torch.Size([2, 2560, 3])
Masked RGB features shape: torch.Size([2, 2560, 252])
Step 8: Point cloud downsampling...
Downsampled point cloud shape: torch.Size([2, 512, 3])
Downsampled features shape: torch.Size([2, 512, 252])
Step 9: Position encoding...
Position embeddings shape: torch.Size([2, 512, 252])
Step 10: Feature concatenation...
Added position embeddings: torch.Size([2, 512, 252])
Added image features: torch.Size([2, 512, 252])
Final concatenated features shape: torch.Size([2, 512, 504])
Step 11: Point cloud feature extraction...
Extracted features shape: torch.Size([2, 512, 252])
Step 12: Max pooling...
Final perception output shape: torch.Size([2, 252])
Step 13: Low-dim encoding...
No low-dim features
✅ Step-by-step forward pass completed!

=== Testing Full Forward Pass ===
Final perception output shape: torch.Size([2, 252])
Expected shape: (2, 252)
No lowdim output (as expected)
✅ Full forward pass test passed!

🎉 All tests passed successfully!