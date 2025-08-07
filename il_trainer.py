#!/usr/bin/env python3

"""
SocialNav2 Imitation Learning Trainer

This script performs Imitation Learning (IL) to train a navigation policy using 
real SocialNav2 dataset. The trained model weights will be saved and later used 
as a pre-trained model to initialize and accelerate Reinforcement Learning (RL).
"""

import os
import pickle
import glob
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gym
from gym import spaces

# Import from habitat-baselines
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import batch_obs

# ===============================================================================
# CONFIGURATION SECTION
# ===============================================================================

# Dataset paths
DATASET_ROOT = "/root/20250805_183840"
RGB_DATA_DIR = os.path.join(DATASET_ROOT, "jaw_rgb_data")
DEPTH_DATA_DIR = os.path.join(DATASET_ROOT, "jaw_depth_data") 
OTHER_DATA_DIR = os.path.join(DATASET_ROOT, "other_data")
TOPDOWN_MAP_DIR = os.path.join(DATASET_ROOT, "topdown_map")

# Model parameters
HIDDEN_SIZE = 512
RNN_TYPE = "LSTM"
NUM_RECURRENT_LAYERS = 2
BACKBONE = "resnet18"

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10
SEQUENCE_LENGTH = 50  # Number of timesteps per sequence
MAX_GRAD_NORM = 0.2
WEIGHT_DECAY = 1e-6

# Action space parameters
NUM_ACTIONS = 4  # [0: STOP, 1: MOVE_FORWARD, 2: TURN_LEFT, 3: TURN_RIGHT]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
MODEL_SAVE_PATH = "il_model_final.pth"
LOG_INTERVAL = 10  # Print training progress every N batches

print(f"Configuration loaded. Using device: {DEVICE}")
print(f"Dataset root: {DATASET_ROOT}")
print(f"Model will be saved to: {MODEL_SAVE_PATH}")


# ===============================================================================
# OBSERVATION SPACE HELPER FUNCTION
# ===============================================================================

def get_observation_spaces():
    """
    Create gym.spaces for the observations that our model expects.
    This matches the SocialNav2 dataset format.
    """
    observation_space = spaces.Dict({
        'agent_0_articulated_agent_jaw_rgb': spaces.Box(
            low=0, high=255, shape=(240, 228, 3), dtype=np.uint8
        ),
        'agent_0_articulated_agent_jaw_depth': spaces.Box(
            low=0.0, high=10.0, shape=(240, 228, 1), dtype=np.float32  # Resized to match RGB
        ),
        # Temporarily removed top_down_map due to inconsistent sizes across episodes
        # 'top_down_map': spaces.Box(
        #     low=0, high=255, shape=(256, 303), dtype=np.uint8
        # ),
        # Placeholder for GPS compass - will be filled when data is ready
        'agent_0_pointgoal_with_gps_compass': spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        ),
    })
    
    action_space = spaces.Discrete(NUM_ACTIONS)
    
    return observation_space, action_space


# ===============================================================================
# IMITATION LEARNING POLICY MODEL
# ===============================================================================

class ImitationLearningPolicy(nn.Module):
    """
    Imitation Learning Policy that uses PointNavResNetNet as backbone.
    
    This model takes multi-modal observations (RGB, Depth, TopDown map, GPS compass)
    and outputs action logits for imitation learning.
    """
    
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Discrete):
        super().__init__()
        
        # Store spaces
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Initialize the core PointNavResNetNet
        self.core_network = PointNavResNetNet(
            observation_space=observation_space,
            action_space=action_space,  # for previous action embedding
            hidden_size=HIDDEN_SIZE,
            num_recurrent_layers=NUM_RECURRENT_LAYERS,
            rnn_type=RNN_TYPE,
            backbone=BACKBONE,
            resnet_baseplanes=32,  # Standard value for ResNet18
            normalize_visual_inputs=True,
            fuse_keys=None,
            force_blind_policy=False,
            discrete_actions=True,
        )
        
        # Action head - single linear layer for action prediction
        # This replaces the CategoricalNet used in RL framework
        self.action_head = nn.Linear(HIDDEN_SIZE, action_space.n)
        
        # Initialize weights to match CategoricalNet initialization
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.constant_(self.action_head.bias, 0)
        
        # Load mixed pretrained weights
        self._load_mixed_pretrained_weights()
        
        print(f"ImitationLearningPolicy initialized:")
        print(f"  - Observation space: {observation_space.spaces.keys()}")
        print(f"  - Action space: {action_space.n} discrete actions")
        print(f"  - Hidden size: {HIDDEN_SIZE}")
        print(f"  - RNN type: {RNN_TYPE}")
    
    def forward(
        self, 
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy.
        
        Args:
            observations: Dict of observation tensors
            rnn_hidden_states: RNN hidden states from previous timestep
            prev_actions: Previous actions taken
            masks: Episode masks (1.0 for continue, 0.0 for episode end)
            
        Returns:
            action_logits: Raw action logits for cross-entropy loss
            rnn_hidden_states: Updated RNN hidden states
        """
        
        # Pass through core network to get features and updated hidden states
        features, rnn_hidden_states, aux_loss_state = self.core_network(
            observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info=None
        )
        
        # Pass features through action head to get action logits
        action_logits = self.action_head(features)
        
        return action_logits, rnn_hidden_states
    
    def get_initial_hidden_state(self, batch_size: int) -> torch.Tensor:
        """
        Get initial hidden state for RNN.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state tensor (for LSTM: packed format [h, c] concatenated)
        """
        # For LSTM, we need both hidden state (h) and cell state (c)
        # But habitat-baselines expects them packed into a single tensor
        # Shape: (num_layers, batch_size, hidden_size) each
        h = torch.zeros(
            NUM_RECURRENT_LAYERS,
            batch_size, 
            HIDDEN_SIZE,
            device=DEVICE,
            dtype=torch.float32
        )
        c = torch.zeros(
            NUM_RECURRENT_LAYERS,
            batch_size, 
            HIDDEN_SIZE,
            device=DEVICE,
            dtype=torch.float32
        )
        # Pack them into a single tensor: (2*num_layers, batch_size, hidden_size)
        return torch.cat([h, c], dim=0)
    
    def _load_mixed_pretrained_weights(self):
        """
        Load mixed pretrained weights:
        1. RGB backbone: ImageNet pretrained weights
        2. Other components: Habitat pretrained weights
        """
        print("\n=== Loading Mixed Pretrained Weights ===")
        
        # 1. Load ImageNet pretrained weights for RGB backbone
        print("1. Loading ImageNet pretrained weights for RGB backbone...")
        try:
            import torchvision.models as models
            imagenet_resnet = models.resnet18(pretrained=True)
            imagenet_state_dict = imagenet_resnet.state_dict()
            print(f"   ✓ Loaded ImageNet ResNet18 weights ({len(imagenet_state_dict)} keys)")
        except Exception as e:
            print(f"   ✗ Failed to load ImageNet weights: {e}")
            imagenet_state_dict = {}
        
        # 2. Load Habitat pretrained weights
        print("2. Loading Habitat pretrained weights...")
        habitat_path = "/root/zjm/SocialNav2/pretrained_model/pretrained_habitat3.pth"
        try:
            habitat_checkpoint = torch.load(habitat_path, map_location='cpu')
            if isinstance(habitat_checkpoint, dict) and 'state_dict' in habitat_checkpoint:
                habitat_state_dict = habitat_checkpoint['state_dict']
            else:
                habitat_state_dict = habitat_checkpoint
            print(f"   ✓ Loaded Habitat weights ({len(habitat_state_dict)} keys)")
        except Exception as e:
            print(f"   ✗ Failed to load Habitat weights: {e}")
            habitat_state_dict = {}
        
        # 3. Create unified weight dictionary
        print("3. Creating unified weight dictionary...")
        unified_state_dict = {}
        
        # Get current model's state dict for reference
        current_state_dict = self.state_dict()
        
        # Track loading statistics
        loaded_from_imagenet = 0
        loaded_from_habitat = 0
        not_loaded = 0
        
        for key in current_state_dict.keys():
            loaded = False
            
            # Try to load RGB backbone weights from ImageNet
            if 'visual_encoder' in key and 'rgb' in key.lower():
                # Map habitat-baselines ResNet keys to torchvision ResNet keys
                imagenet_key = self._map_habitat_to_imagenet_key(key)
                if imagenet_key and imagenet_key in imagenet_state_dict:
                    unified_state_dict[key] = imagenet_state_dict[imagenet_key]
                    loaded_from_imagenet += 1
                    loaded = True
            
            # Try to load from Habitat weights (if not loaded from ImageNet)
            if not loaded:
                # Try direct key match first
                if key in habitat_state_dict:
                    unified_state_dict[key] = habitat_state_dict[key]
                    loaded_from_habitat += 1
                    loaded = True
                else:
                    # Try with different prefixes (e.g., 'actor_critic.net.' prefix)
                    for prefix in ['actor_critic.net.', 'net.', 'policy.']:
                        habitat_key = prefix + key
                        if habitat_key in habitat_state_dict:
                            unified_state_dict[key] = habitat_state_dict[habitat_key]
                            loaded_from_habitat += 1
                            loaded = True
                            break
            
            if not loaded:
                not_loaded += 1
        
        # 4. Load the unified weights with strict=False
        print("4. Loading unified weights into model...")
        missing_keys, unexpected_keys = self.load_state_dict(unified_state_dict, strict=False)
        
        # 5. Print loading report
        print("\n=== Weight Loading Report ===")
        print(f"Total model parameters: {len(current_state_dict)}")
        print(f"Loaded from ImageNet: {loaded_from_imagenet}")
        print(f"Loaded from Habitat: {loaded_from_habitat}")
        print(f"Not loaded (using default init): {not_loaded}")
        
        if missing_keys:
            print(f"\nMissing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
        
        if unexpected_keys:
            print(f"\nUnexpected keys ({len(unexpected_keys)}):")
            for key in unexpected_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")
        
        print("=== Mixed Weight Loading Complete ===\n")
    
    def _map_habitat_to_imagenet_key(self, habitat_key: str) -> str:
        """
        Map habitat-baselines ResNet key to torchvision ResNet key.
        
        Example mappings:
        'core_network.visual_encoder.rgb_encoder.backbone.layer1.0.conv1.weight' 
        -> 'layer1.0.conv1.weight'
        """
        # Remove habitat-specific prefixes
        key = habitat_key
        prefixes_to_remove = [
            'core_network.visual_encoder.rgb_encoder.backbone.',
            'core_network.visual_encoder.backbone.',
            'visual_encoder.rgb_encoder.backbone.',
            'visual_encoder.backbone.',
            'backbone.',
        ]
        
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        
        # Skip certain layers that don't exist in torchvision ResNet
        skip_patterns = ['final_layer', 'compression', 'running_mean_and_var']
        for pattern in skip_patterns:
            if pattern in key:
                return None
        
        # Map specific layer names if needed
        # (torchvision ResNet18 structure should mostly match)
        
        return key if key else None 


# ===============================================================================
# REAL DATASET LOADER
# ===============================================================================

class SocialNav2Dataset(Dataset):
    """
    Dataset loader for SocialNav2 real trajectory data.
    
    Loads multi-modal data from directories:
    - jaw_rgb_data/: RGB images
    - jaw_depth_data/: Depth images  
    - other_data/: Actions, rewards, and metadata
    - topdown_map/: Top-down map sequences (temporarily disabled due to size inconsistency)
    """
    
    def __init__(self, dataset_root: str, sequence_length: int = SEQUENCE_LENGTH):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_root: Root directory containing the four data folders
            sequence_length: Length of sequences to extract for training
        """
        self.dataset_root = dataset_root
        self.sequence_length = sequence_length
        
        # Data directories
        self.rgb_dir = os.path.join(dataset_root, "jaw_rgb_data")
        self.depth_dir = os.path.join(dataset_root, "jaw_depth_data")
        self.other_dir = os.path.join(dataset_root, "other_data")
        self.topdown_dir = os.path.join(dataset_root, "topdown_map")
        
        # Find all available episodes
        self.episode_files = self._find_episode_files()
        
        # Pre-compute valid sequence start indices for each episode
        self.valid_sequences = self._compute_valid_sequences()
        
        print(f"SocialNav2Dataset initialized:")
        print(f"  - Found {len(self.episode_files)} episodes")
        print(f"  - Total valid sequences: {len(self.valid_sequences)}")
        print(f"  - Sequence length: {self.sequence_length}")
    
    def _find_episode_files(self) -> List[str]:
        """Find all episode files that exist in all four data directories."""
        # Get all pickle files from other_data (this should have all episodes)
        other_files = glob.glob(os.path.join(self.other_dir, "*.pkl"))
        episode_names = []
        
        for file_path in other_files:
            # Extract episode name (e.g., "33ypawbKCQf.basis_ep000000")
            filename = os.path.basename(file_path)
            episode_name = filename.replace(".pkl", "")
            
            # Check if corresponding files exist in all directories
            rgb_file = os.path.join(self.rgb_dir, f"{episode_name}.pkl")
            depth_file = os.path.join(self.depth_dir, f"{episode_name}.pkl")
            topdown_file = os.path.join(self.topdown_dir, f"{episode_name}.pkl")
            
            if all(os.path.exists(f) for f in [rgb_file, depth_file, topdown_file]):
                episode_names.append(episode_name)
            else:
                print(f"Warning: Missing data files for episode {episode_name}")
        
        return sorted(episode_names)
    
    def _compute_valid_sequences(self) -> List[Tuple[str, int]]:
        """
        Pre-compute all valid sequence starting positions.
        
        Returns:
            List of (episode_name, start_idx) tuples
        """
        valid_sequences = []
        
        for episode_name in self.episode_files:
            # Load the other_data to get episode length
            other_file = os.path.join(self.other_dir, f"{episode_name}.pkl")
            
            try:
                with open(other_file, 'rb') as f:
                    other_data = pickle.load(f)
                
                episode_length = len(other_data['actions'])
                
                # Generate all valid starting positions for this episode
                for start_idx in range(episode_length - self.sequence_length + 1):
                    valid_sequences.append((episode_name, start_idx))
                    
            except Exception as e:
                print(f"Error loading episode {episode_name}: {e}")
                continue
        
        return valid_sequences
    
    def __len__(self) -> int:
        """Return total number of valid sequences."""
        return len(self.valid_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence of data for training.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Dictionary containing:
            - observations: Dict with RGB, depth, topdown, GPS compass
            - actions: Expert actions for the sequence
            - masks: Episode continuation masks
        """
        episode_name, start_idx = self.valid_sequences[idx]
        
        # Load all data for this episode
        rgb_data = self._load_rgb_data(episode_name)
        depth_data = self._load_depth_data(episode_name)
        other_data = self._load_other_data(episode_name)
        # topdown_data = self._load_topdown_data(episode_name)  # Temporarily disabled
        
        # Extract sequence
        end_idx = start_idx + self.sequence_length
        
        # RGB observations: Keep as (seq_len, H, W, C) for ResNetEncoder
        rgb_seq = rgb_data[start_idx:end_idx]  # (seq_len, 240, 228, 3)
        rgb_seq = torch.from_numpy(rgb_seq).float()  # Keep original [0,255] range for RL framework
        
        # Depth observations: Keep as (seq_len, H, W, C) and resize to match RGB
        depth_seq = depth_data[start_idx:end_idx]  # (seq_len, 256, 256, 1)
        depth_seq = torch.from_numpy(depth_seq).float()
        # Resize depth to match RGB size: (256, 256) -> (240, 228)
        depth_seq = torch.nn.functional.interpolate(
            depth_seq.permute(0, 3, 1, 2),  # Temp permute for interpolate
            size=(240, 228), 
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)  # Back to (seq_len, 240, 228, 1)
        
        # Top-down map: Temporarily disabled due to inconsistent sizes
        # topdown_seq = topdown_data[start_idx:end_idx]  # (seq_len, 256, 303)
        # topdown_seq = torch.from_numpy(topdown_seq).float()  # Keep original range for consistency
        
        # GPS compass placeholder (will be filled when data is ready)
        gps_compass_seq = torch.zeros(self.sequence_length, 2, dtype=torch.float32)
        
        # Actions: Use raw actions [0,1,2,3] directly for CrossEntropyLoss
        raw_actions = other_data['actions'][start_idx:end_idx].squeeze(-1)
        
        # Debug: Check action value range
        unique_actions = np.unique(raw_actions)
        if len(unique_actions) > 0:
            min_action, max_action = unique_actions.min(), unique_actions.max()
            if min_action < 0 or max_action > 3:
                print(f"Warning: Unexpected action values in range [{min_action}, {max_action}]")
                print(f"Unique actions: {unique_actions}")
                # Clamp actions to valid range [0,1,2,3]
                raw_actions = np.clip(raw_actions, 0, 3)
        
        actions_seq = torch.from_numpy(raw_actions).long()  # Use actions directly: [0,1,2,3]
        
        # Final validation: ensure all actions are in [0,1,2,3]
        if actions_seq.min() < 0 or actions_seq.max() >= 4:
            print(f"Error: Invalid action range: [{actions_seq.min()}, {actions_seq.max()}]")
            actions_seq = torch.clamp(actions_seq, 0, 3)
        
        # Masks: Episode continuation masks
        masks_seq = other_data['masks'][start_idx:end_idx]
        masks_seq = torch.from_numpy(masks_seq).bool()  # Convert to boolean for torch.where
        
        # Build observations dictionary
        observations = {
            'agent_0_articulated_agent_jaw_rgb': rgb_seq,
            'agent_0_articulated_agent_jaw_depth': depth_seq,
            # 'top_down_map': topdown_seq,  # Temporarily disabled
            'agent_0_pointgoal_with_gps_compass': gps_compass_seq,
        }
        
        return {
            'observations': observations,
            'actions': actions_seq,
            'masks': masks_seq,
        }
    
    def _load_rgb_data(self, episode_name: str) -> np.ndarray:
        """Load RGB data for an episode."""
        rgb_file = os.path.join(self.rgb_dir, f"{episode_name}.pkl")
        with open(rgb_file, 'rb') as f:
            rgb_data = pickle.load(f)
        return rgb_data['agent_0_articulated_agent_jaw_rgb']
    
    def _load_depth_data(self, episode_name: str) -> np.ndarray:
        """Load depth data for an episode."""
        depth_file = os.path.join(self.depth_dir, f"{episode_name}.pkl")
        with open(depth_file, 'rb') as f:
            depth_data = pickle.load(f)
        return depth_data['agent_0_articulated_agent_jaw_depth']
    
    def _load_other_data(self, episode_name: str) -> Dict[str, np.ndarray]:
        """Load other data (actions, rewards, masks) for an episode."""
        other_file = os.path.join(self.other_dir, f"{episode_name}.pkl")
        with open(other_file, 'rb') as f:
            other_data = pickle.load(f)
        return other_data
    
    def _load_topdown_data(self, episode_name: str) -> np.ndarray:
        """Load top-down map data for an episode."""
        topdown_file = os.path.join(self.topdown_dir, f"{episode_name}.pkl")
        with open(topdown_file, 'rb') as f:
            topdown_data = pickle.load(f)
        return topdown_data['top_down_map']


def create_dataloader(dataset_root: str, batch_size: int = BATCH_SIZE) -> DataLoader:
    """
    Create a DataLoader for the SocialNav2 dataset.
    
    Args:
        dataset_root: Root directory of the dataset
        batch_size: Batch size for training
        
    Returns:
        DataLoader instance
    """
    dataset = SocialNav2Dataset(dataset_root, SEQUENCE_LENGTH)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # Adjust based on your system
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    print(f"DataLoader created: batch_size={batch_size}, num_workers=2")
    return dataloader


# ===============================================================================
# TRAINING FUNCTION
# ===============================================================================

def train():
    """
    Main training function that orchestrates the entire IL training process.
    
    This function:
    1. Sets up device, dataset, and model
    2. Initializes optimizer and loss function
    3. Runs the training loop with proper RNN state management
    4. Saves the final model
    """
    print("=" * 80)
    print("Starting SocialNav2 Imitation Learning Training")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Setup: Device, Spaces, Dataset, and Model
    # -------------------------------------------------------------------------
    
    print(f"\n1. Setting up training environment...")
    print(f"Device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    
    # Create observation and action spaces
    observation_space, action_space = get_observation_spaces()
    print(f"Observation space keys: {list(observation_space.spaces.keys())}")
    print(f"Action space: {action_space}")
    
    # Create dataset and dataloader
    dataloader = create_dataloader(DATASET_ROOT, BATCH_SIZE)
    print(f"Training data ready: {len(dataloader)} batches per epoch")
    
    # Initialize the model
    model = ImitationLearningPolicy(observation_space, action_space)
    model = model.to(DEVICE)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # -------------------------------------------------------------------------
    # 2. Optimizer and Loss Function
    # -------------------------------------------------------------------------
    
    print(f"\n2. Setting up optimizer and loss function...")
    
    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    
    # Cross-entropy loss for action classification
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Gradient clipping: max_norm={MAX_GRAD_NORM}")
    
    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    
    print(f"\n3. Starting training loop...")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, Sequence length: {SEQUENCE_LENGTH}")
    print("-" * 80)
    
    model.train()
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move batch to device
            observations = {}
            for key, value in batch_data['observations'].items():
                observations[key] = value.to(DEVICE)  # (batch_size, seq_len, ...)
            
            actions = batch_data['actions'].to(DEVICE)  # (batch_size, seq_len)
            masks = batch_data['masks'].to(DEVICE)      # (batch_size, seq_len)
            
            batch_size, seq_len = actions.shape
            
            # ----------------------------------------------------------------
            # RNN State Management and Sequence Processing
            # ----------------------------------------------------------------
            
            # Initialize RNN hidden states for this batch
            rnn_hidden_states = model.get_initial_hidden_state(batch_size)
            
            # Initialize previous actions (start with action 0)
            prev_actions = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
            
            # Accumulate loss over the entire sequence
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Process each timestep in the sequence
            for t in range(seq_len):
                # Prepare observations for current timestep
                current_obs = {}
                for key, value in observations.items():
                    current_obs[key] = value[:, t]  # (batch_size, ...)
                
                # Get current actions and masks
                current_actions = actions[:, t]  # (batch_size,)
                current_masks = masks[:, t]      # (batch_size,)
                
                # Forward pass through the model
                action_logits, rnn_hidden_states = model(
                    observations=current_obs,
                    rnn_hidden_states=rnn_hidden_states,
                    prev_actions=prev_actions,
                    masks=current_masks
                )
                
                # Compute loss for current timestep
                step_loss = criterion(action_logits, current_actions)
                total_loss += step_loss
                
                # Compute accuracy for current timestep
                predicted_actions = torch.argmax(action_logits, dim=1)
                correct_predictions += (predicted_actions == current_actions).sum().item()
                total_predictions += batch_size
                
                # Update previous actions for next timestep
                # Use ground truth actions for teacher forcing
                prev_actions = current_actions
                
                # Handle episode boundaries: reset hidden states where mask is 0
                # Note: This is handled automatically by the RNN through masks
                
            # Average loss over sequence length
            avg_loss = total_loss / seq_len
            
            # ----------------------------------------------------------------
            # Backpropagation and Optimization
            # ----------------------------------------------------------------
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Backward pass
            avg_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            # Optimizer step
            optimizer.step()
            
            # ----------------------------------------------------------------
            # Logging and Statistics
            # ----------------------------------------------------------------
            
            batch_loss = avg_loss.item()
            batch_accuracy = correct_predictions / total_predictions
            
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            num_batches += 1
            global_step += 1
            
            # Print progress every LOG_INTERVAL batches
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Batch [{batch_idx+1}/{len(dataloader)}] "
                      f"Loss: {batch_loss:.4f} "
                      f"Acc: {batch_accuracy:.4f} "
                      f"Step: {global_step}")
        
        # ----------------------------------------------------------------
        # End of Epoch Statistics
        # ----------------------------------------------------------------
        
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Accuracy: {avg_epoch_accuracy:.4f}")
        print(f"  Total Steps: {global_step}")
        print("-" * 80)
    
    # -------------------------------------------------------------------------
    # 4. Save Final Model
    # -------------------------------------------------------------------------
    
    print(f"\n4. Saving trained model...")
    
    # Save the model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'hidden_size': HIDDEN_SIZE,
            'rnn_type': RNN_TYPE,
            'num_recurrent_layers': NUM_RECURRENT_LAYERS,
            'backbone': BACKBONE,
            'num_actions': NUM_ACTIONS,
            'sequence_length': SEQUENCE_LENGTH,
        },
        'training_stats': {
            'final_loss': avg_epoch_loss,
            'final_accuracy': avg_epoch_accuracy,
            'total_epochs': NUM_EPOCHS,
            'total_steps': global_step,
        }
    }, MODEL_SAVE_PATH)
    
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Final training loss: {avg_epoch_loss:.4f}")
    print(f"Final training accuracy: {avg_epoch_accuracy:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


# ===============================================================================
# MAIN EXECUTION
# ===============================================================================

if __name__ == "__main__":
    """
    Main execution block.
    
    This script can be run directly to start the imitation learning training:
    python il_trainer.py
    """
    try:
        train()
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print("=" * 80)
        raise 