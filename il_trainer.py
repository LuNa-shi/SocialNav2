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
from collections import defaultdict, deque
import time

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
DATASET_ROOT = "/root/zwj/Falcon/falcon_imitation_data/20250807_183821"
RGB_DATA_DIR = os.path.join(DATASET_ROOT, "jaw_rgb_data")
DEPTH_DATA_DIR = os.path.join(DATASET_ROOT, "jaw_depth_data") 
OTHER_DATA_DIR = os.path.join(DATASET_ROOT, "other_data")
TOPDOWN_MAP_DIR = os.path.join(DATASET_ROOT, "topdown_map")

# Model parameters
HIDDEN_SIZE = 512
RNN_TYPE = "LSTM"
NUM_RECURRENT_LAYERS = 2
# Note: backbone will be set to "resnet50" in model initialization to match pretrained weights

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 50
SEQUENCE_LENGTH = 30  # Number of timesteps per sequence (set to fit shortest episodes)
MAX_GRAD_NORM = 0.2
WEIGHT_DECAY = 1e-6

# Distributed training (DDP) switches
USE_DDP = True                 # Enable DistributedDataParallel
DDP_BACKEND = "nccl"           # Backend: nccl/gloo (nccl for GPUs)
DDP_FP16 = False               # Optional: enable torch.cuda.amp autocast
DDP_FIND_UNUSED = False        # Set True if graph has unused params
DDP_BROADCAST_BUFFERS = False  # Avoid broadcasting running stats buffers
DDP_INIT_METHOD = "env://"     # Use environment variables for init

# Action space parameters
NUM_ACTIONS = 4  # [0: STOP, 1: MOVE_FORWARD, 2: TURN_LEFT, 3: TURN_RIGHT]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
MODEL_SAVE_PATH = "il_model_final.pth"
CHECKPOINT_DIR = "outputs"
SAVE_BEST = True                  # Save best checkpoint instead of only last
SAVE_BEST_BY = "train_accuracy"       # train_loss or train_accuracy
SAVE_LAST = True                  # Also save last epoch model
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
        # Only use depth input - remove RGB for simplicity
        'agent_0_articulated_agent_jaw_depth': spaces.Box(
            low=0.0, high=10.0, shape=(240, 228, 1), dtype=np.float32
        ),
        # Use standard Habitat GPS compass sensor key to enable automatic processing
        'pointgoal_with_gps_compass': spaces.Box(
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
        # Use ResNet50 BottleNeck architecture to match Habitat pretrained weights
        self.core_network = PointNavResNetNet(
            observation_space=observation_space,
            action_space=action_space,  # keep original action space (4), prev_action_embedding will be resized by checkpoint if aligned
            hidden_size=HIDDEN_SIZE,
            num_recurrent_layers=NUM_RECURRENT_LAYERS,
            rnn_type=RNN_TYPE,
            backbone="resnet50",  # Use ResNet50 BottleNeck architecture
            resnet_baseplanes=32,  # Match Habitat pretrained weights baseplanes
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
        Load Habitat pretrained weights only (simplified approach).
        Using depth-only input for simplicity.
        """
        print("\n=== Loading Habitat Pretrained Weights ===")
        
        # Load Habitat pretrained weights
        print("1. Loading Habitat pretrained weights...")
        habitat_path = "/root/zjm/SocialNav2/pretrained_model/pretrained_habitat3.pth"
        try:
            habitat_checkpoint = torch.load(habitat_path, map_location='cpu')
            if isinstance(habitat_checkpoint, dict) and 'state_dict' in habitat_checkpoint:
                habitat_state_dict = habitat_checkpoint['state_dict']
            else:
                habitat_state_dict = habitat_checkpoint
            print(f"   ✓ Loaded Habitat weights ({len(habitat_state_dict)} keys)")
            
            # Debug: Show first few habitat keys
            print("   Debug - First 10 Habitat keys:")
            for i, key in enumerate(list(habitat_state_dict.keys())[:10]):
                print(f"     {i+1}. {key}")
                
        except Exception as e:
            print(f"   ✗ Failed to load Habitat weights: {e}")
            habitat_state_dict = {}
            return
        
        # Create unified weight dictionary
        print("2. Creating unified weight dictionary...")
        unified_state_dict = {}
        
        # Get current model's state dict for reference
        current_state_dict = self.state_dict()
        
        # Track loading statistics
        loaded_from_habitat = 0
        not_loaded = 0
        
        for key in current_state_dict.keys():
            loaded = False
            
            # Try direct key match first
            if key in habitat_state_dict:
                unified_state_dict[key] = habitat_state_dict[key]
                loaded_from_habitat += 1
                loaded = True
                print(f"   Direct match: {key}")
            else:
                # Try with Habitat prefix mapping
                # Our model: core_network.xxx -> Habitat: actor_critic.net.xxx
                habitat_key = key.replace('core_network.', 'actor_critic.net.')
                if habitat_key in habitat_state_dict:
                    unified_state_dict[key] = habitat_state_dict[habitat_key]
                    loaded_from_habitat += 1
                    loaded = True
                    print(f"   Habitat match: {key} <- {habitat_key}")
                else:
                    # Try other common prefixes as fallback
                    for prefix in ['net.', 'policy.']:
                        fallback_key = prefix + key.replace('core_network.', '')
                        if fallback_key in habitat_state_dict:
                            unified_state_dict[key] = habitat_state_dict[fallback_key]
                            loaded_from_habitat += 1
                            loaded = True
                            print(f"   Fallback match: {key} <- {fallback_key}")
                            break
            
            if not loaded:
                not_loaded += 1
                print(f"   Not found: {key}")
        
        # Explicitly map action head from RL checkpoint (better warm start)
        ah_w = 'action_head.weight'
        ah_b = 'action_head.bias'
        rl_w = 'actor_critic.action_distribution.linear.weight'
        rl_b = 'actor_critic.action_distribution.linear.bias'
        if rl_w in habitat_state_dict and rl_b in habitat_state_dict:
            unified_state_dict[ah_w] = habitat_state_dict[rl_w]
            unified_state_dict[ah_b] = habitat_state_dict[rl_b]
            print(f"   Map action head: {ah_w} <- {rl_w}")
            print(f"   Map action head: {ah_b} <- {rl_b}")
        else:
            print("   RL action head not found in checkpoint; keep IL head init")
        
        
        # Load the unified weights with strict=True to ensure complete matching
        print("3. Loading unified weights into model...")
        try:
            missing_keys, unexpected_keys = self.load_state_dict(unified_state_dict, strict=True)
            print("   ✓ All weights loaded successfully with strict=True")
        except RuntimeError as e:
            print(f"   ✗ Strict loading failed: {e}")
            print("   → Falling back to strict=False for debugging...")
            missing_keys, unexpected_keys = self.load_state_dict(unified_state_dict, strict=False)
        
        # Print loading report
        print("\n=== Weight Loading Report ===")
        print(f"Total model parameters: {len(current_state_dict)}")
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
        
        print("=== Habitat Weight Loading Complete ===\n")
    
 


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
        
        # Load all data for this episode (RGB removed for simplicity)
        depth_data = self._load_depth_data(episode_name)
        other_data = self._load_other_data(episode_name)
        # topdown_data = self._load_topdown_data(episode_name)  # Temporarily disabled
        
        # Extract sequence
        end_idx = start_idx + self.sequence_length
        
        # Depth observations: Keep as (seq_len, H, W, C), resize to target size
        depth_seq = depth_data[start_idx:end_idx]  # (seq_len, 256, 256, 1)
        depth_seq = torch.from_numpy(depth_seq).float()
        # Resize depth to target size: (256, 256) -> (240, 228)
        depth_seq = torch.nn.functional.interpolate(
            depth_seq.permute(0, 3, 1, 2),  # Temp permute for interpolate
            size=(240, 228), 
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)  # Back to (seq_len, 240, 228, 1)
        
        # Top-down map: Temporarily disabled due to inconsistent sizes
        # topdown_seq = topdown_data[start_idx:end_idx]  # (seq_len, 256, 303)
        # topdown_seq = torch.from_numpy(topdown_seq).float()  # Keep original range for consistency
        
        # GPS compass: load from other_data if available; otherwise zeros
        if 'agent_0_pointgoal_with_gps_compass' in other_data:
            gps_arr = other_data['agent_0_pointgoal_with_gps_compass'][start_idx:end_idx]
            gps_compass_seq = torch.from_numpy(gps_arr).float()
        else:
            gps_compass_seq = torch.zeros(self.sequence_length, 2, dtype=torch.float32)
        
        # Actions: Use raw actions [0,1,2,3] directly for CrossEntropyLoss
        raw_actions = other_data['actions'][start_idx:end_idx].squeeze(-1)
        
        # If actions in new dataset are [1,2,3], shift to [0,1,2]
        if raw_actions.min() >= 1 and raw_actions.max() <= 3:
            raw_actions = raw_actions - 1
        
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
        
        # Build observations dictionary (RGB removed for simplicity)
        observations = {
            'agent_0_articulated_agent_jaw_depth': depth_seq,
            'pointgoal_with_gps_compass': gps_compass_seq,
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
    
    # If DDP, use DistributedSampler and disable shuffle in DataLoader
    if USE_DDP and torch.cuda.is_available() and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
    else:
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
    # 0. (Optional) Initialize distributed training
    # -------------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = USE_DDP and torch.cuda.is_available() and world_size > 1
    if is_ddp:
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.distributed.init_process_group(backend=DDP_BACKEND, init_method=DDP_INIT_METHOD)
        is_main_process = (torch.distributed.get_rank() == 0)
    else:
        is_main_process = True
    
    # -------------------------------------------------------------------------
    # 1. Setup: Device, Spaces, Dataset, and Model
    # -------------------------------------------------------------------------
    
    if is_main_process:
        print(f"\n1. Setting up training environment...")
        print(f"Device: {DEVICE}")
        print(f"Dataset root: {DATASET_ROOT}")
    
    # Create observation and action spaces
    observation_space, action_space = get_observation_spaces()
    if is_main_process:
        print(f"Observation space keys: {list(observation_space.spaces.keys())}")
        print(f"Action space: {action_space}")
    
    # Create dataset and dataloader
    dataloader = create_dataloader(DATASET_ROOT, BATCH_SIZE)
    if is_main_process:
        print(f"Training data ready: {len(dataloader)} batches per epoch")
    
    # Initialize the model
    model = ImitationLearningPolicy(observation_space, action_space).to(DEVICE)
    
    # Wrap with DDP if enabled
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank % torch.cuda.device_count()],
            output_device=local_rank % torch.cuda.device_count(),
            find_unused_parameters=DDP_FIND_UNUSED,
            broadcast_buffers=DDP_BROADCAST_BUFFERS,
        )
    
    # Count total parameters (only main process)
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # -------------------------------------------------------------------------
    # 2. Optimizer and Loss Function
    # -------------------------------------------------------------------------
    
    if is_main_process:
        print(f"\n2. Setting up optimizer and loss function...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8
    )
    criterion = nn.CrossEntropyLoss(reduction='mean')
    if is_main_process:
        print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
        print(f"Loss function: CrossEntropyLoss")
        print(f"Gradient clipping: max_norm={MAX_GRAD_NORM}")
    
    scaler = torch.cuda.amp.GradScaler(enabled=DDP_FP16)
    
    # -------------------------------------------------------------------------
    # 3. Training Loop
    # -------------------------------------------------------------------------
    
    if is_main_process:
        print(f"\n3. Starting training loop...")
        print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, Sequence length: {SEQUENCE_LENGTH}")
        print("-" * 80)
    
    if is_ddp and hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
        dataloader.sampler.set_epoch(0)
    
    model.train()
    global_iter = 0
    t_start = time.time()
    loss_window = deque(maxlen=50)
    acc_window = deque(maxlen=50)
    
    for epoch in range(NUM_EPOCHS):
        if is_ddp and hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            observations = {k: v.to(DEVICE) for k, v in batch_data['observations'].items()}
            actions = batch_data['actions'].to(DEVICE)
            masks = batch_data['masks'].to(DEVICE)
            batch_size, seq_len = actions.shape
            
            rnn_hidden_states = model.module.get_initial_hidden_state(batch_size) if is_ddp else model.get_initial_hidden_state(batch_size)
            prev_actions = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for t in range(seq_len):
                current_obs = {k: v[:, t] for k, v in observations.items()}
                current_actions = actions[:, t]
                # Use boolean (B,1) masks to satisfy prev_action embedding (torch.where expects bool)
                # and also keep single-step RNN path
                current_masks = masks[:, t].bool()
                
                with torch.cuda.amp.autocast(enabled=DDP_FP16):
                    action_logits, rnn_hidden_states = (model.module if is_ddp else model)(
                        observations=current_obs,
                        rnn_hidden_states=rnn_hidden_states,
                        prev_actions=prev_actions,
                        masks=current_masks,
                    )
                    step_loss = criterion(action_logits, current_actions)
                
                total_loss += step_loss
                predicted_actions = torch.argmax(action_logits, dim=1)
                correct_predictions += (predicted_actions == current_actions).sum().item()
                total_predictions += batch_size
                prev_actions = current_actions
            
            avg_loss = total_loss / seq_len
            optimizer.zero_grad()
            scaler.scale(avg_loss).backward() if DDP_FP16 else avg_loss.backward()
            
            # Unscale grads for AMP before clipping and compute grad norm
            if DDP_FP16:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            scaler.step(optimizer) if DDP_FP16 else optimizer.step()
            if DDP_FP16:
                scaler.update()
            
            # Learning rate (assume single param group)
            cur_lr = optimizer.param_groups[0]["lr"]
            
            batch_loss = avg_loss.item()
            batch_accuracy = correct_predictions / total_predictions
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            num_batches += 1
            global_iter += 1
            loss_window.append(batch_loss)
            acc_window.append(batch_accuracy)
            
            if is_main_process and (batch_idx + 1) % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                it_per_sec = global_iter / max(1e-6, elapsed)
                remaining_iters = ((NUM_EPOCHS - (epoch + 1)) * len(dataloader)) + (len(dataloader) - (batch_idx + 1))
                eta_hours = (remaining_iters / max(1e-6, it_per_sec)) / 3600.0
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | iter {global_iter} ({batch_idx+1}/{len(dataloader)}) | "
                    f"loss {batch_loss:.4f} (avg {np.mean(loss_window):.4f}) | "
                    f"acc {batch_accuracy:.4f} (avg {np.mean(acc_window):.4f}) | "
                    f"lr {cur_lr:.2e} | grad {float(grad_norm):.2f} | "
                    f"{it_per_sec:.2f} it/s | ETA {eta_hours:.2f}h"
                )
        
        # Compute epoch metrics
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        avg_epoch_accuracy = epoch_accuracy / max(1, num_batches)
        
        # Save best/last (main process only)
        if is_main_process:
            if SAVE_BEST:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                best_path = os.path.join(CHECKPOINT_DIR, "il_model_best.pth")
                metric = avg_epoch_loss if SAVE_BEST_BY == "train_loss" else avg_epoch_accuracy
                better = (lambda cur, best: cur < best) if SAVE_BEST_BY == "train_loss" else (lambda cur, best: cur > best)
                # Track best in function static attribute
                if not hasattr(train, "_best_metric"):
                    train._best_metric = float("inf") if SAVE_BEST_BY == "train_loss" else float("-inf")
                if better(metric, train._best_metric):
                    to_save = model.module if is_ddp else model
                    torch.save(to_save.state_dict(), best_path)
                    train._best_metric = metric
                    print(f"[BEST] Saved new best to {best_path} (metric={metric:.4f}, by={SAVE_BEST_BY})")
            
            if SAVE_LAST:
                to_save = model.module if is_ddp else model
                torch.save(to_save.state_dict(), MODEL_SAVE_PATH)
            
            print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Average Accuracy: {avg_epoch_accuracy:.4f}")
            print(f"  Total iters: {global_iter}")
    
    # -------------------------------------------------------------------------
    # 4. Save the final model (only main process)
    # -------------------------------------------------------------------------
    if is_main_process:
        to_save = model.module if is_ddp else model
        torch.save(to_save.state_dict(), MODEL_SAVE_PATH)
        print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    
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