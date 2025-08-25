#!/usr/bin/env python3

"""
SocialNav2 Imitation Learning Trainer

This script implements imitation learning for the SocialNav2 task using expert demonstrations.
The training uses the official Habitat-baselines PackedSequence mechanism for proper RNN sequence processing.

Key Features:
- Complete episode training (no fixed-length sequences)
- PackedSequence-based RNN processing for proper temporal dependencies
- Multi-GPU support with DistributedDataParallel (DDP)
- Mixed precision training (FP16)
- Comprehensive logging and monitoring
"""

import os
import sys
import time
import pickle
import glob
import random
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import PackedSequence
from torch.utils.tensorboard import SummaryWriter

# Import Habitat-baselines components
sys.path.append('/root/zjm/SocialNav2/habitat-baselines')
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_episode_ids,
    build_rnn_build_seq_info,
    build_rnn_inputs,
    build_rnn_out_from_seq
)
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet

from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.tensor_dict import TensorDict
import gym
from gym import spaces

# ===============================================================================
# CONFIGURATION SECTION
# ===============================================================================

# Dataset paths
DATASET_ROOT = "/data/il_dataset/expert_data_20250813_064841"
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
BATCH_SIZE = 256
NUM_EPOCHS = 30
SEQUENCE_LENGTH = 3   # Use 3-frame window: [t-2, t-1, t]
MAX_GRAD_NORM = 0.2
WEIGHT_DECAY = 1e-6

SAMPLES_PER_EPOCH = 300000   # Total sample windows per epoch (controls __len__)
USE_RANDOM_SAMPLING = False  # obsolete after dataloader rewrite
MIN_OVERLAP_RATIO = 0.0      # placeholder (unused)
SEQUENCES_PER_EPISODE = 0  # deprecated after triple-frame rewrite

# Validation parameters
VAL_EPISODES_COUNT = 100        # Number of episodes for validation
VAL_BATCH_SIZE = 32            # Smaller batch size for validation to avoid OOM
VAL_INTERVAL = 1               # Run validation every N epochs

# DataLoader
NUM_WORKERS = 8 # Number of DataLoader workers

# Distributed training (DDP) switches
USE_DDP = True                 # Enable DistributedDataParallel
DDP_BACKEND = "nccl"           # Backend: nccl/gloo (nccl for GPUs)
DDP_FP16 = True               # Optional: enable torch.cuda.amp autocast
DDP_FIND_UNUSED = False        # Set True if graph has unused params
DDP_BROADCAST_BUFFERS = False  # Avoid broadcasting running stats buffers
DDP_INIT_METHOD = "env://"     # Use environment variables for init

# Action space parameters
NUM_ACTIONS = 4  # [0: STOP, 1: MOVE_FORWARD, 2: TURN_LEFT, 3: TURN_RIGHT]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output paths
MODEL_SAVE_PATH = "il_model_0820.pth"
CHECKPOINT_DIR = "checkpoints0820"
TENSORBOARD_DIR = "runs"
SAVE_BEST = True                  # Save best checkpoint instead of only last
SAVE_BEST_BY = "val_accuracy"         # train_loss, train_accuracy, val_loss, val_accuracy
SAVE_LAST = True                  # Also save last epoch model
SAVE_INTERVAL = 10                 # Save checkpoint every N epochs
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
    
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Discrete, skip_pretrained: bool = False):
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
        
        # Load mixed pretrained weights (skip if requested)
        if not skip_pretrained and not os.environ.get('SKIP_PRETRAINED_LOADING'):
            self._load_mixed_pretrained_weights()
        
        print(f"ImitationLearningPolicy initialized:")
        print(f"  - Observation space: {observation_space.spaces.keys()}")
        print(f"  - Action space: {action_space.n} discrete actions")
        print(f"  - Hidden size: {HIDDEN_SIZE}")
        print(f"  - RNN type: {RNN_TYPE}")
        print(f"  - Pretrained weights: {'skipped' if skip_pretrained or os.environ.get('SKIP_PRETRAINED_LOADING') else 'loaded'}")

        # ======================================================================
        # Plan-C: Custom LSTM to bypass Habitat's RNNStateEncoder assertion
        # ======================================================================
        # Extract the exact LSTM configuration from core_network's state_encoder
        original_lstm = self.core_network.state_encoder.rnn
        self.custom_lstm = nn.LSTM(
            input_size=original_lstm.input_size,
            hidden_size=original_lstm.hidden_size, 
            num_layers=original_lstm.num_layers,
            bias=True,  # Match Habitat default
            batch_first=False,  # Match Habitat default
            dropout=0.0,  # Match Habitat default
            bidirectional=False  # Match Habitat default
        )
        
        # Copy weights from original LSTM to ensure perfect alignment
        if not skip_pretrained and not os.environ.get('SKIP_PRETRAINED_LOADING'):
            self._copy_lstm_weights()
        
        # Passthrough module to extract pre-RNN features from PointNavResNetNet
        class _PassthroughRNN(nn.Module):
            def forward(self, x, hidden_states, masks, rnn_build_seq_info=None):
                # Just return features without RNN processing
                return x, hidden_states
        
        self._passthrough_rnn = _PassthroughRNN()
    
    def _copy_lstm_weights(self):
        """Copy LSTM weights from core_network.state_encoder.rnn to custom_lstm"""
        try:
            original_state_dict = self.core_network.state_encoder.rnn.state_dict()
            self.custom_lstm.load_state_dict(original_state_dict)
            print("✓ Successfully copied LSTM weights from core_network.state_encoder.rnn")
        except Exception as e:
            print(f"⚠ Warning: Could not copy LSTM weights: {e}")
            print("  Custom LSTM will use random initialization")
    
    def _sync_lstm_weights_back(self):
        """Sync trained custom_lstm weights back to core_network.state_encoder.rnn"""
        try:
            trained_state_dict = self.custom_lstm.state_dict()
            self.core_network.state_encoder.rnn.load_state_dict(trained_state_dict)
            print("✓ Successfully synced trained custom_lstm weights back to core_network.state_encoder.rnn")
        except Exception as e:
            print(f"⚠ Warning: Could not sync LSTM weights back: {e}")
    
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
        # print(f"rnn_hidden_states shape: {rnn_hidden_states.shape}")
        # print(f"masks shape: {masks.shape}")
        # print(f"batch_size:{rnn_hidden_states.size(0)}")
        # ======================================================================
        # Bypass RNNStateEncoder, use custom LSTM(fix rnn bug from habitat:assert rnn_build_seq_info is not None)
        # ======================================================================
        
        # 1) Get pre-RNN features by temporarily replacing state_encoder
        original_state_encoder = self.core_network.state_encoder
        self.core_network.state_encoder = self._passthrough_rnn
        pre_rnn_features, _, _ = self.core_network(
            observations, rnn_hidden_states, prev_actions, masks, rnn_build_seq_info=None
        )
        # Restore original state_encoder
        self.core_network.state_encoder = original_state_encoder
        
        # 2) Unpack Habitat-style hidden states: (batch, 2*num_layers, hidden) -> (h, c)
        # Note: rnn_hidden_states should be (batch, 2*num_layers, hidden) from get_initial_hidden_state
        # print(f"NUM_RECURRENT_LAYERS = {NUM_RECURRENT_LAYERS}")
        # print(f"HIDDEN_SIZE = {HIDDEN_SIZE}")
        # print(f"Expected 2*num_layers = {2 * NUM_RECURRENT_LAYERS}")
        batch_size = rnn_hidden_states.size(0)
        num_layers = rnn_hidden_states.size(1) // 2
        hidden_size = rnn_hidden_states.size(2)
        
        # Split into h and c, then permute to (num_layers, batch, hidden) for LSTM
        h_packed, c_packed = torch.chunk(rnn_hidden_states, 2, dim=1)  # Each: (batch, num_layers, hidden)
        h_0 = h_packed.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden)
        c_0 = c_packed.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden)
        
        # 3) Apply episode boundary resets based on masks
        if masks.dtype != torch.bool:
            mask_bool = masks > 0.5
        else:
            mask_bool = masks
        # print(f"mask_bool shape: {mask_bool.shape}")
        # Ensure masks are properly shaped for broadcasting with hidden states
        # masks: (batch, 1) or (batch,) -> (1, batch, 1) for LSTM hidden states (num_layers, batch, hidden)
        if mask_bool.dim() == 2:  # (batch, 1)
            mask_bool = mask_bool.squeeze(-1)  # (batch,)
        # print(f"mask_bool shape after squeeze: {mask_bool.shape}")
        # Reshape to (1, batch, 1) for broadcasting with (num_layers, batch, hidden)
        not_done = mask_bool.view(1, batch_size, 1).to(h_0.dtype)
        h_0 = h_0 * not_done
        c_0 = c_0 * not_done
        
        # --- One-time equivalence self-check against official LSTMStateEncoder.rnn ---
        # This prints once to verify custom_lstm matches the original Habitat LSTM on the same inputs
        if not hasattr(self, "_lstm_equiv_checked"):
            self._lstm_equiv_checked = True
            try:
                with torch.no_grad():
                    lstm_in_check = pre_rnn_features.unsqueeze(0)  # (1, B, input_size)
                    official_lstm = self.core_network.state_encoder.rnn
                    out_official, (h1o, c1o) = official_lstm(lstm_in_check, (h_0, c_0))
                    out_custom_chk, (h1c, c1c) = self.custom_lstm(lstm_in_check, (h_0, c_0))

                    feat_official = out_official.squeeze(0)
                    feat_custom_chk = out_custom_chk.squeeze(0)

                    def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
                        return float((a - b).abs().max().item())

                    # weight-by-weight equality
                    same_weights = True
                    off_sd = official_lstm.state_dict()
                    cus_sd = self.custom_lstm.state_dict()
                    if off_sd.keys() != cus_sd.keys():
                        same_weights = False
                    else:
                        for k in off_sd.keys():
                            if off_sd[k].shape != cus_sd[k].shape or not torch.allclose(off_sd[k], cus_sd[k]):
                                same_weights = False
                                break

                    print(
                        "[LSTM-CHECK] weight_equal=", same_weights,
                        " feat_diff=", f"{_maxdiff(feat_official, feat_custom_chk):.3e}",
                        " h_diff=", f"{_maxdiff(h1o, h1c):.3e}",
                        " c_diff=", f"{_maxdiff(c1o, c1c):.3e}",
                    )
            except Exception as _e:
                print("[LSTM-CHECK] skipped due to error:", str(_e))

        # 4) Single-step LSTM forward: (seq_len=1, batch, input_size)
        lstm_input = pre_rnn_features.unsqueeze(0)  # (1, batch, input_size)
        lstm_output, (h_1, c_1) = self.custom_lstm(lstm_input, (h_0, c_0))
        
        # Extract features from LSTM output
        features = lstm_output.squeeze(0)  # (batch, hidden_size)
        
        # 5) Re-pack hidden states back to Habitat format: (batch, 2*num_layers, hidden)
        h_1_packed = h_1.permute(1, 0, 2)  # (batch, num_layers, hidden)
        c_1_packed = c_1.permute(1, 0, 2)  # (batch, num_layers, hidden)
        rnn_hidden_states = torch.cat([h_1_packed, c_1_packed], dim=1)  # (batch, 2*num_layers, hidden)
        
        # 6) Action prediction
        action_logits = self.action_head(features)
        
        return action_logits, rnn_hidden_states
    
    @property
    def hidden_state_shape(self):
        """Hidden state shape for LSTM compatibility with Habitat (includes both h and c states)."""
        return (2 * NUM_RECURRENT_LAYERS, HIDDEN_SIZE)
    
    @property 
    def hidden_state_shape_lens(self):
        """Hidden state shape lengths for compatibility with Habitat."""
        return [HIDDEN_SIZE]
    
    def get_initial_hidden_state(self, batch_size: int) -> torch.Tensor:
        """
        Get initial hidden state for RNN following Habitat's LSTM convention.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Initial hidden state tensor (2*num_layers, batch_size, hidden_size) for LSTM
        """
        # For LSTM, Habitat's LSTMStateEncoder expects packed [h, c] states
        # Shape: (num_layers, batch_size, hidden_size) each, concatenated to (2*num_layers, batch_size, hidden_size)
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
        # print(f"get_initial_hidden_state: creating shape ({batch_size}, {2*NUM_RECURRENT_LAYERS}, {HIDDEN_SIZE})")
        result = torch.cat([h, c], dim=0)
        result = result.permute(1, 0, 2)
        return result
    
    def _load_mixed_pretrained_weights(self):
        """
        Load Habitat pretrained weights with proper action dimension handling.
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
                
        except Exception as e:
            print(f"   ✗ Failed to load Habitat weights: {e}")
            habitat_state_dict = {}
            return
        
        # Create unified weight dictionary
        print("2. Creating unified weight dictionary with action dimension handling...")
        unified_state_dict = {}
        
        # Get current model's state dict for reference
        current_state_dict = self.state_dict()
        
        # Track loading statistics
        loaded_from_habitat = 0
        action_weights_handled = 0
        not_loaded = 0
        
        for key in current_state_dict.keys():
            loaded = False
            current_shape = current_state_dict[key].shape
            
            # Handle action head mapping first
            if key == 'action_head.weight':
                habitat_key = 'actor_critic.action_distribution.linear.weight'
                if habitat_key in habitat_state_dict:
                    habitat_weight = habitat_state_dict[habitat_key]
                    if habitat_weight.shape == current_shape:
                        unified_state_dict[key] = habitat_weight
                        loaded_from_habitat += 1
                        action_weights_handled += 1
                        loaded = True
                        print(f"   Action head weight: {key} <- {habitat_key} {habitat_weight.shape}")
                    else:
                        print(f"   Action head weight shape mismatch: {key} {current_shape} vs {habitat_key} {habitat_weight.shape}")
            elif key == 'action_head.bias':
                habitat_key = 'actor_critic.action_distribution.linear.bias'
                if habitat_key in habitat_state_dict:
                    habitat_bias = habitat_state_dict[habitat_key]
                    if habitat_bias.shape == current_shape:
                        unified_state_dict[key] = habitat_bias
                        loaded_from_habitat += 1
                        action_weights_handled += 1
                        loaded = True
                        print(f"   Action head bias: {key} <- {habitat_key} {habitat_bias.shape}")
                    else:
                        print(f"   Action head bias shape mismatch: {key} {current_shape} vs {habitat_key} {habitat_bias.shape}")
            
            # Handle prev_action_embedding (no dimension adjustment needed - shapes match!)
            elif key == 'core_network.prev_action_embedding.weight':
                habitat_key = 'actor_critic.net.prev_action_embedding.weight'
                if habitat_key in habitat_state_dict:
                    habitat_weight = habitat_state_dict[habitat_key]
                    if habitat_weight.shape == current_shape:
                        # Direct copy - shapes should match perfectly
                        unified_state_dict[key] = habitat_weight
                        loaded_from_habitat += 1
                        action_weights_handled += 1
                        loaded = True
                        print(f"   Prev action embedding: {key} <- {habitat_key} {habitat_weight.shape}")
                    else:
                        print(f"   Prev action embedding shape mismatch: {key} {current_shape} vs {habitat_key} {habitat_weight.shape}")
            
            # Handle custom_lstm weights mapping
            elif key.startswith('custom_lstm.'):
                # Map custom_lstm.xxx -> actor_critic.net.state_encoder.rnn.xxx
                lstm_key = key.replace('custom_lstm.', '')
                habitat_key = f'actor_critic.net.state_encoder.rnn.{lstm_key}'
                if habitat_key in habitat_state_dict:
                    habitat_weight = habitat_state_dict[habitat_key]
                    if habitat_weight.shape == current_shape:
                        unified_state_dict[key] = habitat_weight
                        loaded_from_habitat += 1
                        loaded = True
                        print(f"   Custom LSTM weight: {key} <- {habitat_key} {habitat_weight.shape}")
                    else:
                        print(f"   Custom LSTM weight shape mismatch: {key} {current_shape} vs {habitat_key} {habitat_weight.shape}")
            
            # Handle other weights with standard mapping
            else:
                # Try direct key match first
                if key in habitat_state_dict:
                    habitat_weight = habitat_state_dict[key]
                    if habitat_weight.shape == current_shape:
                        unified_state_dict[key] = habitat_weight
                        loaded_from_habitat += 1
                        loaded = True
                    else:
                        print(f"   Shape mismatch: {key} {current_shape} vs {habitat_weight.shape}")
                else:
                    # Try with Habitat prefix mapping
                    # Our model: core_network.xxx -> Habitat: actor_critic.net.xxx
                    habitat_key = key.replace('core_network.', 'actor_critic.net.')
                    if habitat_key in habitat_state_dict:
                        habitat_weight = habitat_state_dict[habitat_key]
                        if habitat_weight.shape == current_shape:
                            unified_state_dict[key] = habitat_weight
                            loaded_from_habitat += 1
                            loaded = True
                        else:
                            print(f"   Shape mismatch: {key} {current_shape} vs {habitat_key} {habitat_weight.shape}")
            
            if not loaded:
                not_loaded += 1
        
        # Load the unified weights
        print("3. Loading unified weights into model...")
        try:
            missing_keys, unexpected_keys = self.load_state_dict(unified_state_dict, strict=False)
            print("   ✓ Weights loaded successfully with strict=False")
        except RuntimeError as e:
            print(f"   ✗ Weight loading failed: {e}")
        
        # Print loading report
        print("\n=== Weight Loading Report ===")
        print(f"Total model parameters: {len(current_state_dict)}")
        print(f"Loaded from Habitat: {loaded_from_habitat}")
        print(f"Action weights handled: {action_weights_handled}")
        print(f"Not loaded (using default init): {not_loaded}")
        
        if missing_keys:
            print(f"\nMissing keys ({len(missing_keys)}):")
            for key in missing_keys[:10]:  # Show first 10
                print(f"  - {key}")
            if len(missing_keys) > 10:
                print(f"  ... and {len(missing_keys) - 10} more")
        
        print("=== Habitat Weight Loading Complete ===\n")
    
 


# ===============================================================================
# REAL DATASET LOADER
# ===============================================================================

# -------------------------------------------------------------------------------
# NEW DATASET: Triple-Frame Random Window Dataset (length = 3)
# -------------------------------------------------------------------------------

class SocialNav2Dataset(Dataset):
    """
    Dataset for 3-frame windows [t-2, t-1, t] from episodes.
    Supports both training (random sampling) and validation (exhaustive coverage) modes.
    """
    def __init__(self, dataset_root: str, mode: str = 'train', val_episodes: List[str] = None):
        """
        Args:
            dataset_root: Root directory of the dataset
            mode: 'train' or 'val'
            val_episodes: List of episode names for validation (only used when mode='val')
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.mode = mode
        # Directories
        self.depth_dir = os.path.join(dataset_root, "jaw_depth_data")
        self.other_dir = os.path.join(dataset_root, "other_data")
        
        # Discover all episodes and load lengths
        all_episodes = self._discover_episodes()
        all_episode_lengths = self._load_episode_lengths(all_episodes)
        
        # Filter out episodes that are too short (< 3 frames)
        valid_episodes = []
        invalid_count = 0
        for ep in all_episodes:
            if all_episode_lengths[ep] >= 3:
                valid_episodes.append(ep)
            else:
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"⚠ Warning: Filtered out {invalid_count} episodes with < 3 frames")
        
        # Split episodes based on mode
        if mode == 'train':
            if val_episodes is None:
                val_episodes = []
            # Training: use all valid episodes except validation ones
            self.episode_files = [ep for ep in valid_episodes if ep not in val_episodes]
            self.episode_lengths = {ep: all_episode_lengths[ep] for ep in self.episode_files}
            self.N_ep = len(self.episode_files)
            print(f"SocialNav2Dataset (TRAIN): {len(self.episode_files)} episodes, samples/epoch={SAMPLES_PER_EPOCH}")
            
        elif mode == 'val':
            if val_episodes is None:
                raise ValueError("val_episodes must be provided when mode='val'")
            # Validation: use only specified validation episodes
            self.episode_files = [ep for ep in val_episodes if ep in valid_episodes]
            self.episode_lengths = {ep: all_episode_lengths[ep] for ep in self.episode_files}
            self._precompute_val_samples()
            print(f"SocialNav2Dataset (VAL): {len(self.episode_files)} episodes, {len(self.val_samples)} total windows")
            
            else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'val'")
        
        if len(self.episode_files) == 0:
            raise ValueError(f"No valid episodes found for mode '{mode}'!")

    # ---------------- internal helpers ----------------
    def _discover_episodes(self):
        other_pkls = glob.glob(os.path.join(self.other_dir, "*.pkl"))
        return [os.path.splitext(os.path.basename(p))[0] for p in other_pkls]

    def _load_episode_lengths(self, episode_files):
        meta = {}
        for ep in episode_files:
            with open(os.path.join(self.other_dir, f"{ep}.pkl"), 'rb') as f:
                data = pickle.load(f)
            meta[ep] = len(data['actions'])
        return meta
    
    def _precompute_val_samples(self):
        """Pre-compute all (episode, start_timestep) combinations for validation."""
        self.val_samples = []
        for ep in self.episode_files:
            ep_len = self.episode_lengths[ep]
            # Generate all possible 3-frame windows: [0,1,2], [1,2,3], ..., [ep_len-3, ep_len-2, ep_len-1]
            for start_t in range(ep_len - 2):
                self.val_samples.append((ep, start_t))

    # ---------------- required Dataset API ----------------
    def __len__(self):
        if self.mode == 'train':
            return SAMPLES_PER_EPOCH  # Original training logic
        else:  # validation
            return len(self.val_samples)  # Deterministic length

    def __getitem__(self, index):
        if self.mode == 'train':
            # ------------------------------------------------------------------
            # TRAINING: Original deterministic (index → episode, timestep) mapping
            # ------------------------------------------------------------------
            # Thus we iterate windows sequentially inside each episode while the
            # outer DataLoader (with shuffle/DistributedSampler) decides the final
            # global ordering.

            # 1) episode selection (cyclic, then shuffled by DataLoader if needed)
            # All episodes in self.episode_files are guaranteed to have ≥ 3 frames
            ep_idx = index % self.N_ep
            ep = self.episode_files[ep_idx]
            ep_len = self.episode_lengths[ep]
            
            # 2) current timestep t ∈ [2, ep_len-1]
            windows_per_episode = ep_len - 2  # each valid t gives one 3-frame window
            offset = (index // self.N_ep) % windows_per_episode
            t = offset + 2

            indices = [t - 2, t - 1, t]
            
        else:  # validation
            # ------------------------------------------------------------------
            # VALIDATION: Direct lookup from pre-computed samples
            # ------------------------------------------------------------------
            ep, start_t = self.val_samples[index]
            indices = [start_t, start_t + 1, start_t + 2]

        # 3. Load depth sequence & resize to (240,228)
        with open(os.path.join(self.depth_dir, f"{ep}.pkl"), 'rb') as f:
            depth_all = pickle.load(f)['agent_0_articulated_agent_jaw_depth']
        depth_seq = torch.from_numpy(depth_all[indices]).float()  # (3,H,W,1)
        depth_seq = F.interpolate(depth_seq.permute(0,3,1,2), size=(240,228), mode='bilinear', align_corners=False).permute(0,2,3,1)

        # 4. Load other data (actions, gps, masks)
        with open(os.path.join(self.other_dir, f"{ep}.pkl"), 'rb') as f:
            other = pickle.load(f)

        # GPS (optional)
        if 'agent_0_pointgoal_with_gps_compass' in other:
            gps_seq = torch.from_numpy(other['agent_0_pointgoal_with_gps_compass'][indices]).float()
        else:
            gps_seq = torch.zeros(3,2,dtype=torch.float32)

        # Actions & masks (need action label for each of the 3 steps)
        actions_seq = torch.from_numpy(other['actions'][indices]).long()
        masks_seq = torch.from_numpy(other['masks'][indices]).bool()

        observations = {
            'agent_0_articulated_agent_jaw_depth': depth_seq,
            'pointgoal_with_gps_compass': gps_seq,
        }
        
        return {
            'observations': observations,
            'actions': actions_seq,
            'masks': masks_seq,
        }


# Note: We no longer need a custom collate function since we're using fixed-length sequences
# PyTorch's default collate function will handle batching our fixed-size tensors


def create_dataloader(dataset_root: str, mode: str = 'train', batch_size: int = None, val_episodes: List[str] = None) -> DataLoader:
    """
    Create a DataLoader for training or validation.
    
    Args:
        dataset_root: Root directory of the dataset
        mode: 'train' or 'val'
        batch_size: Batch size (defaults to BATCH_SIZE for train, VAL_BATCH_SIZE for val)
        val_episodes: List of episode names for validation
        
    Returns:
        DataLoader instance
    """
    # Set default batch size based on mode
    if batch_size is None:
        batch_size = BATCH_SIZE if mode == 'train' else VAL_BATCH_SIZE
    
    # Create dataset
    dataset = SocialNav2Dataset(dataset_root, mode=mode, val_episodes=val_episodes)
    
    if mode == 'train':
        # Training DataLoader: support DDP, shuffling
    if USE_DDP and torch.cuda.is_available() and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
                num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
                num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True,  # Ensure consistent batch sizes
            )
    else:  # validation
        # Validation DataLoader: no DDP, no shuffling, deterministic
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False,  # Keep all validation samples
        )
    
    print(f"DataLoader ({mode.upper()}) created: batch_size={batch_size}, {len(dataset)} samples")
    return dataloader


# ===============================================================================
# VALIDATION EPISODES SELECTION
# ===============================================================================

def select_validation_episodes(dataset_root: str, count: int = VAL_EPISODES_COUNT) -> List[str]:
    """
    Select validation episodes deterministically (first 'count' episodes alphabetically).
    
    Args:
        dataset_root: Root directory of the dataset
        count: Number of episodes to select for validation
        
    Returns:
        List of episode names for validation
    """
    # Discover all episodes
    other_dir = os.path.join(dataset_root, "other_data")
    all_episodes = [os.path.splitext(os.path.basename(p))[0] 
                   for p in glob.glob(os.path.join(other_dir, "*.pkl"))]
    
    # Sort for deterministic selection
    all_episodes.sort()
    
    # Select first 'count' episodes
    val_episodes = all_episodes[:count]
    
    print(f"Selected {len(val_episodes)} episodes for validation out of {len(all_episodes)} total")
    return val_episodes


# ===============================================================================
# VALIDATION FUNCTION
# ===============================================================================

def validate(model, val_dataloader, device, is_ddp=False):
    """
    Run validation on the validation dataset.
    
    Args:
        model: The trained model
        val_dataloader: DataLoader for validation data
        device: Device to run validation on
        is_ddp: Whether using distributed training
        
    Returns:
        tuple: (avg_loss, accuracy) validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            # Extract batch data
            observations_batch = batch_data["observations"]  # [batch_size, seq_len, ...]
            actions_batch = batch_data["actions"]           # [batch_size, seq_len]
            masks_batch = batch_data["masks"]               # [batch_size, seq_len]
            
            batch_size, seq_len = actions_batch.shape
            
            # Initialize RNN hidden states for the batch
            rnn_hidden_states = (model.module if is_ddp else model).get_initial_hidden_state(batch_size).to(device)
            
            # Initialize prev_actions for the batch (start with zeros)
            prev_actions_batch = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Process each timestep in the sequence
            for t in range(seq_len):
                # Current timestep data
                current_observations = {
                    key: val[:, t].to(device) for key, val in observations_batch.items()
                }
                current_actions = actions_batch[:, t].to(device)
                current_masks = masks_batch[:, t].to(device)
                # Align with training: first step uses start token -> force mask=False at t==0
                if t == 0:
                    current_masks = torch.zeros_like(current_masks)
                
                # Forward pass
                action_logits, rnn_hidden_states = (model.module if is_ddp else model)(
                    current_observations,
                    rnn_hidden_states,
                    prev_actions_batch,
                    current_masks
                )
                
                # Calculate loss
                loss = F.cross_entropy(action_logits, current_actions)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted_actions = torch.argmax(action_logits, dim=1)
                correct_predictions += (predicted_actions == current_actions).sum().item()
                total_predictions += batch_size
                
                # Update prev_actions for next timestep (use ground truth for validation)
                prev_actions_batch = current_actions
            
            # Print progress occasionally
            if batch_idx % 50 == 0:
                print(f"Validation batch {batch_idx}/{len(val_dataloader)}")
    
    # Calculate final metrics
    avg_loss = total_loss / (len(val_dataloader) * seq_len)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    model.train()  # Switch back to training mode
    return avg_loss, accuracy


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
    # Initialize TensorBoard (only on main process)
    # -------------------------------------------------------------------------
    writer = None
    if is_main_process:
        # Create unique run name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"socialnav2_il_{timestamp}"
        tensorboard_path = os.path.join(TENSORBOARD_DIR, run_name)
        writer = SummaryWriter(tensorboard_path)
        print(f"TensorBoard logging to: {tensorboard_path}")
        print(f"Run: tensorboard --logdir={TENSORBOARD_DIR}")
        
        # Log hyperparameters
        sampling_info = "TripleFrameWindow(3)"
        writer.add_text("config/hyperparameters", f"""
        Learning Rate: {LEARNING_RATE}
        Batch Size: {BATCH_SIZE}
        Sequence Length: {SEQUENCE_LENGTH}
        Sampling Strategy: {sampling_info}
        Hidden Size: {HIDDEN_SIZE}
        RNN Type: {RNN_TYPE}
        Num RNN Layers: {NUM_RECURRENT_LAYERS}
        Max Grad Norm: {MAX_GRAD_NORM}
        Weight Decay: {WEIGHT_DECAY}
        Epochs: {NUM_EPOCHS}
        """)
    
    # -------------------------------------------------------------------------
    # Initialize best metric tracking
    # -------------------------------------------------------------------------
    best_metric = float("inf") if SAVE_BEST_BY in ["train_loss", "val_loss"] else float("-inf")
    best_epoch = 0
    
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
        print(f"Action space: {action_space.n} discrete actions")
    
    # Select validation episodes deterministically on every process
    # Deterministic function => no need to broadcast in DDP
    val_episodes = select_validation_episodes(DATASET_ROOT)
    
    # Create training and validation dataloaders
    train_dataloader = create_dataloader(DATASET_ROOT, mode='train', val_episodes=val_episodes)
    val_dataloader = create_dataloader(DATASET_ROOT, mode='val', val_episodes=val_episodes)
    
    if is_main_process:
        print(f"Training data ready: {len(train_dataloader)} batches per epoch")
        print(f"Validation data ready: {len(val_dataloader)} batches per validation")
    
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
        print(f"Sequences per epoch: {len(train_dataloader) * BATCH_SIZE}")
        print(f"Sequence length: {SEQUENCE_LENGTH} timesteps")
        print("-" * 80)
    
    if is_ddp and hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
        train_dataloader.sampler.set_epoch(0)
    
    model.train()
    global_iter = 0
    t_start = time.time()
    loss_window = deque(maxlen=50)
    acc_window = deque(maxlen=50)
    
    for epoch in range(NUM_EPOCHS):
        if is_ddp and hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            # Move batch data to device
            observations = {k: v.to(DEVICE) for k, v in batch_data['observations'].items()}
            actions = batch_data['actions'].to(DEVICE)  # (batch_size, seq_len)
            masks = batch_data['masks'].to(DEVICE)  # (batch_size, seq_len)
            batch_size, seq_len = actions.shape
            
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Initialize RNN hidden state for the whole batch
            rnn_hidden_states = (model.module.get_initial_hidden_state(batch_size) if is_ddp
                                 else model.get_initial_hidden_state(batch_size))
            # print(f"Initial hidden state shape: {rnn_hidden_states.shape}")
            # print(f"Training batch_size: {batch_size}")
            prev_actions_batch = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)

            # Time-step vectorized forward over the batch
            for t in range(seq_len):
                current_obs = {k: v[:, t] for k, v in observations.items()}  # (B, ...)
                current_actions = actions[:, t]  # (B,)
                current_masks = masks[:, t].bool().unsqueeze(-1)  # (B, 1)
                # Force first-step to use start token (align with eval): mask=False at t==0
                if t == 0:
                    current_masks = torch.zeros_like(current_masks)

            with torch.cuda.amp.autocast(enabled=DDP_FP16):
                    action_logits, rnn_hidden_states = (model.module if is_ddp else model)(
                        observations=current_obs,
                    rnn_hidden_states=rnn_hidden_states,
                        prev_actions=prev_actions_batch,
                        masks=current_masks,
                    )
                    step_loss = criterion(action_logits, current_actions)  # mean over B

                total_loss += step_loss
                predicted_actions = torch.argmax(action_logits, dim=1)
                correct_predictions += (predicted_actions == current_actions).sum().item()
                total_predictions += batch_size
                # Use model's own predictions as next step input (autoregressive) instead of teacher-forcing
                prev_actions_batch = predicted_actions.detach()
            
            # step_loss was averaged across batch; average further across time
            avg_loss = total_loss / seq_len
            
            # Backward pass and optimization
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
                remaining_iters = ((NUM_EPOCHS - (epoch + 1)) * len(train_dataloader)) + (len(train_dataloader) - (batch_idx + 1))
                eta_hours = (remaining_iters / max(1e-6, it_per_sec)) / 3600.0
                
                # Console logging
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | iter {global_iter} ({batch_idx+1}/{len(train_dataloader)}) | "
                    f"loss {batch_loss:.4f} (avg {np.mean(loss_window):.4f}) | "
                    f"acc {batch_accuracy:.4f} (avg {np.mean(acc_window):.4f}) | "
                    f"lr {cur_lr:.2e} | grad {float(grad_norm):.2f} | "
                    f"{it_per_sec:.2f} it/s | ETA {eta_hours:.2f}h | "
                    f"seq_len {seq_len} | steps {total_predictions}"
                )
                
                # TensorBoard logging
                if writer is not None:
                    writer.add_scalar("train/loss_batch", batch_loss, global_iter)
                    writer.add_scalar("train/accuracy_batch", batch_accuracy, global_iter)
                    writer.add_scalar("train/loss_avg", np.mean(loss_window), global_iter)
                    writer.add_scalar("train/accuracy_avg", np.mean(acc_window), global_iter)
                    writer.add_scalar("train/learning_rate", cur_lr, global_iter)
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_iter)
                    writer.add_scalar("train/iterations_per_second", it_per_sec, global_iter)
                    writer.add_scalar("train/sequences_per_batch", batch_size, global_iter)
                    writer.add_scalar("train/steps_per_batch", total_predictions, global_iter)
        
        # Compute epoch metrics
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        avg_epoch_accuracy = epoch_accuracy / max(1, num_batches)
        
        # Log epoch metrics
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.6f}")
            print(f"  Average Accuracy: {avg_epoch_accuracy:.4f}")
            print(f"  Total Iterations: {global_iter}")
            print(f"  Batches: {num_batches}")
            print(f"{'='*60}")
            
            # TensorBoard epoch logging
            if writer is not None:
                writer.add_scalar("epoch/loss", avg_epoch_loss, epoch + 1)
                writer.add_scalar("epoch/accuracy", avg_epoch_accuracy, epoch + 1)
                writer.add_scalar("epoch/total_iterations", global_iter, epoch + 1)
        
        # -------------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------------
        val_loss, val_accuracy = None, None
        if (epoch + 1) % VAL_INTERVAL == 0 and is_main_process:
            print(f"\n{'='*60}")
            print(f"Running validation after epoch {epoch + 1}...")
            print(f"{'='*60}")
            
            val_loss, val_accuracy = validate(model, val_dataloader, DEVICE, is_ddp)
            
            print(f"Validation Results:")
            print(f"  Validation Loss: {val_loss:.6f}")
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
            print(f"{'='*60}")
            
            # TensorBoard validation logging
            if writer is not None:
                writer.add_scalar("validation/loss", val_loss, epoch + 1)
                writer.add_scalar("validation/accuracy", val_accuracy, epoch + 1)
        
        # -------------------------------------------------------------------------
        # Model Saving Strategy
        # -------------------------------------------------------------------------
        if is_main_process:
            # Determine current metric based on SAVE_BEST_BY setting
            if SAVE_BEST_BY == "train_loss":
                current_metric = avg_epoch_loss
            elif SAVE_BEST_BY == "train_accuracy":
                current_metric = avg_epoch_accuracy
            elif SAVE_BEST_BY == "val_loss":
                current_metric = val_loss if val_loss is not None else float("inf")
            elif SAVE_BEST_BY == "val_accuracy":
                current_metric = val_accuracy if val_accuracy is not None else float("-inf")
            else:
                raise ValueError(f"Unknown SAVE_BEST_BY value: {SAVE_BEST_BY}")
            
            # Determine if current metric is better
            is_better = (current_metric < best_metric) if SAVE_BEST_BY in ["train_loss", "val_loss"] else (current_metric > best_metric)
            
            # Save best model (only if we have the required metric)
            can_save_best = True
            if SAVE_BEST_BY in ["val_loss", "val_accuracy"] and val_loss is None:
                can_save_best = False
                
            if SAVE_BEST and can_save_best and is_better:
                best_metric = current_metric
                best_epoch = epoch + 1
                
                # Sync custom_lstm weights back before saving best model
                model_for_sync = model.module if is_ddp else model
                model_for_sync._sync_lstm_weights_back()
                
                best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
                save_checkpoint(
                    model, optimizer, epoch + 1, avg_epoch_loss, avg_epoch_accuracy,
                    best_path, is_best=True, is_ddp=is_ddp
                )
                
                # Also save just the model weights for easy loading
                model_weights_path = os.path.join(CHECKPOINT_DIR, "best_model_weights.pth")
                model_state = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save(model_state, model_weights_path)
                
                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar("best/epoch", best_epoch, epoch + 1)
                    writer.add_scalar("best/loss", avg_epoch_loss, epoch + 1)
                    writer.add_scalar("best/accuracy", avg_epoch_accuracy, epoch + 1)
            
            # Save checkpoint at intervals
            if (epoch + 1) % SAVE_INTERVAL == 0:
                interval_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1:03d}.pth")
                save_checkpoint(
                    model, optimizer, epoch + 1, avg_epoch_loss, avg_epoch_accuracy,
                    interval_path, is_best=False, is_ddp=is_ddp
                )
            
            # Save last epoch
            if SAVE_LAST:
                # Ensure custom LSTM weights are synced back before saving "last"
                model_for_sync = model.module if is_ddp else model
                model_for_sync._sync_lstm_weights_back()
                
                last_path = os.path.join(CHECKPOINT_DIR, "last_model.pth")
                save_checkpoint(
                    model, optimizer, epoch + 1, avg_epoch_loss, avg_epoch_accuracy,
                    last_path, is_best=False, is_ddp=is_ddp
                )
                
                # Also save the original format for compatibility
                model_state = model.module.state_dict() if is_ddp else model.state_dict()
                torch.save(model_state, MODEL_SAVE_PATH)
    
    # -------------------------------------------------------------------------
    # 4. Save the final model (only main process)
    # -------------------------------------------------------------------------
    if is_main_process:
        # Sync custom_lstm weights back to core_network.state_encoder.rnn
        model_for_sync = model.module if is_ddp else model
        model_for_sync._sync_lstm_weights_back()
        
        # Final model save
        final_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
        avg_final_loss = epoch_loss / max(1, num_batches)
        avg_final_accuracy = epoch_accuracy / max(1, num_batches)
        save_checkpoint(
            model, optimizer, NUM_EPOCHS, avg_final_loss, avg_final_accuracy,
            final_path, is_best=False, is_ddp=is_ddp
        )
        
        # Save final model weights in original format
        model_state = model.module if is_ddp else model
        torch.save(model_state.state_dict(), MODEL_SAVE_PATH)
        
        # Training summary
        total_time = time.time() - t_start
        print(f"\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total Training Time: {total_time/3600:.2f} hours ({total_time:.0f} seconds)")
        print(f"Total Iterations: {global_iter}")
        print(f"Average Time per Iteration: {total_time/global_iter:.3f} seconds")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best {SAVE_BEST_BY}: {best_metric:.6f}")
        print(f"Final Model saved to: {MODEL_SAVE_PATH}")
        print(f"Best Model saved to: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
        
        # Log final summary to TensorBoard
        if writer is not None:
            writer.add_text("training/summary", f"""
            Training Completed Successfully!
            
            Total Time: {total_time/3600:.2f} hours
            Total Iterations: {global_iter}
            Best Epoch: {best_epoch}
            Best {SAVE_BEST_BY}: {best_metric:.6f}
            
            Model files:
            - Final: {MODEL_SAVE_PATH}
            - Best: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}
            - Last: {os.path.join(CHECKPOINT_DIR, 'last_model.pth')}
            """)
            
            # Add hyperparameter summary
            writer.add_hparams(
                {
                    'learning_rate': LEARNING_RATE,
                    'batch_size': BATCH_SIZE,
                    'hidden_size': HIDDEN_SIZE,
                    'rnn_layers': NUM_RECURRENT_LAYERS,
                    'max_grad_norm': MAX_GRAD_NORM,
                    'weight_decay': WEIGHT_DECAY,
                },
                {
                    'final_loss': avg_final_loss,
                    'final_accuracy': avg_final_accuracy,
                    'best_metric': best_metric,
                    'best_metric_type': SAVE_BEST_BY,
                    'best_epoch': best_epoch,
                    'total_iterations': global_iter,
                }
            )
            
            writer.close()
            print(f"TensorBoard logs saved to: {tensorboard_path}")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


def save_checkpoint(
    model, 
    optimizer, 
    epoch, 
    loss, 
    accuracy, 
    filepath, 
    is_best=False,
    is_ddp=False
):
    """
    Save a complete training checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch number
        loss: Current loss value
        accuracy: Current accuracy value
        filepath: Path to save the checkpoint
        is_best: Whether this is the best checkpoint
        is_ddp: Whether using DistributedDataParallel
    """
    # Get the actual model (unwrap DDP if necessary)
    model_state = model.module.state_dict() if is_ddp else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'training_config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'rnn_type': RNN_TYPE,
            'num_recurrent_layers': NUM_RECURRENT_LAYERS,
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    checkpoint_type = "BEST" if is_best else "CHECKPOINT"
    print(f"[{checkpoint_type}] Saved to {filepath} (epoch={epoch}, loss={loss:.4f}, acc={accuracy:.4f})")


def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary with checkpoint information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Accuracy: {checkpoint['accuracy']:.4f}")
    
    return checkpoint


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