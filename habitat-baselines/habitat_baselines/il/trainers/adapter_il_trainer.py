#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import json
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset, DataLoader
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from omegaconf import OmegaConf

# Import your existing policy. 
# Make sure the file containing Adapt3RPolicy is in the Python path.
from habitat_baselines.rl.ddppo.policy.adapt3r_policy import Adapt3RPolicy

# A high-level overview of the implementation steps:
# 1. Create a PyTorch Dataset to load the expert observation-action pairs.
#    - This assumes you have expert data stored, for example, in JSON files.
# 2. Implement a custom collate function to handle the dictionary-based
#    observations when creating batches.
# 3. Create a new IL Trainer class that:
#    a. Initializes the Adapt3RPolicy.
#    b. Initializes the Dataset and DataLoader.
#    c. Implements the training loop to perform supervised learning.
# 4. Register the new trainer with the baseline_registry so it can be
#    instantiated from a config file.

# ############################################################################ #
# 1. Expert Dataset
# ############################################################################ #

class SocialNavExpertDataset(Dataset):
    """
    A PyTorch Dataset for loading expert demonstrations for imitation learning.
    This class assumes your data is stored as a list of episodes, where each
    episode is a JSON file containing a sequence of (observation, action) steps.
    """

    def __init__(self, data_path: str, max_len: int = -1):
        """
        Args:
            data_path: Path to the directory containing the expert demonstrations.
            max_len: The maximum number of samples to load. -1 for all.
        """
        self.data_path = data_path
        self.demonstrations = []

        # Find all demonstration files (e.g., episode_*.json)
        demo_files = [f for f in os.listdir(data_path) if f.endswith(".json")]

        logger.info(f"Found {len(demo_files)} demonstration files in {data_path}.")

        for filename in demo_files:
            file_path = os.path.join(self.data_path, filename)
            with open(file_path, "r") as f:
                episode_data = json.load(f)
                # Each step in the episode is a training sample
                for step in episode_data["trajectory"]:
                    self.demonstrations.append({
                        "observation": step["observation"],
                        "action": step["action"],
                    })

        if max_len > 0:
            self.demonstrations = self.demonstrations[:max_len]
        
        logger.info(f"Loaded {len(self.demonstrations)} state-action pairs.")

    def __len__(self):
        return len(self.demonstrations)

    def __getitem__(self, idx):
        demo = self.demonstrations[idx]
        
        # Process observations: Convert lists to tensors.
        # Your Adapt3RNet expects specific keys and tensor formats.
        # This part MUST be adapted to match your exact observation structure.
        obs_dict = {}
        for key, value in demo["observation"].items():
            # Example: convert to torch.tensor.
            # You may need specific dtype and shape adjustments.
            obs_dict[key] = torch.tensor(value, dtype=torch.float32)
        
        # The action is the target label for our classification task.
        expert_action = torch.tensor(demo["action"], dtype=torch.long)

        return obs_dict, expert_action

# ############################################################################ #
# 2. Custom Collate Function
# ############################################################################ #

def custom_collate_fn(batch: List[tuple]) -> tuple:
    """
    Custom collate function to batch dictionary-based observations.
    
    Args:
        batch: A list of (observation_dict, action) tuples.

    Returns:
        A tuple containing:
        - A single dictionary of batched observations.
        - A tensor of batched actions.
    """
    batched_obs = {}
    actions = []

    # Get all observation keys from the first sample
    obs_keys = batch[0][0].keys()

    for key in obs_keys:
        # Stack tensors for each key
        batched_obs[key] = torch.stack([sample[0][key] for sample in batch])

    # Stack the action tensors
    actions = torch.stack([sample[1] for sample in batch])

    return batched_obs, actions


# ############################################################################ #
# 3. Imitation Learning Trainer
# ############################################################################ #

@baseline_registry.register_trainer(name="adapt3r_il_trainer")
class Adapt3RILTrainer(BaseILTrainer):
    """
    Trainer for Imitation Learning with the Adapt3R Policy.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.optimizer = None
        self.device = (
            torch.device("cuda", self.config.habitat_baselines.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if config is not None:
            logger.info(f"config: {OmegaConf.to_yaml(config)}")

    def train(self) -> None:
        """Main training routine."""
        config = self.config

        # --- 1. Initialize Policy and Optimizer ---
        # The Adapt3RPolicy.from_config method is the standard way to build it.
        # It needs observation_space and action_space. We can create dummy spaces
        # as they are mainly used by the policy constructor to get shapes and keys.
        # In a full Habitat setup, this would come from the environment.
        
        # TODO: Define dummy spaces that match your expert data structure.
        # This is a critical step. The keys and shapes must be correct.
        dummy_obs_space = ... # You must define this based on your data.
        dummy_action_space = ... # e.g., spaces.Discrete(num_actions)

        self.policy = Adapt3RPolicy.from_config(
            config=config,
            observation_space=dummy_obs_space,
            action_space=dummy_action_space,
        )
        self.policy.to(self.device)
        self.policy.train()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(config.habitat_baselines.il.lr),
        )
        
        # --- 2. Initialize Dataset and DataLoader ---
        il_cfg = config.habitat_baselines.il
        dataset = SocialNavExpertDataset(data_path=il_cfg.data_path)
        
        data_loader = DataLoader(
            dataset,
            batch_size=il_cfg.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=4, # Increase for faster data loading
        )
        
        num_epochs = il_cfg.epochs

        with TensorboardWriter(
            config.habitat_baselines.tensorboard_dir, flush_secs=self.flush_secs
        ) as writer:
            
            step_id = 0
            start_time = time.time()

            for epoch in range(num_epochs):
                logger.info(f"--- Starting Epoch {epoch + 1}/{num_epochs} ---")
                
                for batch_idx, batch in enumerate(data_loader):
                    observations, expert_actions = batch

                    # Move data to the correct device
                    observations = {k: v.to(self.device) for k, v in observations.items()}
                    expert_actions = expert_actions.to(self.device)
                    
                    # --- 3. Forward Pass ---
                    # For stateless SFT/IL, we can provide zero-d out recurrent states.
                    # The policy network expects specific inputs.
                    batch_size = expert_actions.shape[0]
                    rnn_hidden_states = torch.zeros(
                        self.policy.net.num_recurrent_layers,
                        batch_size,
                        self.policy.net.recurrent_hidden_size,
                        device=self.device,
                    )
                    # For SFT, prev_actions and masks are typically not used or are reset.
                    prev_actions = torch.zeros(
                        batch_size, 1, device=self.device, dtype=torch.long
                    )
                    masks = torch.ones(batch_size, 1, device=self.device)

                    # Get the policy's output features
                    features, _ = self.policy.net(
                        observations, rnn_hidden_states, prev_actions, masks
                    )

                    # Get the action distribution (e.g., a Categorical distribution)
                    action_distribution = self.policy.action_distribution(features)

                    # --- 4. Calculate Loss ---
                    # The loss is the negative log probability of the expert action.
                    log_probs = action_distribution.log_probs(expert_actions.squeeze(-1))
                    loss = -log_probs.mean()

                    # --- 5. Backward Pass and Optimization ---
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # --- 6. Logging ---
                    if step_id % config.habitat_baselines.log_interval == 0:
                        accuracy = (action_distribution.mode() == expert_actions.squeeze(-1)).float().mean()
                        log_str = (
                            f"Epoch: {epoch + 1} | Batch: {batch_idx}/{len(data_loader)} | "
                            f"Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}"
                        )
                        logger.info(log_str)
                        writer.add_scalar("IL Loss", loss.item(), step_id)
                        writer.add_scalar("IL Accuracy", accuracy.item(), step_id)

                    step_id += 1

                # --- 7. Save Checkpoint ---
                if (epoch + 1) % il_cfg.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        config.habitat_baselines.checkpoint_folder,
                        f"il_epoch_{epoch+1}.pth"
                    )
                    self.save_checkpoint(self.policy.state_dict(), checkpoint_path)
            
            logger.info(f"Training finished in {(time.time() - start_time) / 60:.2f} minutes.")

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """
        Optional: Implement evaluation on a validation dataset.
        This would follow a similar structure to the training loop but without
        the backward pass and optimizer step.
        """
        logger.info(f"Evaluation not implemented for {self.__class__.__name__}.")
        pass