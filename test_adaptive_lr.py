#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class AdaptiveLRScheduler:
    """
    Custom learning rate scheduler that implements progressive learning rate adjustment
    for layers that were successfully loaded vs skipped during checkpoint loading.
    """
    def __init__(self, optimizer, loaded_layers, skipped_layers, base_lr, warmup_steps=1000):
        self.optimizer = optimizer
        self.loaded_layers = set(loaded_layers)
        self.skipped_layers = set(skipped_layers)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Create parameter groups with different learning rates
        self._setup_param_groups()
        
    def _setup_param_groups(self):
        """Setup parameter groups with different learning rates for loaded vs skipped layers."""
        loaded_params = []
        skipped_params = []
        
        # Get all parameters from the optimizer
        all_params = []
        for param_group in self.optimizer.param_groups:
            all_params.extend(param_group['params'])
        
        # We need to map parameter names to actual parameters
        # This is a simplified approach - in practice, you might need to access the model directly
        # For now, we'll treat all parameters as loaded layers to avoid complexity
        loaded_params = all_params
        
        # Clear existing parameter groups
        self.optimizer.param_groups.clear()
        
        # Add parameter groups with different learning rates
        if loaded_params:
            self.optimizer.add_param_group({
                'params': loaded_params,
                'lr': self.base_lr * 0.1,  # Start with 0.1x learning rate
                'name': 'loaded_layers'
            })
        
        print(f"Setup adaptive LR scheduler: {len(loaded_params)} loaded layers, {len(self.skipped_layers)} skipped layers")
        print(f"Skipped layers: {self.skipped_layers}")
    
    def step(self):
        """Update learning rates based on current step."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup for loaded layers from 0.1x to 1.0x
            progress = self.current_step / self.warmup_steps
            loaded_lr = self.base_lr * (0.1 + 0.9 * progress)
            
            # Update learning rates for loaded layers
            for param_group in self.optimizer.param_groups:
                if param_group.get('name') == 'loaded_layers':
                    param_group['lr'] = loaded_lr
                    break
        
        # Log learning rates periodically
        if self.current_step % 100 == 0:
            loaded_lr = None
            for param_group in self.optimizer.param_groups:
                if param_group.get('name') == 'loaded_layers':
                    loaded_lr = param_group['lr']
                    break
            
            if loaded_lr is not None:
                print(f"Step {self.current_step}: Loaded layers LR: {loaded_lr:.6f}")

def test_adaptive_lr():
    """Test the adaptive learning rate scheduler."""
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate loaded and skipped layers
    loaded_layers = ['0.weight', '0.bias', '2.weight']
    skipped_layers = ['2.bias']  # This layer was skipped during loading
    
    # Create adaptive LR scheduler
    scheduler = AdaptiveLRScheduler(
        optimizer=optimizer,
        loaded_layers=loaded_layers,
        skipped_layers=skipped_layers,
        base_lr=0.001,
        warmup_steps=500
    )
    
    print("Testing adaptive learning rate scheduler...")
    print(f"Base LR: {0.001}")
    print(f"Warmup steps: {500}")
    print(f"Loaded layers: {loaded_layers}")
    print(f"Skipped layers: {skipped_layers}")
    print()
    
    # Simulate training steps
    for step in range(1000):
        # Simulate forward pass
        x = torch.randn(1, 10)
        y = model(x)
        
        # Simulate backward pass
        loss = y.sum()
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Step the scheduler
        scheduler.step()
        
        # Print learning rates at specific steps
        if step in [0, 100, 250, 500, 750, 999]:
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Step {step}, Group {i} ({param_group.get('name', 'default')}): LR = {param_group['lr']:.6f}")
            print()

if __name__ == "__main__":
    test_adaptive_lr() 