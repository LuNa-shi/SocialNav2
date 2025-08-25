#!/usr/bin/env python3
import os
import sys
import argparse
import torch

"""
Convert IL weights (our ImitationLearningPolicy) to a Habitat-Baselines RL checkpoint format
so it can be used by habitat_baselines.run evaluator (expects 'state_dict' and 'extra_state').

Usage:
  # Single-agent RL-style
  python tools/convert_il_to_rl_ckpt.py \
    --il-weights /root/zjm/SocialNav2/checkpoints0820/last_model.pth \
    --out /root/zjm/SocialNav2/checkpoints0820/last_model_rl.pth


  # Multi-agent wrapper (keyed by int 0) with top-level extra_state
  python tools/convert_il_to_rl_ckpt.py \
    --il-weights /root/zjm/SocialNav2/checkpoints0820/last_model.pth \
    --out /root/zjm/SocialNav2/checkpoints0820/last_model_rl.pth \
    --multi-agent

It supports two IL formats:
  1) Pure state_dict file (e.g., best_model_weights.pth) -> directly map keys
  2) Our IL checkpoint with 'model_state_dict' -> extract then map

We also add a minimal 'extra_state' with a 'step' field to satisfy evaluator needs.
"""


def load_il_state_dict(il_path: str) -> dict:
    ckpt = torch.load(il_path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            return ckpt['model_state_dict']
        # Sometimes users pass a pure state_dict
        # Verify it looks like a state_dict (tensor values)
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError(f"Unrecognized IL weight format at: {il_path}")


def map_il_to_rl_keys(il_state: dict) -> dict:
    rl_state = {}
    for k, v in il_state.items():
        new_k = k
        # Map core network prefix to RL actor_critic.net
        if new_k.startswith('core_network.'):
            new_k = new_k.replace('core_network.', 'actor_critic.net.')
        # Map IL action head to RL action_distribution
        if new_k == 'action_head.weight':
            new_k = 'actor_critic.action_distribution.linear.weight'
        elif new_k == 'action_head.bias':
            new_k = 'actor_critic.action_distribution.linear.bias'
        rl_state[new_k] = v
    return rl_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--il-weights', type=str, required=True, help='Path to IL weights (.pth)')
    parser.add_argument('--out', type=str, required=True, help='Output RL checkpoint path (.pth)')
    parser.add_argument('--multi-agent', action='store_true', help='Wrap as multi-agent dict keyed by int 0')
    args = parser.parse_args()

    il_path = os.path.abspath(args.il_weights)
    out_path = os.path.abspath(args.out)

    if not os.path.exists(il_path):
        print(f"[ERROR] IL weight file not found: {il_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading IL weights from: {il_path}")
    il_state = load_il_state_dict(il_path)
    print(f"  Loaded {len(il_state)} parameters")

    print("Mapping IL keys to RL keys...")
    rl_state = map_il_to_rl_keys(il_state)
    print(f"  Mapped to {len(rl_state)} parameters")

    # Build RL checkpoint dict with minimal fields expected by evaluator
    base_ckpt = {
        'state_dict': rl_state,
        'extra_state': {
            'step': 0,           # required by evaluator
        },
    }

    if args.multi_agent:
        # Provide top-level extra_state for ppo_trainer._eval_checkpoint
        # and per-agent entry for MultiAgentAccessMgr.load_state_dict
        wrapped = {
            'extra_state': {'step': 0},
            0: base_ckpt,  # agent 0 ckpt containing its own state_dict/extra_state
        }
        torch.save(wrapped, out_path)
        print(f"[OK] Saved Multi-Agent RL-style checkpoint to: {out_path}")
        print("Set in eval yaml:\n"
              f"  habitat_baselines.eval.eval_ckpt_path_dir: {out_path}\n"
              "And set:\n  habitat_baselines.rl.ddppo.reset_critic: True (recommended)")
    else:
        torch.save(base_ckpt, out_path)
        print(f"[OK] Saved RL-style checkpoint to: {out_path}")
        print("Set in eval yaml:\n"
              f"  habitat_baselines.eval.eval_ckpt_path_dir: {out_path}\n"
              "And set:\n  habitat_baselines.rl.ddppo.reset_critic: True (recommended)")


if __name__ == '__main__':
    main() 