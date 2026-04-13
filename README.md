# SocialNav2 🧍🤖

A project for **social navigation** in embodied environments.

This repository is built on top of previous work, and we further adapted, trained, and evaluated the system for our own experiments.

## Overview ✨

SocialNav2 focuses on **PointGoal social navigation** in 3D indoor environments, where an agent must:
- 🎯 reach the target position,
- 🚶 avoid dynamic human obstacles,
- 🧠 navigate efficiently,
- 🤝 maintain social compliance.

Our current approach uses a **two-stage training paradigm**:
1. **Imitation Learning (IL)** for policy pre-training
2. **Reinforcement Learning (RL)** for fine-tuning

## Current Progress 🚀

Our recent progress is based on a two-stage pipeline:

- 📦 **Expert Data Collection**  
  We first collect high-quality expert trajectories in Habitat and filter them to keep only successful, high-reward episodes.

- 🧠 **Imitation Learning Pre-training**  
  We use **Behavioral Cloning (BC)** to train the policy from expert demonstrations, giving the agent a strong initial navigation behavior.

- 🔥 **Reinforcement Learning Fine-tuning**  
  We then fine-tune the pretrained model with **PPO**, allowing the agent to further improve success rate, path efficiency, and collision avoidance.

- 👀 **Social Awareness Enhancement**  
  During RL fine-tuning, we incorporate auxiliary socially-aware objectives such as:
  - human count estimation,
  - current human position tracking,
  - future trajectory forecasting.

## Model Design 🏗️

Our policy network includes:
- 🖼️ a visual encoder for egocentric observations,
- 📍 a state encoder for pose and goal information,
- 🧠 a GRU-based temporal memory module,
- 🎯 a policy head and a value head for action prediction and RL optimization.

## Results 📊

Our current experiments show that the **IL + RL** pipeline performs better than both:
- training with **RL only from scratch**, and
- using **IL only** without RL fine-tuning.

### Official evaluation results

| Method | SR ↑ | SPL ↑ | PSC ↑ | H-Coll ↓ | Total ↑ |
|------|------:|------:|------:|------:|------:|
| Baseline (RL-only) | 0.5400 | 0.4997 | 0.8630 | 0.3920 | 0.6248 |
| Ours (IL-only) | 0.6280 | 0.5761 | 0.8609 | 0.3540 | 0.6823 |
| Ours (IL+RL) | **0.6480** | **0.6010** | 0.8607 | **0.3420** | **0.6977** |

These results show that:
- ✅ IL provides a strong initialization,
- ✅ RL fine-tuning improves efficiency and robustness,
- ✅ the combined method achieves the best overall performance.

## Environment 🗺️

- Habitat simulator
- HM3D environments
- Bullet physics engine

## Notes 📌

This repository is based on previous social navigation work.  
We built on top of the existing project and further adapted it for our own training, experiments, and evaluation.

## Status 🌱

Current focus:
- improving social compliance,
- reducing human collision rate,
- building stronger embodied navigation policies,
- and exploring better training strategies for dynamic environments.
