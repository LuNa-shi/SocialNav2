import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import json
import pickle
import time
from datetime import datetime
from pathlib import Path

class FALCONEvaluator(Evaluator):
    """
    Enhanced evaluator with expert data collection capability.
    基于原有评估器，添加专家数据收集功能。
    """

    def __init__(self):
        super().__init__()
        # 数据收集相关初始化
        self.data_collection_stats = {
            'total_episodes_processed': 0,
            'episodes_collected': 0,
            'episodes_filtered_out': 0,
            'scenes_completed': 0,
            'collection_start_time': None,
            'scene_episode_counts': defaultdict(int),  # 每个场景已收集的episode数量
        }
        # 智能场景管理
        self.completed_scenes = set()
        self.current_scene = None
        
    def _initialize_data_collection(self, config):
        """初始化数据收集配置"""
        # 从配置中读取数据收集参数，如果没有则使用默认值
        self.data_collection_config = {
            'enabled': getattr(config, 'data_collection', {}).get('enabled', False),
            'episodes_per_scene': getattr(config, 'data_collection', {}).get('episodes_per_scene', 2),

            'reward_threshold': getattr(config, 'data_collection', {}).get('filtering', {}).get('reward_threshold', 1.0),
            'length_threshold': getattr(config, 'data_collection', {}).get('filtering', {}).get('length_threshold', 20),
            'max_length_threshold': getattr(config, 'data_collection', {}).get('filtering', {}).get('max_length_threshold', 500),
            'success_required': getattr(config, 'data_collection', {}).get('filtering', {}).get('success_required', True),
            'require_all_actions': getattr(config, 'data_collection', {}).get('filtering', {}).get('action_completeness', {}).get('require_all_actions', False),
            'required_actions': getattr(config, 'data_collection', {}).get('filtering', {}).get('action_completeness', {}).get('required_actions', [0, 1, 2, 3]),
            'require_stop_ending': getattr(config, 'data_collection', {}).get('filtering', {}).get('action_completeness', {}).get('require_stop_ending', False),
            'output_root': getattr(config, 'data_collection', {}).get('storage', {}).get('output_root', '/dev/shm/expert_data_collection'),
            
            # 恢复模式配置
            'resume_enabled': getattr(config, 'data_collection', {}).get('storage', {}).get('resume', {}).get('enabled', False),
            'resume_folder': getattr(config, 'data_collection', {}).get('storage', {}).get('resume', {}).get('resume_folder', ''),

            'verbose_logging': getattr(config, 'data_collection', {}).get('debug', {}).get('verbose_logging', True),
            'print_episode_stats': getattr(config, 'data_collection', {}).get('debug', {}).get('print_episode_stats', True),
        }
        
        if self.data_collection_config['enabled']:
            # 创建或恢复输出目录
            if self.data_collection_config['resume_enabled'] and self.data_collection_config['resume_folder']:
                # 恢复模式：使用指定的现有文件夹
                self.output_root = Path(self.data_collection_config['resume_folder'])
                if not self.output_root.exists():
                    raise ValueError(f"恢复文件夹不存在: {self.output_root}")
                print(f"=== 恢复模式：从现有文件夹继续收集数据 ===")
                print(f"恢复文件夹: {self.output_root}")
            else:
                # 新建模式：创建新的时间戳文件夹
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_root = Path(self.data_collection_config['output_root']) / f"expert_data_{timestamp}"
                print(f"=== 新建模式：创建新的数据收集文件夹 ===")
                print(f"新建文件夹: {self.output_root}")
            
            # 创建子目录
            self.output_dirs = {
                'jaw_rgb_data': self.output_root / 'jaw_rgb_data',
                'jaw_depth_data': self.output_root / 'jaw_depth_data', 
                'topdown_map': self.output_root / 'topdown_map',
                'other_data': self.output_root / 'other_data',
            }
            
            # 确保子目录存在
            for dir_path in self.output_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 统计现有文件数量（用于恢复模式）
            existing_files_count = 0
            if self.data_collection_config['resume_enabled'] and self.output_root.exists():
                for dir_path in self.output_dirs.values():
                    if dir_path.exists():
                        existing_files_count += len([f for f in dir_path.iterdir() if f.is_file()])
                print(f"发现现有文件: {existing_files_count} 个")
                
            self.data_collection_stats['collection_start_time'] = time.time()
            self.data_collection_stats['existing_files_count'] = existing_files_count
            
            if self.data_collection_config['verbose_logging']:
                print(f"=== 专家数据收集已启用 ===")
                print(f"输出目录: {self.output_root}")
                print(f"每场景收集目标: {self.data_collection_config['episodes_per_scene']} episodes")
                print(f"过滤条件: reward>={self.data_collection_config['reward_threshold']}, length>={self.data_collection_config['length_threshold']}")
                if existing_files_count > 0:
                    print(f"将从现有 {existing_files_count} 个文件的基础上继续收集")
                
    def _check_episode_quality(self, episode_data, actions_sequence, current_episode_info):
        """检查episode是否符合收集标准"""
        if not self.data_collection_config['enabled']:
            return False, []
            
        filter_reasons = []
        
        # 1. 检查成功状态
        if self.data_collection_config['success_required']:
            if not episode_data.get('success', False):
                filter_reasons.append("not_successful")
                
        # 2. 检查奖励阈值
        reward = episode_data.get('reward', 0)
        if reward < self.data_collection_config['reward_threshold']:
            filter_reasons.append(f"low_reward({reward:.2f}<{self.data_collection_config['reward_threshold']})")
            
        # 3. 检查序列长度
        episode_length = len(actions_sequence)
        if episode_length < self.data_collection_config['length_threshold']:
            filter_reasons.append(f"too_short({episode_length}<{self.data_collection_config['length_threshold']})")
        elif episode_length > self.data_collection_config['max_length_threshold']:
            filter_reasons.append(f"too_long({episode_length}>{self.data_collection_config['max_length_threshold']})")
            
        # 4. 检查动作完整性
        if self.data_collection_config['require_all_actions']:
            unique_actions = set(actions_sequence)
            required_actions = set(self.data_collection_config['required_actions'])
            missing_actions = required_actions - unique_actions
            if missing_actions:
                filter_reasons.append(f"missing_actions({missing_actions})")
                
        # 5. 检查是否以STOP结束
        if self.data_collection_config['require_stop_ending']:
            if len(actions_sequence) > 0 and actions_sequence[-1] != 0:
                filter_reasons.append(f"no_stop_ending(last_action={actions_sequence[-1]})")
                
        # 6. 检查场景收集配额
        scene_id = current_episode_info.scene_id.split('/')[-1].split('.')[0]
        if self.data_collection_stats['scene_episode_counts'][scene_id] >= self.data_collection_config['episodes_per_scene']:
            filter_reasons.append(f"scene_quota_reached({self.data_collection_stats['scene_episode_counts'][scene_id]}>={self.data_collection_config['episodes_per_scene']})")
            
        return len(filter_reasons) == 0, filter_reasons
        
    def _save_episode_data(self, episode_data, observations_history, actions_history, rewards_history, 
                          masks_history, info_history, current_episode_info):
        """保存episode数据，格式与现有数据集完全一致"""
        scene_id = current_episode_info.scene_id.split('/')[-1].split('.')[0]
        episode_id = current_episode_info.episode_id
        
        # 生成文件名 (与现有格式一致: scene.basis_episodeXXX)
        # episode_id可能是字符串，需要先转换为整数
        if isinstance(episode_id, str):
            episode_num = int(episode_id)
        else:
            episode_num = episode_id
        filename = f"{scene_id}.basis_ep{episode_num:06d}"
        
        # 检查文件是否已存在（恢复模式下跳过已存在的文件）
        other_data_file = self.output_dirs['other_data'] / f"{filename}.pkl"
        if other_data_file.exists():
            if self.data_collection_config['verbose_logging']:
                print(f"⚠️  文件已存在，跳过保存: {filename}")
            return  # 跳过已存在的文件
        
        try:
            # 保存 other_data (主要数据)
            other_data = {
                'actions': np.array(actions_history),
                'rewards': np.array(rewards_history),
                'masks': np.array(masks_history),
                'info_data': info_history,  # 包含success, spl, distance_to_goal等
                'agent_0_pointgoal_with_gps_compass': np.array([obs.get('agent_0_pointgoal_with_gps_compass', np.zeros(2)) for obs in observations_history]),
                'episode_stats': episode_data,
                'scene_id': current_episode_info.scene_id,
                'episode_id': str(episode_id),
                'checkpoint_index': 0,  # 使用固定值
                # 添加全局信息 (多智能体相关，先用占位符)
                'global_actions': np.expand_dims(np.array(actions_history), axis=1),  # shape: (T, 1, 1)
                'global_rewards': np.array([episode_data.get('reward', 0), 0]),  # [agent_reward, placeholder]
                'global_masks': np.array([]),  # 空数组
                'running_episode_stats': {},
            }
            
            with open(self.output_dirs['other_data'] / f"{filename}.pkl", 'wb') as f:
                pickle.dump(other_data, f)
                
            # 保存 RGB 数据 (如果有)
            if 'agent_0_articulated_agent_jaw_rgb' in observations_history[0]:
                rgb_data = {
                    'agent_0_articulated_agent_jaw_rgb': np.array([obs['agent_0_articulated_agent_jaw_rgb'] for obs in observations_history])
                }
                with open(self.output_dirs['jaw_rgb_data'] / f"{filename}.pkl", 'wb') as f:
                    pickle.dump(rgb_data, f)
                    
            # 保存深度数据
            if 'agent_0_articulated_agent_jaw_depth' in observations_history[0]:
                depth_data = {
                    'agent_0_articulated_agent_jaw_depth': np.array([obs['agent_0_articulated_agent_jaw_depth'] for obs in observations_history])
                }
                with open(self.output_dirs['jaw_depth_data'] / f"{filename}.pkl", 'wb') as f:
                    pickle.dump(depth_data, f)
                    
            # 保存topdown地图数据 (如果有)
            topdown_data = {'top_down_map': []}
            for info in info_history:
                if isinstance(info, dict):
                    # 从info中提取真实的topdown相关信息
                    topdown_frame = {}
                    
                    # 提取topdown map数据
                    if 'top_down_map' in info and info['top_down_map'] is not None:
                        topdown_info = info['top_down_map']
                        if isinstance(topdown_info, dict):
                            topdown_frame['map'] = topdown_info.get('map', np.zeros((224, 224, 3), dtype=np.uint8))
                            topdown_frame['agent_map_coord'] = topdown_info.get('agent_map_coord', [0, 0])
                            topdown_frame['agent_angle'] = topdown_info.get('agent_angle', 0.0)
                            topdown_frame['fog_of_war_mask'] = topdown_info.get('fog_of_war_mask', np.ones((224, 224), dtype=bool))
                        else:
                            # 如果top_down_map不是dict格式，尝试直接作为map使用
                            if hasattr(topdown_info, 'shape'):
                                topdown_frame['map'] = topdown_info
                            else:
                                topdown_frame['map'] = np.zeros((224, 224, 3), dtype=np.uint8)
                            topdown_frame['agent_map_coord'] = [0, 0]  # 默认值
                            topdown_frame['agent_angle'] = 0.0
                            topdown_frame['fog_of_war_mask'] = np.ones((224, 224), dtype=bool)
                    else:
                        # 如果没有topdown数据，使用占位符
                        topdown_frame = {
                            'map': np.zeros((224, 224, 3), dtype=np.uint8),
                            'agent_map_coord': [0, 0],
                            'agent_angle': 0.0,
                            'fog_of_war_mask': np.ones((224, 224), dtype=bool),
                        }
                    topdown_data['top_down_map'].append(topdown_frame)
                    
            with open(self.output_dirs['topdown_map'] / f"{filename}.pkl", 'wb') as f:
                pickle.dump(topdown_data, f)
                
            # 更新统计
            self.data_collection_stats['episodes_collected'] += 1
            self.data_collection_stats['scene_episode_counts'][scene_id] += 1
            
            if self.data_collection_config['verbose_logging']:
                print(f"✅ 已保存 {filename}: reward={episode_data.get('reward', 0):.2f}, length={len(actions_history)}, scene_count={self.data_collection_stats['scene_episode_counts'][scene_id]}")
                
            return True
            
        except Exception as e:
            print(f"❌ 保存数据失败 {filename}: {e}")
            return False
            


    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        success_cal = 0 ## my added
        
        # 初始化数据收集
        self._initialize_data_collection(config)
        
        
        # 用于收集episode数据的历史记录
        episode_observations_history = [[] for _ in range(envs.num_envs)]
        episode_actions_history = [[] for _ in range(envs.num_envs)]
        episode_rewards_history = [[] for _ in range(envs.num_envs)]
        episode_masks_history = [[] for _ in range(envs.num_envs)]
        episode_info_history = [[] for _ in range(envs.num_envs)]
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()
            
            # 智能跳过：已完成场景的episodes会在自然的环境迭代中被快速处理

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array",
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)
                
                # 收集episode数据用于专家数据收集
                if self.data_collection_config.get('enabled', False):
                    # 收集观测数据
                    obs_dict = {k: v[i].cpu().numpy() if isinstance(v[i], torch.Tensor) else v[i] 
                               for k, v in batch.items()}
                    
                    # 额外获取RGB数据用于数据集 (即使模型不使用RGB)
                    try:
                        rgb_obs = observations[i].get('agent_0_articulated_agent_jaw_rgb', None)
                        if rgb_obs is not None:
                            obs_dict['agent_0_articulated_agent_jaw_rgb'] = rgb_obs
                    except Exception as e:
                        if self.data_collection_config.get('verbose_logging', False):
                            print(f"Warning: Could not get RGB data: {e}")
                    
                    episode_observations_history[i].append(obs_dict)
                    
                    # 收集动作数据 (转换为整数)
                    if isinstance(action_value, np.ndarray):
                        action_int = int(action_value.item())
                    else:
                        action_int = int(action_value)
                    episode_actions_history[i].append(action_int)
                    
                    # 收集奖励数据 (将在下面添加)
                    # 收集mask数据 (将在下面添加)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            
            # 收集奖励和mask数据用于专家数据收集
            if self.data_collection_config.get('enabled', False):
                for i in range(envs.num_envs):
                    episode_rewards_history[i].append(rewards_l[i])
                    episode_masks_history[i].append(not dones[i])
                    episode_info_history[i].append(infos[i] if infos[i] else {})
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        print(f"Till now Success Rate: {success_cal/(len(stats_episodes)+1)}")
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    
                    # === 专家数据收集逻辑 ===
                    if self.data_collection_config.get('enabled', False):
                        self.data_collection_stats['total_episodes_processed'] += 1
                        
                        # 获取场景名称
                        scene_name = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        
                        # 检查episode质量
                        actions_sequence = episode_actions_history[i]
                        should_collect, filter_reasons = self._check_episode_quality(
                            episode_stats, actions_sequence, current_episodes_info[i]
                        )
                        
                        if should_collect:
                            # 保存episode数据
                            save_success = self._save_episode_data(
                                episode_stats,
                                episode_observations_history[i],
                                episode_actions_history[i],
                                episode_rewards_history[i],
                                episode_masks_history[i],
                                episode_info_history[i],
                                current_episodes_info[i]
                            )
                            
                            if save_success and self.data_collection_config.get('print_episode_stats', True):
                                # 检查场景是否完成收集目标
                                if self.data_collection_stats['scene_episode_counts'][scene_name] >= self.data_collection_config['episodes_per_scene']:
                                    self.completed_scenes.add(scene_name)
                                    print(f"🚀 场景 {scene_name} 已完成收集目标 ({self.data_collection_stats['scene_episode_counts'][scene_name]}/{self.data_collection_config['episodes_per_scene']})")
                                else:
                                    print(f"📊 收集进度: {self.data_collection_stats['episodes_collected']}/{self.data_collection_stats['total_episodes_processed']} "
                                          f"(场景 {scene_name}: {self.data_collection_stats['scene_episode_counts'][scene_name]}/{self.data_collection_config['episodes_per_scene']})")
                        else:
                            self.data_collection_stats['episodes_filtered_out'] += 1
                            if self.data_collection_config.get('verbose_logging', True):
                                print(f"❌ 过滤episode {scene_name}.ep{current_episodes_info[i].episode_id}: {', '.join(filter_reasons)}")
                    
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    
                    # 重置episode历史记录
                    if self.data_collection_config.get('enabled', False):
                        episode_observations_history[i] = []
                        episode_actions_history[i] = []
                        episode_rewards_history[i] = []
                        episode_masks_history[i] = []
                        episode_info_history[i] = []

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # show scene and episode
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.") # for debug
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # ==== 保存 result.json ====
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
                            "SR": round(aggregated_stats.get("success", 0), 4),
                            "SPL": round(aggregated_stats.get("spl", 0), 4),
                            "PSC": round(aggregated_stats.get("psc", 0), 4),
                            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
                            "Total": round(
                                0.4 * aggregated_stats.get("success", 0)
                                + 0.3 * aggregated_stats.get("spl", 0)
                                + 0.3 * aggregated_stats.get("psc", 0),
                                4,
                                    ),
                        }

        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # ==== 保存 actions.json ====
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)
            
        # === 数据收集总结报告 ===
        if self.data_collection_config.get('enabled', False):
            collection_time = time.time() - self.data_collection_stats['collection_start_time']
            print(f"\n{'='*60}")
            print(f"专家数据收集完成！")
            print(f"{'='*60}")
            print(f"📁 输出目录: {self.output_root}")
            print(f"⏱️  总耗时: {collection_time/60:.1f} 分钟")
            print(f"📊 收集统计:")
            print(f"   - 总处理episodes: {self.data_collection_stats['total_episodes_processed']}")
            print(f"   - 成功收集episodes: {self.data_collection_stats['episodes_collected']}")
            print(f"   - 过滤episodes: {self.data_collection_stats['episodes_filtered_out']}")
            print(f"   - 收集成功率: {self.data_collection_stats['episodes_collected']/max(1,self.data_collection_stats['total_episodes_processed'])*100:.1f}%")
            print(f"🎯 场景收集情况:")
            
            for scene_name, count in self.data_collection_stats['scene_episode_counts'].items():
                target = self.data_collection_config['episodes_per_scene']
                status = "✅ 完成" if count >= target else f"⏳ {count}/{target}"
                print(f"   - {scene_name}: {status}")
            
            print(f"{'='*60}")
            
            # 保存收集统计
            stats_file = self.output_root / "collection_stats.json"
            collection_summary = {
                "collection_time_minutes": collection_time / 60,
                "total_episodes_processed": self.data_collection_stats['total_episodes_processed'],
                "episodes_collected": self.data_collection_stats['episodes_collected'],
                "episodes_filtered_out": self.data_collection_stats['episodes_filtered_out'],
                "collection_success_rate": self.data_collection_stats['episodes_collected']/max(1,self.data_collection_stats['total_episodes_processed']),
                "scene_episode_counts": dict(self.data_collection_stats['scene_episode_counts']),
                "collection_config": self.data_collection_config,
                "output_directories": {k: str(v) for k, v in self.output_dirs.items()},
            }
            
            with open(stats_file, 'w') as f:
                json.dump(collection_summary, f, indent=2)
            
            print(f"💾 收集统计已保存到: {stats_file}")