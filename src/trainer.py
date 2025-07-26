"""MEO-LEO集群路由训练脚本 - 简化版本"""
import sys
import os
# 获取项目根目录路径（假设当前文件在 src 目录，上级目录就是根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from config import Config
from satellites import LEOSatellite, MEOSatellite
from rl_agent import RLAgent
from routing import route_request_optimized
from environment import analyze_network_topology, update_dynamic_meo_clusters
from data.data_loader import load_complete_environment, validate_dynamic_meo_data


class PacketInfo:
    """包信息类，简化包的状态管理"""
    def __init__(self, packet_id: int, src: int, dst: int, start_slot: int):
        self.id = packet_id
        self.src = src
        self.dst = dst
        self.current_pos = src
        self.start_slot = start_slot
        self.end_slot = None
        self.path = []
        self.hops_completed = 0
        self.status = 'new'  # new, routing, completed, failed
        self.routing_stats = {}
        self.last_update_slot = start_slot


class TrainingEnvironment:
    """简化的训练环境类"""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.setup_directories()

        # 训练统计
        self.episode_rewards = []
        self.episode_success_rates = []
        self.episode_avg_path_lengths = []
        self.episode_convergence_rates = []

        # 动态MEO相关统计
        self.meo_reassignment_episodes = []
        self.network_efficiency_evolution = []

    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('output.log_level', 'INFO'))
        log_file = self.config.get('output.log_file', 'logs/training.log')

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.get('output.model_save_path', 'models/'),
            self.config.get('output.results_path', 'results/'),
            'logs/'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def train(self):
        """执行训练"""
        self.logger.info("开始训练...")

        # 设置随机种子
        random_seed = self.config.get('simulation.random_seed', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 获取数据文件路径
        data_file = self.config.get('data.data_file', 'data/data.json')
        if not os.path.exists(data_file):
            self.logger.error(f"数据文件不存在: {data_file}")
            return

        # 验证数据兼容性
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            validate_dynamic_meo_data(data)
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return

        # 初始化智能体
        agent = RLAgent(
            learning_rate=self.config.get('rl_agent.learning_rate', 0.1),
            gamma=self.config.get('rl_agent.gamma', 0.9),
            epsilon=self.config.get('rl_agent.epsilon', 0.1),
            epsilon_decay=self.config.get('rl_agent.epsilon_decay', 0.995),
            epsilon_min=self.config.get('rl_agent.epsilon_min', 0.01)
        )

        # 训练参数
        num_episodes = self.config.get('training.num_episodes', 1000)
        save_interval = self.config.get('training.save_interval', 100)

        self.logger.info(f"开始训练 {num_episodes} 个episodes")

        for episode in range(num_episodes):
            episode_reward, success_rate, avg_path_length, convergence_rate = self.run_episode_simplified(
                agent, data_file, episode
            )

            # 记录统计信息
            self.episode_rewards.append(episode_reward)
            self.episode_success_rates.append(success_rate)
            self.episode_avg_path_lengths.append(avg_path_length)
            self.episode_convergence_rates.append(convergence_rate)

            # 衰减探索率
            agent.decay_epsilon()

            # 日志输出
            if episode % 50 == 0:
                agent_stats = agent.get_statistics()
                self.logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Success Rate={success_rate:.2%}, "
                    f"Avg Path Length={avg_path_length:.2f}, "
                    f"Convergence Rate={convergence_rate:.2%}, "
                    f"Epsilon={agent.epsilon:.3f}, "
                    f"States={agent_stats['total_states']}"
                )

            # 保存模型
            if episode % save_interval == 0 and episode > 0:
                self.save_model(agent, episode)

        # 训练完成
        self.logger.info("训练完成!")
        self.save_final_results(agent)

        if self.config.get('output.plot_results', True):
            self.plot_training_results()

    def run_episode_simplified(self, agent: RLAgent, data_file: str, episode: int) -> Tuple[float, float, float, float]:
        """
        简化的episode运行逻辑，专注于路由训练
        """
        # 加载数据
        with open(data_file, 'r') as f:
            data = json.load(f)

        train_queries = data.get('train_queries', [])
        num_time_slots = data.get('num_train_slots', self.config.get('network.num_time_slots', 50))

        if not train_queries:
            self.logger.warning("没有训练查询数据")
            return 0.0, 0.0, 0.0, 0.0

        # 统计变量
        total_reward = 0.0
        successful_routes = 0
        total_routes = 0
        total_path_length = 0
        routing_attempts = 0
        quick_convergences = 0  # 快速收敛的路由数量

        # 随机选择部分时间槽进行训练（提高效率）
        selected_slots = sorted(random.sample(
            range(min(num_time_slots, len(data.get('sat_positions_per_slot', [])))),
            min(10, num_time_slots)  # 每个episode最多处理10个时间槽
        ))

        meo_reassignments_this_episode = 0

        for slot_idx, current_slot in enumerate(selected_slots):
            # 加载当前时间槽的网络环境
            try:
                leos, meos, _ = load_complete_environment(current_slot, data_file)
            except Exception as e:
                self.logger.debug(f"加载时间槽 {current_slot} 失败: {e}")
                continue

            # 动态MEO重分配（每几个episode一次）
            if episode % 10 == 0 and slot_idx == 0:
                if self.config.get('network.enable_dynamic_meo_reassignment', False):
                    try:
                        original_assignments = {leo.id: leo.meo_id for leo in leos.values()}
                        new_assignments = update_dynamic_meo_clusters(leos, meos)
                        reassignments = sum(1 for leo_id in original_assignments
                                          if original_assignments[leo_id] != new_assignments.get(leo_id, -1))
                        if reassignments > 0:
                            meo_reassignments_this_episode += reassignments
                            self.logger.debug(f"Episode {episode}, Slot {current_slot}: {reassignments} MEO reassignments")
                    except Exception as e:
                        self.logger.debug(f"MEO重分配失败: {e}")

            # 获取当前时间槽的查询
            slot_queries = [q for q in train_queries if q['time'] == current_slot]

            # 如果当前槽没有查询，随机生成一些查询用于训练
            if not slot_queries and len(leos) >= 2:
                num_random_queries = random.randint(1, 3)
                leo_ids = list(leos.keys())
                for _ in range(num_random_queries):
                    src, dst = random.sample(leo_ids, 2)
                    slot_queries.append({'src': src, 'dst': dst, 'time': current_slot})

            # 处理查询
            for query in slot_queries:
                if total_routes >= 100:  # 限制每个episode的最大查询数
                    break

                src_id = query['src']
                dst_id = query['dst']

                # 验证节点存在性
                if src_id not in leos or dst_id not in leos:
                    continue

                total_routes += 1
                routing_attempts += 1

                # 执行路由
                start_time = datetime.now()
                try:
                    path, routing_stats = route_request_optimized(
                        src_id, dst_id, leos, meos, agent, k_paths=2, max_hops=20
                    )
                except Exception as e:
                    self.logger.debug(f"路由异常: {e}")
                    path, routing_stats = [], {'success': False, 'error': str(e)}

                routing_time = (datetime.now() - start_time).total_seconds()

                # 计算奖励
                reward = self.calculate_routing_reward(
                    path, routing_stats, src_id, dst_id, leos, meos, routing_time
                )
                total_reward += reward

                # 统计成功路由
                if routing_stats.get('success', False) and len(path) > 1:
                    successful_routes += 1
                    total_path_length += len(path) - 1

                    # 检查是否快速收敛（路由时间短且路径合理）
                    if routing_time < 0.1 and len(path) <= 10:
                        quick_convergences += 1

                # 记录网络效率（每几个查询一次）
                if routing_attempts % 20 == 0:
                    try:
                        topology_analysis = analyze_network_topology(leos, meos)
                        network_efficiency = topology_analysis.get('network_efficiency', 0.0)
                        self.network_efficiency_evolution.append({
                            'episode': episode,
                            'slot': current_slot,
                            'efficiency': network_efficiency
                        })
                    except Exception as e:
                        self.logger.debug(f"网络分析失败: {e}")

        # 记录MEO重分配统计
        if meo_reassignments_this_episode > 0:
            self.meo_reassignment_episodes.append(meo_reassignments_this_episode)

        # 计算统计指标
        success_rate = successful_routes / total_routes if total_routes > 0 else 0.0
        avg_path_length = total_path_length / successful_routes if successful_routes > 0 else 0.0
        convergence_rate = quick_convergences / routing_attempts if routing_attempts > 0 else 0.0

        return total_reward, success_rate, avg_path_length, convergence_rate

    def calculate_routing_reward(self, path: List[int], routing_stats: Dict,
                               src_id: int, dst_id: int,
                               leos: Dict[int, LEOSatellite],
                               meos: Dict[int, MEOSatellite],
                               routing_time: float) -> float:
        """
        改进的奖励计算函数
        """
        base_reward = 0.0

        if not routing_stats.get('success', False) or len(path) < 2:
            # 路由失败
            failure_penalty = self.config.get('environment.reward_failure', -5.0)
            return failure_penalty

        # 基础成功奖励
        success_reward = self.config.get('environment.reward_success', 10.0)
        base_reward += success_reward

        # 路径长度惩罚/奖励
        path_length = len(path) - 1
        if path_length > 0:
            hop_penalty = self.config.get('environment.reward_hop', -0.1)
            base_reward += hop_penalty * path_length

            # 路径效率奖励（相对于最短理论距离）
            if src_id in leos and dst_id in leos:
                src_leo = leos[src_id]
                dst_leo = leos[dst_id]
                theoretical_distance = abs(src_leo.latitude - dst_leo.latitude) + abs(src_leo.longitude - dst_leo.longitude)
                if theoretical_distance > 0:
                    efficiency = theoretical_distance / path_length
                    base_reward += min(efficiency * 2.0, 5.0)  # 效率奖励，最大5分

        # 负载均衡奖励
        if len(path) > 1:
            path_loads = []
            for leo_id in path:
                if leo_id in leos:
                    path_loads.append(leos[leo_id].load)

            if path_loads:
                avg_load = sum(path_loads) / len(path_loads)
                max_load = self.config.get('network.max_load_per_satellite', 10)

                if avg_load < max_load * 0.5:
                    load_balance_reward = self.config.get('environment.reward_load_balance', 1.0)
                    base_reward += load_balance_reward

        # 路由策略奖励
        routing_strategy = routing_stats.get('routing_strategy', 'unknown')
        if routing_strategy == 'inter_cluster':
            inter_cluster_reward = self.config.get('environment.reward_inter_cluster_success', 2.0)
            base_reward += inter_cluster_reward
        elif routing_strategy == 'intra_cluster':
            base_reward += 0.5  # 集群内路由小奖励

        # 路由时间奖励（快速路由给予奖励）
        if routing_time < 0.05:  # 50ms以内
            base_reward += 1.0
        elif routing_time > 0.5:  # 超过500ms惩罚
            base_reward -= 1.0

        return base_reward

    def save_model(self, agent: RLAgent, episode: int):
        """保存模型"""
        model_path = self.config.get('output.model_save_path', 'models/')
        model_file = os.path.join(model_path, f'rl_agent_episode_{episode}.json')

        model_data = agent.save_model()
        model_data.update({
            'episode': episode,
            'dynamic_meo_enabled': True,
            'training_stats': {
                'current_success_rate': self.episode_success_rates[-1] if self.episode_success_rates else 0.0,
                'current_avg_path_length': self.episode_avg_path_lengths[-1] if self.episode_avg_path_lengths else 0.0,
                'meo_reassignments': len(self.meo_reassignment_episodes)
            }
        })

        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        self.logger.info(f"模型已保存到: {model_file}")

    def save_final_results(self, agent: RLAgent):
        """保存最终结果"""
        results_path = self.config.get('output.results_path', 'results/')

        # 保存最终模型
        final_model_file = os.path.join(results_path, 'final_model.json')
        model_data = agent.save_model()
        model_data.update({
            'dynamic_meo_enabled': True,
            'training_completed': datetime.now().isoformat(),
            'final_statistics': agent.get_statistics()
        })

        with open(final_model_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        # 保存训练统计
        stats_file = os.path.join(results_path, 'training_stats.json')
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_success_rates': self.episode_success_rates,
            'episode_avg_path_lengths': self.episode_avg_path_lengths,
            'episode_convergence_rates': self.episode_convergence_rates,
            'meo_reassignment_episodes': self.meo_reassignment_episodes,
            'network_efficiency_evolution': self.network_efficiency_evolution,
            'config': self.config.config,
            'final_agent_stats': agent.get_statistics()
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"最终结果已保存到: {results_path}")

    def plot_training_results(self):
        """绘制训练结果"""
        results_path = self.config.get('output.results_path', 'results/')

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('MEO-LEO路由系统训练结果', fontsize=16)

        # 奖励曲线
        axes[0, 0].plot(self.episode_rewards, 'b-', alpha=0.7)
        # 添加平滑曲线
        if len(self.episode_rewards) > 10:
            smoothed = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(self.episode_rewards)), smoothed, 'r-', linewidth=2, label='平滑曲线')
            axes[0, 0].legend()
        axes[0, 0].set_title('训练奖励')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)

        # 成功率曲线
        axes[0, 1].plot(self.episode_success_rates, 'g-', alpha=0.7)
        if len(self.episode_success_rates) > 10:
            smoothed = np.convolve(self.episode_success_rates, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(range(9, len(self.episode_success_rates)), smoothed, 'r-', linewidth=2)
        axes[0, 1].set_title('路由成功率')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # 平均路径长度
        axes[0, 2].plot(self.episode_avg_path_lengths, 'orange', alpha=0.7)
        if len(self.episode_avg_path_lengths) > 10:
            smoothed = np.convolve(self.episode_avg_path_lengths, np.ones(10)/10, mode='valid')
            axes[0, 2].plot(range(9, len(self.episode_avg_path_lengths)), smoothed, 'r-', linewidth=2)
        axes[0, 2].set_title('平均路径长度')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Path Length')
        axes[0, 2].grid(True, alpha=0.3)

        # 收敛率
        axes[1, 0].plot(self.episode_convergence_rates, 'purple', alpha=0.7)
        axes[1, 0].set_title('快速收敛率')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Convergence Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        # MEO重分配统计
        if self.meo_reassignment_episodes:
            axes[1, 1].hist(self.meo_reassignment_episodes, bins=20, alpha=0.7, color='cyan')
            axes[1, 1].set_title('MEO重分配分布')
            axes[1, 1].set_xlabel('重分配次数')
            axes[1, 1].set_ylabel('频次')
        else:
            axes[1, 1].text(0.5, 0.5, '无MEO重分配数据', ha='center', va='center', transform=axes[1, 1].transAxes)

        # 网络效率演变
        if self.network_efficiency_evolution:
            episodes = [item['episode'] for item in self.network_efficiency_evolution]
            efficiencies = [item['efficiency'] for item in self.network_efficiency_evolution]
            axes[1, 2].scatter(episodes, efficiencies, alpha=0.6, s=10)
            axes[1, 2].set_title('网络效率演变')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Network Efficiency')
        else:
            axes[1, 2].text(0.5, 0.5, '无网络效率数据', ha='center', va='center', transform=axes[1, 2].transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(results_path, 'training_results_enhanced.png'), dpi=300, bbox_inches='tight')

        if self.config.get('output.plot_results', True):
            plt.show()

        self.logger.info(f"训练结果图表已保存到: {results_path}")