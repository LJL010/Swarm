import random
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import defaultdict


class RLAgent:
    """改进的Q-learning agent，支持更复杂的状态表示和动作选择"""

    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 使用defaultdict避免KeyError
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # 记录访问次数，用于动态学习率
        self.visit_count: Dict[Tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # 经验回放缓冲区
        self.experience_buffer = []
        self.buffer_size = 10000

    def _create_state_representation(self, current_leo: int, target_leo: int,
                                     leos: Dict, meos: Dict,
                                     available_actions: List[int]) -> Tuple:
        """
        创建更丰富的状态表示，包含网络拓扑和负载信息
        """
        if current_leo not in leos or target_leo not in leos:
            return (current_leo, target_leo, 0, 0, 0)

        current_satellite = leos[current_leo]
        target_satellite = leos[target_leo]

        # 基础位置信息
        current_meo = current_satellite.meo_id
        target_meo = target_satellite.meo_id

        # 负载状态（离散化）
        current_load_level = min(current_satellite.load // 3, 3)  # 0-3级别

        # 邻居数量（连通性指标）
        neighbor_count = min(len(current_satellite.neighbors), 5)  # 最多5个邻居

        # 是否跨集群
        is_inter_cluster = 1 if current_meo != target_meo else 0

        # 可用动作数量
        action_count = min(len(available_actions), 5)

        return (current_leo, target_leo, current_meo, target_meo,
                current_load_level, neighbor_count, is_inter_cluster, action_count)

    def choose_action(self, current_leo: int, target_leo: int, available_actions: List[int],
                      leos: Dict = None, meos: Dict = None) -> int:
        """
        改进的动作选择，使用更丰富的状态表示
        """
        if not available_actions:
            return current_leo  # 如果没有可用动作，返回当前位置

        # 创建状态表示
        if leos is not None and meos is not None:
            state = self._create_state_representation(current_leo, target_leo, leos, meos, available_actions)
        else:
            state = (current_leo, target_leo)  # 向后兼容

        # epsilon-greedy策略，加入动态调整
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        # 选择Q值最高的动作
        q_values = {action: self.q_table[state][action] for action in available_actions}

        # 如果所有Q值都相同，随机选择
        if len(set(q_values.values())) == 1:
            return random.choice(available_actions)

        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]

        return random.choice(best_actions)

    def update(self, current_leo: int, target_leo: int, action: int, reward: float,
               next_leo: int, next_target: int, next_actions: List[int],
               leos: Dict = None, meos: Dict = None, done: bool = False):
        """
        改进的Q值更新，支持动态学习率和经验回放
        """
        # 创建状态表示
        if leos is not None and meos is not None:
            state = self._create_state_representation(current_leo, target_leo, leos, meos, [action])
            next_state = self._create_state_representation(next_leo, next_target, leos, meos, next_actions)
        else:
            state = (current_leo, target_leo)
            next_state = (next_leo, next_target)

        # 计算TD目标
        if done or not next_actions:
            td_target = reward
        else:
            next_q_values = [self.q_table[next_state][a] for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0.0
            td_target = reward + self.gamma * max_next_q

        # 动态学习率（基于访问次数）
        self.visit_count[state][action] += 1
        visit_count = self.visit_count[state][action]
        dynamic_lr = self.lr / (1 + 0.01 * visit_count)  # 随访问次数递减

        # 更新Q值
        old_q = self.q_table[state][action]
        td_error = td_target - old_q
        self.q_table[state][action] = old_q + dynamic_lr * td_error

        # 经验回放
        experience = (state, action, reward, next_state, next_actions, done)
        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(experience)

        # 定期进行经验回放
        if len(self.experience_buffer) > 100 and random.random() < 0.1:
            self._replay_experience()

    def _replay_experience(self, batch_size: int = 32):
        """
        经验回放，从历史经验中学习
        """
        if len(self.experience_buffer) < batch_size:
            return

        # 随机采样经验
        batch = random.sample(self.experience_buffer, batch_size)

        for state, action, reward, next_state, next_actions, done in batch:
            if done or not next_actions:
                td_target = reward
            else:
                next_q_values = [self.q_table[next_state][a] for a in next_actions]
                max_next_q = max(next_q_values) if next_q_values else 0.0
                td_target = reward + self.gamma * max_next_q

            # 使用较小的学习率进行经验回放更新
            old_q = self.q_table[state][action]
            td_error = td_target - old_q
            self.q_table[state][action] = old_q + 0.01 * td_error

    def decay_epsilon(self):
        """
        衰减探索率
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, state: Tuple, action: int) -> float:
        """
        获取Q值
        """
        return self.q_table[state][action]

    def get_state_value(self, state: Tuple, actions: List[int]) -> float:
        """
        获取状态值（所有动作的最大Q值）
        """
        if not actions:
            return 0.0
        return max(self.q_table[state][action] for action in actions)

    def save_model(self) -> Dict:
        """
        保存模型参数
        """
        return {
            'q_table': {str(k): dict(v) for k, v in self.q_table.items()},
            'visit_count': {str(k): dict(v) for k, v in self.visit_count.items()},
            'learning_rate': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }

    def load_model(self, model_data: Dict):
        """
        加载模型参数
        """
        # 重构Q表
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_str, actions in model_data.get('q_table', {}).items():
            try:
                state = eval(state_str)  # 注意：生产环境应使用更安全的方法
                for action, q_value in actions.items():
                    self.q_table[state][int(action)] = float(q_value)
            except:
                continue

        # 重构访问计数
        self.visit_count = defaultdict(lambda: defaultdict(int))
        for state_str, actions in model_data.get('visit_count', {}).items():
            try:
                state = eval(state_str)
                for action, count in actions.items():
                    self.visit_count[state][int(action)] = int(count)
            except:
                continue

        # 加载其他参数
        self.lr = model_data.get('learning_rate', self.lr)
        self.gamma = model_data.get('gamma', self.gamma)
        self.epsilon = model_data.get('epsilon', self.epsilon)
        self.epsilon_decay = model_data.get('epsilon_decay', self.epsilon_decay)
        self.epsilon_min = model_data.get('epsilon_min', self.epsilon_min)

    def get_statistics(self) -> Dict:
        """
        获取训练统计信息
        """
        total_states = len(self.q_table)
        total_state_actions = sum(len(actions) for actions in self.q_table.values())
        total_visits = sum(sum(actions.values()) for actions in self.visit_count.values())

        return {
            'total_states': total_states,
            'total_state_actions': total_state_actions,
            'total_visits': total_visits,
            'experience_buffer_size': len(self.experience_buffer),
            'current_epsilon': self.epsilon
        }