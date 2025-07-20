"""
网络仿真模块
负责运行性能仿真和数据收集
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import TrafficClass, SimulationParameters
from routing import QoSRoutingAlgorithm, BaselineAlgorithms
from utils import calculate_path_delay, GeographyUtils


@dataclass
class SimulationResult:
    """仿真结果数据结构"""
    traffic_rate: float
    algorithm: str
    traffic_class: str
    delays: List[float]
    packet_losses: List[float]
    path_lengths: List[int]
    successful_routes: int
    total_attempts: int
    iol_utilization: float


class NetworkSimulation:
    """网络仿真器"""

    def __init__(self, network, algorithm: QoSRoutingAlgorithm,
                 params: SimulationParameters = None):
        self.network = network
        self.algorithm = algorithm
        self.params = params or SimulationParameters()
        self.baseline_algorithms = BaselineAlgorithms(network)

        # 仿真结果存储
        self.simulation_results = {
            'delays': {'class_a': [], 'class_b': []},
            'packet_loss': {'class_a': [], 'class_b': []},
            'iol_utilization': [],
            'detailed_results': []
        }

        # 设置随机种子
        random.seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)

    def run_comprehensive_simulation(self, algorithms: List[str] = None) -> Dict:
        """运行综合仿真实验"""
        if algorithms is None:
            algorithms = ['M-BMDP', 'BMDP', 'MDSP']

        print("开始综合仿真实验...")
        traffic_rates = self._get_traffic_rates()

        all_results = {}

        for algorithm_name in algorithms:
            print(f"\n测试算法: {algorithm_name}")
            algorithm_results = self._run_algorithm_simulation(algorithm_name, traffic_rates)
            all_results[algorithm_name] = algorithm_results

        # 保存详细结果
        self.simulation_results['comprehensive'] = all_results

        print("\n综合仿真完成!")
        return all_results

    def _run_algorithm_simulation(self, algorithm_name: str,
                                  traffic_rates: List[float]) -> Dict:
        """运行特定算法的仿真"""
        algorithm_results = {
            'class_a': {'delays': [], 'packet_losses': [], 'success_rates': []},
            'class_b': {'delays': [], 'packet_losses': [], 'success_rates': []},
            'iol_utilization': []
        }

        for rate in traffic_rates:
            print(f"  流量速率: {rate} kbps")

            # 为每个流量速率运行多次仿真
            class_a_results = self._run_traffic_simulation(
                algorithm_name, rate, TrafficClass.CLASS_A
            )
            class_b_results = self._run_traffic_simulation(
                algorithm_name, rate, TrafficClass.CLASS_B
            )

            # 统计结果
            algorithm_results['class_a']['delays'].append(np.mean(class_a_results.delays))
            algorithm_results['class_a']['packet_losses'].append(np.mean(class_a_results.packet_losses))
            algorithm_results['class_a']['success_rates'].append(
                class_a_results.successful_routes / class_a_results.total_attempts
            )

            algorithm_results['class_b']['delays'].append(np.mean(class_b_results.delays))
            algorithm_results['class_b']['packet_losses'].append(np.mean(class_b_results.packet_losses))
            algorithm_results['class_b']['success_rates'].append(
                class_b_results.successful_routes / class_b_results.total_attempts
            )

            # IOL利用率
            iol_util = self._calculate_current_iol_utilization()
            algorithm_results['iol_utilization'].append(iol_util)

        return algorithm_results

    def _run_traffic_simulation(self, algorithm_name: str, traffic_rate: float,
                                traffic_class: TrafficClass) -> SimulationResult:
        """运行特定流量的仿真"""
        delays = []
        packet_losses = []
        path_lengths = []
        successful_routes = 0

        # 生成测试用例
        test_pairs = self._generate_random_pairs(self.params.default_iterations // 2)

        for source, destination in test_pairs:
            # 根据算法类型选择路由方法
            path = self._route_with_algorithm(
                algorithm_name, source, destination, traffic_class, traffic_rate / 1000
            )

            if path:
                # 计算性能指标
                delay, packet_loss = self._calculate_performance_metrics(path, traffic_class)

                delays.append(delay)
                packet_losses.append(packet_loss)
                path_lengths.append(len(path) - 1)

                if delay < float('inf') and packet_loss < 1.0:
                    successful_routes += 1
            else:
                # 路径不存在
                delays.append(float('inf'))
                packet_losses.append(1.0)
                path_lengths.append(0)

        return SimulationResult(
            traffic_rate=traffic_rate,
            algorithm=algorithm_name,
            traffic_class=traffic_class.value,
            delays=delays,
            packet_losses=packet_losses,
            path_lengths=path_lengths,
            successful_routes=successful_routes,
            total_attempts=len(test_pairs),
            iol_utilization=self._calculate_current_iol_utilization()
        )

    def _route_with_algorithm(self, algorithm_name: str, source: str, destination: str,
                              traffic_class: TrafficClass, bandwidth_req: float) -> List[str]:
        """使用指定算法进行路由"""
        if algorithm_name == 'M-BMDP':
            return self.algorithm.modified_bmdp_algorithm(
                source, destination, traffic_class, bandwidth_req
            )
        elif algorithm_name == 'BMDP':
            return self.baseline_algorithms.bmdp_algorithm(
                source, destination, bandwidth_req
            )
        elif algorithm_name == 'MDSP':
            return self.baseline_algorithms.mdsp_algorithm(source, destination)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def run_hop_limit_analysis(self, hop_limits: List[int] = None) -> Dict:
        """运行跳数限制分析"""
        if hop_limits is None:
            hop_limits = [3, 4, 5, 6, 7]

        print("开始跳数限制分析...")
        hop_results = {'hop_limits': hop_limits, 'delays': [], 'success_rates': []}

        test_pairs = self._generate_random_pairs(50)  # 较小的测试集

        for hop_limit in hop_limits:
            print(f"  跳数限制: {hop_limit}")

            delays = []
            successful = 0

            for source, destination in test_pairs:
                path = self.algorithm.improved_hop_limit_algorithm(
                    source, destination, hop_limit, TrafficClass.CLASS_A
                )

                if path:
                    delay, _ = self._calculate_performance_metrics(path, TrafficClass.CLASS_A)
                    if delay < float('inf'):
                        delays.append(delay)
                        successful += 1
                    else:
                        delays.append(float('inf'))
                else:
                    delays.append(float('inf'))

            valid_delays = [d for d in delays if d < float('inf')]
            avg_delay = np.mean(valid_delays) if valid_delays else float('inf')
            success_rate = successful / len(test_pairs)

            hop_results['delays'].append(avg_delay)
            hop_results['success_rates'].append(success_rate)

        self.simulation_results['hop_limit_analysis'] = hop_results
        print("跳数限制分析完成!")
        return hop_results

    def run_special_area_analysis(self) -> Dict:
        """运行特殊区域（反向裂隙区域）分析"""
        print("开始特殊区域分析...")

        # 识别极地区域的卫星
        polar_satellites = [
            sat for sat in self.network.leo_satellites
            if GeographyUtils.is_in_reversed_crevice_zone(sat)
        ]

        normal_satellites = [
            sat for sat in self.network.leo_satellites
            if not GeographyUtils.is_in_reversed_crevice_zone(sat)
        ]

        if not polar_satellites or not normal_satellites:
            print("未找到足够的极地或正常区域卫星")
            return {}

        special_results = {
            'polar_to_normal': {'delays': [], 'success_rate': 0},
            'normal_to_polar': {'delays': [], 'success_rate': 0},
            'polar_to_polar': {'delays': [], 'success_rate': 0},
            'normal_to_normal': {'delays': [], 'success_rate': 0}
        }

        # 测试不同类型的路由
        test_scenarios = [
            ('polar_to_normal', polar_satellites[:5], normal_satellites[:5]),
            ('normal_to_polar', normal_satellites[:5], polar_satellites[:5]),
            ('polar_to_polar', polar_satellites[:3],
             polar_satellites[3:6] if len(polar_satellites) > 3 else polar_satellites[:3]),
            ('normal_to_normal', normal_satellites[:5],
             normal_satellites[5:10] if len(normal_satellites) > 5 else normal_satellites[:5])
        ]

        for scenario_name, sources, destinations in test_scenarios:
            delays = []
            successful = 0
            total = 0

            for source_sat in sources:
                for dest_sat in destinations:
                    if source_sat.id == dest_sat.id:
                        continue

                    path = self.algorithm.special_area_routing(
                        source_sat.id, dest_sat.id, TrafficClass.CLASS_A
                    )

                    total += 1
                    if path:
                        delay, _ = self._calculate_performance_metrics(path, TrafficClass.CLASS_A)
                        if delay < float('inf'):
                            delays.append(delay)
                            successful += 1
                        else:
                            delays.append(float('inf'))
                    else:
                        delays.append(float('inf'))

            if total > 0:
                special_results[scenario_name]['delays'] = delays
                special_results[scenario_name]['success_rate'] = successful / total
                special_results[scenario_name]['avg_delay'] = np.mean([d for d in delays if d < float('inf')]) if any(
                    d < float('inf') for d in delays) else float('inf')

        self.simulation_results['special_area_analysis'] = special_results
        print("特殊区域分析完成!")
        return special_results

    def _generate_random_pairs(self, num_pairs: int) -> List[Tuple[str, str]]:
        """生成随机的源-目的卫星对"""
        pairs = []
        satellite_ids = list(self.network.satellites.keys())

        for _ in range(num_pairs):
            source = random.choice(satellite_ids)
            destination = random.choice(satellite_ids)
            while destination == source:
                destination = random.choice(satellite_ids)
            pairs.append((source, destination))

        return pairs

    def _calculate_performance_metrics(self, path: List[str],
                                       traffic_class: TrafficClass) -> Tuple[float, float]:
        """计算性能指标"""
        if not path or len(path) < 2:
            return float('inf'), 1.0

        # 计算端到端延迟
        total_delay = calculate_path_delay(path, self.network, self.params.packet_size)

        # 计算包丢失率
        packet_loss_rate = 0.0

        for i in range(len(path) - 1):
            # 模拟基于队列长度和利用率的丢包
            utilization = random.uniform(0.3, 0.9)
            queue_length = random.uniform(0.1, 0.8)

            # 基本丢包率计算
            if utilization > 0.8:
                packet_loss_rate += 0.02 * (utilization - 0.8) / 0.2

            if queue_length > 0.7:
                packet_loss_rate += 0.03 * (queue_length - 0.7) / 0.3

        # 延迟敏感流量的特殊处理
        if traffic_class == TrafficClass.CLASS_A:
            if total_delay > 100:  # 延迟阈值100ms
                packet_loss_rate = min(1.0, packet_loss_rate + 0.5)

        packet_loss_rate = min(1.0, packet_loss_rate)

        return total_delay, packet_loss_rate

    def _calculate_current_iol_utilization(self) -> float:
        """计算当前IOL利用率"""
        iol_links = self.network.get_iol_links()

        if not iol_links:
            return 0.0

        total_utilization = 0.0
        for link in iol_links:
            # 模拟IOL利用率
            utilization = random.uniform(0.4, 0.85)
            total_utilization += utilization

        return total_utilization / len(iol_links)

    def _get_traffic_rates(self) -> List[float]:
        """获取流量速率列表"""
        return list(range(
            self.params.traffic_rate_min,
            self.params.traffic_rate_max + 1,
            self.params.traffic_rate_step
        ))

    def plot_comprehensive_results(self, results: Dict):
        """绘制综合仿真结果"""
        traffic_rates = self._get_traffic_rates()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Class A 延迟对比
        self._plot_delay_comparison(ax1, results, traffic_rates, 'class_a', 'Class A Traffic Delay')

        # 2. Class B 延迟对比
        self._plot_delay_comparison(ax2, results, traffic_rates, 'class_b', 'Class B Traffic Delay')

        # 3. 包丢失率对比
        self._plot_packet_loss_comparison(ax3, results, traffic_rates)

        # 4. IOL利用率
        self._plot_iol_utilization(ax4, results, traffic_rates)

        plt.tight_layout()
        plt.show()

    def _plot_delay_comparison(self, ax, results: Dict, traffic_rates: List[float],
                               traffic_class: str, title: str):
        """绘制延迟对比图"""
        colors = {'M-BMDP': 'red', 'BMDP': 'blue', 'MDSP': 'green'}
        markers = {'M-BMDP': 'o', 'BMDP': 's', 'MDSP': '^'}

        for algorithm in results:
            delays = results[algorithm][traffic_class]['delays']
            valid_delays = [d if d < float('inf') else 200 for d in delays]

            ax.plot(traffic_rates, valid_delays,
                    color=colors.get(algorithm, 'black'),
                    marker=markers.get(algorithm, 'o'),
                    label=f'{algorithm} algorithm',
                    linewidth=2, markersize=6)

        ax.set_xlabel('Data transmission (Kbps)')
        ax.set_ylabel('Delay (ms)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_packet_loss_comparison(self, ax, results: Dict, traffic_rates: List[float]):
        """绘制包丢失率对比图"""
        colors = {'M-BMDP': 'red', 'BMDP': 'blue', 'MDSP': 'green'}
        markers = {'M-BMDP': 'o', 'BMDP': 's', 'MDSP': '^'}

        for algorithm in results:
            losses = [l * 100 for l in results[algorithm]['class_a']['packet_losses']]

            ax.plot(traffic_rates, losses,
                    color=colors.get(algorithm, 'black'),
                    marker=markers.get(algorithm, 'o'),
                    label=f'{algorithm} algorithm',
                    linewidth=2, markersize=6)

        ax.set_xlabel('Data transmission (Kbps)')
        ax.set_ylabel('Packet loss rate (%)')
        ax.set_title('Packet Loss Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_iol_utilization(self, ax, results: Dict, traffic_rates: List[float]):
        """绘制IOL利用率图"""
        if 'M-BMDP' in results:
            utilizations = [u * 100 for u in results['M-BMDP']['iol_utilization']]

            ax.plot(traffic_rates, utilizations,
                    color='purple', marker='o',
                    label='M-BMDP algorithm',
                    linewidth=2, markersize=6)

        ax.set_xlabel('Data transmission (Kbps)')
        ax.set_ylabel('IOL Utilization (%)')
        ax.set_title('Inter-Orbit Link Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_hop_limit_analysis(self, hop_results: Dict):
        """绘制跳数限制分析结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        hop_limits = hop_results['hop_limits']
        delays = hop_results['delays']
        success_rates = hop_results['success_rates']

        # 延迟vs跳数限制
        valid_delays = [d if d < float('inf') else 150 for d in delays]
        ax1.plot(hop_limits, valid_delays, 'o-', color='purple', linewidth=2, markersize=8)
        ax1.set_xlabel('Hop Limit')
        ax1.set_ylabel('Average Delay (ms)')
        ax1.set_title('Delay vs Hop Limit')
        ax1.grid(True, alpha=0.3)

        # 成功率vs跳数限制
        ax2.plot(hop_limits, [sr * 100 for sr in success_rates], 's-',
                 color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Hop Limit')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate vs Hop Limit')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_special_area_analysis(self, special_results: Dict):
        """绘制特殊区域分析结果"""
        scenarios = list(special_results.keys())
        success_rates = [special_results[s]['success_rate'] * 100 for s in scenarios]
        avg_delays = [special_results[s].get('avg_delay', float('inf')) for s in scenarios]

        # 处理无穷大延迟
        avg_delays = [d if d < float('inf') else 200 for d in avg_delays]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 成功率对比
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        bars1 = ax1.bar(range(len(scenarios)), success_rates, color=colors)
        ax1.set_xlabel('Routing Scenarios')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate in Different Areas')
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax1.grid(True, alpha=0.3)

        # 在条形图上添加数值
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{rate:.1f}%', ha='center', va='bottom')

        # 平均延迟对比
        bars2 = ax2.bar(range(len(scenarios)), avg_delays, color=colors)
        ax2.set_xlabel('Routing Scenarios')
        ax2.set_ylabel('Average Delay (ms)')
        ax2.set_title('Average Delay in Different Areas')
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45)
        ax2.grid(True, alpha=0.3)

        # 在条形图上添加数值
        for bar, delay in zip(bars2, avg_delays):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{delay:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def generate_simulation_report(self) -> str:
        """生成仿真报告"""
        report = "# 网络仿真实验报告\n\n"

        if 'comprehensive' in self.simulation_results:
            report += "## 综合算法对比\n\n"
            comprehensive = self.simulation_results['comprehensive']

            for algorithm in comprehensive:
                class_a_avg_delay = np.mean(comprehensive[algorithm]['class_a']['delays'])
                class_a_avg_loss = np.mean(comprehensive[algorithm]['class_a']['packet_losses']) * 100
                class_b_avg_delay = np.mean(comprehensive[algorithm]['class_b']['delays'])
                class_b_avg_loss = np.mean(comprehensive[algorithm]['class_b']['packet_losses']) * 100

                report += f"### {algorithm} 算法\n"
                report += f"- Class A 平均延迟: {class_a_avg_delay:.2f} ms\n"
                report += f"- Class A 平均丢包率: {class_a_avg_loss:.2f}%\n"
                report += f"- Class B 平均延迟: {class_b_avg_delay:.2f} ms\n"
                report += f"- Class B 平均丢包率: {class_b_avg_loss:.2f}%\n\n"

        if 'hop_limit_analysis' in self.simulation_results:
            report += "## 跳数限制分析\n\n"
            hop_results = self.simulation_results['hop_limit_analysis']

            for i, hop_limit in enumerate(hop_results['hop_limits']):
                delay = hop_results['delays'][i]
                success_rate = hop_results['success_rates'][i] * 100
                report += f"- 跳数限制 {hop_limit}: 延迟 {delay:.2f} ms, 成功率 {success_rate:.1f}%\n"

        if 'special_area_analysis' in self.simulation_results:
            report += "\n## 特殊区域分析\n\n"
            special_results = self.simulation_results['special_area_analysis']

            for scenario, results in special_results.items():
                success_rate = results['success_rate'] * 100
                avg_delay = results.get('avg_delay', float('inf'))
                delay_str = f"{avg_delay:.2f} ms" if avg_delay < float('inf') else "N/A"

                report += f"- {scenario.replace('_', ' ').title()}: 成功率 {success_rate:.1f}%, 平均延迟 {delay_str}\n"

        return report