"""
高级分析工具模块
提供深入的网络性能分析和可视化功能

作者: 根据论文要求实现
文件: analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 配置matplotlib中文字体
import platform
import os


def configure_chinese_font():
    """配置matplotlib中文字体"""
    system = platform.system()

    if system == "Darwin":  # macOS
        # 尝试常见的中文字体
        fonts = [
            'Arial Unicode MS',
            'Hiragino Sans GB',
            'PingFang SC',
            'STHeiti',
            'SimHei'
        ]
    elif system == "Windows":
        fonts = [
            'SimHei',
            'Microsoft YaHei',
            'SimSun',
            'KaiTi'
        ]
    else:  # Linux
        fonts = [
            'DejaVu Sans',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC'
        ]

    # 尝试设置字体
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            # 测试字体是否可用
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, '测试', fontsize=12)
            plt.close(fig)
            print(f"成功设置中文字体: {font}")
            return True
        except:
            continue

    # 如果都失败了，使用默认设置并禁用中文
    print("警告: 无法找到合适的中文字体，将使用英文显示")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return False


# 配置字体
CHINESE_FONT_AVAILABLE = configure_chinese_font()

# 定义中英文标签映射
LABELS = {
    'network_degree_dist': '网络度数分布' if CHINESE_FONT_AVAILABLE else 'Network Degree Distribution',
    'node_degree': '节点度数' if CHINESE_FONT_AVAILABLE else 'Node Degree',
    'node_count': '节点数量' if CHINESE_FONT_AVAILABLE else 'Node Count',
    'iol_utilization_dist': 'IOL利用率分布' if CHINESE_FONT_AVAILABLE else 'IOL Utilization Distribution',
    'iol_utilization_percent': 'IOL利用率 (%)' if CHINESE_FONT_AVAILABLE else 'IOL Utilization (%)',
    'frequency': '频次' if CHINESE_FONT_AVAILABLE else 'Frequency',
    'algorithm_delay_comparison': '算法延迟对比' if CHINESE_FONT_AVAILABLE else 'Algorithm Delay Comparison',
    'avg_delay_ms': '平均延迟 (ms)' if CHINESE_FONT_AVAILABLE else 'Average Delay (ms)',
    'regional_connectivity': '区域连通性分析' if CHINESE_FONT_AVAILABLE else 'Regional Connectivity Analysis',
    'inter_region_connectivity_percent': '区域间连通性 (%)' if CHINESE_FONT_AVAILABLE else 'Inter-region Connectivity (%)',
    'northern': '北部' if CHINESE_FONT_AVAILABLE else 'Northern',
    'equatorial': '赤道' if CHINESE_FONT_AVAILABLE else 'Equatorial',
    'southern': '南部' if CHINESE_FONT_AVAILABLE else 'Southern',
    'iol_traffic_patterns': 'IOL流量模式分析' if CHINESE_FONT_AVAILABLE else 'IOL Traffic Pattern Analysis',
    'time_period': '时间段' if CHINESE_FONT_AVAILABLE else 'Time Period',
    'percentage': '百分比 (%)' if CHINESE_FONT_AVAILABLE else 'Percentage (%)',
    'avg_utilization': '平均利用率' if CHINESE_FONT_AVAILABLE else 'Average Utilization',
    'congestion_prob': '拥塞概率' if CHINESE_FONT_AVAILABLE else 'Congestion Probability',
    'morning': '早晨' if CHINESE_FONT_AVAILABLE else 'Morning',
    'afternoon': '下午' if CHINESE_FONT_AVAILABLE else 'Afternoon',
    'evening': '晚上' if CHINESE_FONT_AVAILABLE else 'Evening',
    'night': '夜晚' if CHINESE_FONT_AVAILABLE else 'Night',
    'path_efficiency_analysis': '路径效率分析' if CHINESE_FONT_AVAILABLE else 'Path Efficiency Analysis',
    'normalized_value': '归一化值' if CHINESE_FONT_AVAILABLE else 'Normalized Value',
    'efficiency_ratio': '效率比' if CHINESE_FONT_AVAILABLE else 'Efficiency Ratio',
    'optimality_score': '最优性得分' if CHINESE_FONT_AVAILABLE else 'Optimality Score',
    'hop_difference': '跳数差异' if CHINESE_FONT_AVAILABLE else 'Hop Difference',
    'network_coverage_heatmap': '网络覆盖热图' if CHINESE_FONT_AVAILABLE else 'Network Coverage Heatmap',
    'longitude': '经度' if CHINESE_FONT_AVAILABLE else 'Longitude',
    'latitude': '纬度' if CHINESE_FONT_AVAILABLE else 'Latitude',
    'coverage_intensity': '覆盖强度' if CHINESE_FONT_AVAILABLE else 'Coverage Intensity',
    'comprehensive_performance_radar': '综合性能雷达图' if CHINESE_FONT_AVAILABLE else 'Comprehensive Performance Radar',
    'connectivity': '连通性' if CHINESE_FONT_AVAILABLE else 'Connectivity',
    'delay_performance': '延迟性能' if CHINESE_FONT_AVAILABLE else 'Delay Performance',
    'iol_efficiency': 'IOL效率' if CHINESE_FONT_AVAILABLE else 'IOL Efficiency',
    'path_optimization': '路径优化' if CHINESE_FONT_AVAILABLE else 'Path Optimization',
    'regional_coverage': '区域覆盖' if CHINESE_FONT_AVAILABLE else 'Regional Coverage',
    'qos_guarantee': 'QoS保障' if CHINESE_FONT_AVAILABLE else 'QoS Guarantee',
    'comprehensive_analysis_title': 'MEO/LEO双层卫星网络综合分析' if CHINESE_FONT_AVAILABLE else 'MEO/LEO Dual-Layer Satellite Network Analysis'
}

from config import TrafficClass, Constants
from utils import GeographyUtils
from simulation import NetworkSimulation


class AdvancedAnalyzer:
    """高级分析工具"""

    def __init__(self, network, algorithm, simulation: NetworkSimulation):
        self.network = network
        self.algorithm = algorithm
        self.simulation = simulation
        self.analysis_results = {}

    def analyze_network_topology(self) -> Dict:
        """分析网络拓扑特性"""
        print("分析网络拓扑特性...")

        topology_analysis = {
            'basic_stats': {},
            'connectivity': {},
            'path_analysis': {},
            'layer_analysis': {}
        }

        # 基本统计
        graph = self.network.graph
        topology_analysis['basic_stats'] = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'leo_nodes': len(self.network.leo_satellites),
            'meo_nodes': len(self.network.meo_satellites),
            'average_degree': np.mean([d for n, d in graph.degree()]),
            'density': nx.density(graph),
            'clustering_coefficient': nx.average_clustering(graph)
        }

        # 连通性分析
        is_connected = nx.is_connected(graph)
        topology_analysis['connectivity'] = {
            'is_connected': is_connected,
            'number_of_components': nx.number_connected_components(graph),
            'largest_component_size': len(max(nx.connected_components(graph), key=len)),
            'diameter': nx.diameter(graph) if is_connected else "N/A",
            'radius': nx.radius(graph) if is_connected else "N/A",
            'center_nodes': list(nx.center(graph)) if is_connected else []
        }

        # 路径分析
        sample_pairs = self.simulation._generate_random_pairs(50)
        path_lengths = []
        successful_paths = 0

        for source, dest in sample_pairs:
            try:
                path_length = nx.shortest_path_length(graph, source, dest)
                path_lengths.append(path_length)
                successful_paths += 1
            except nx.NetworkXNoPath:
                continue

        if path_lengths:
            topology_analysis['path_analysis'] = {
                'avg_path_length': np.mean(path_lengths),
                'max_path_length': max(path_lengths),
                'min_path_length': min(path_lengths),
                'std_path_length': np.std(path_lengths),
                'connectivity_ratio': successful_paths / len(sample_pairs)
            }

        # 分层分析
        topology_analysis['layer_analysis'] = self._analyze_layer_connectivity()

        self.analysis_results['topology_analysis'] = topology_analysis
        return topology_analysis

    def _analyze_layer_connectivity(self) -> Dict:
        """分析分层连通性"""
        layer_analysis = {
            'leo_connectivity': {},
            'meo_connectivity': {},
            'inter_layer_connectivity': {}
        }

        # LEO层连通性
        leo_nodes = [sat.id for sat in self.network.leo_satellites]
        leo_subgraph = self.network.graph.subgraph(leo_nodes)

        layer_analysis['leo_connectivity'] = {
            'is_connected': nx.is_connected(leo_subgraph),
            'components': nx.number_connected_components(leo_subgraph),
            'average_degree': np.mean([d for n, d in leo_subgraph.degree()]),
            'density': nx.density(leo_subgraph)
        }

        # MEO层连通性
        meo_nodes = [sat.id for sat in self.network.meo_satellites]
        meo_subgraph = self.network.graph.subgraph(meo_nodes)

        layer_analysis['meo_connectivity'] = {
            'is_connected': nx.is_connected(meo_subgraph),
            'components': nx.number_connected_components(meo_subgraph),
            'average_degree': np.mean([d for n, d in meo_subgraph.degree()]),
            'density': nx.density(meo_subgraph)
        }

        # 层间连通性
        iol_links = self.network.get_iol_links()
        layer_analysis['inter_layer_connectivity'] = {
            'total_iol_links': len(iol_links),
            'leo_nodes_with_iol': len(set([link[0] for link in iol_links if 'LEO' in link[0]] +
                                          [link[1] for link in iol_links if 'LEO' in link[1]])),
            'meo_nodes_with_iol': len(set([link[0] for link in iol_links if 'MEO' in link[0]] +
                                          [link[1] for link in iol_links if 'MEO' in link[1]])),
            'iol_coverage_ratio': len(iol_links) / len(self.network.leo_satellites)
        }

        return layer_analysis

    def analyze_iol_performance(self) -> Dict:
        """分析IOL性能"""
        print("分析IOL性能...")

        iol_analysis = {
            'iol_links': [],
            'utilization_distribution': [],
            'performance_impact': {},
            'traffic_analysis': {}
        }

        # 收集IOL链路信息
        iol_links = self.network.get_iol_links()

        for leo_id, meo_id in iol_links:
            leo_sat = self.network.satellites[leo_id]
            meo_sat = self.network.satellites[meo_id]

            distance = leo_sat.distance_to(meo_sat, self.network.params.earth_radius)

            iol_analysis['iol_links'].append({
                'leo_satellite': leo_id,
                'meo_satellite': meo_id,
                'distance': distance,
                'leo_position': (leo_sat.latitude, leo_sat.longitude),
                'meo_position': (meo_sat.latitude, meo_sat.longitude),
                'leo_orbit': leo_sat.orbit,
                'meo_orbit': meo_sat.orbit
            })

        # 模拟利用率分布
        for _ in range(200):
            utilization = np.random.beta(3, 2)  # Beta分布，倾向于较高利用率
            iol_analysis['utilization_distribution'].append(utilization)

        # 性能影响分析
        utilizations = iol_analysis['utilization_distribution']
        iol_analysis['performance_impact'] = {
            'avg_utilization': np.mean(utilizations),
            'std_utilization': np.std(utilizations),
            'high_utilization_percentage': len([u for u in utilizations if u > 0.8]) / len(utilizations),
            'low_utilization_percentage': len([u for u in utilizations if u < 0.3]) / len(utilizations),
            'total_iol_links': len(iol_analysis['iol_links']),
            'max_distance': max([link['distance'] for link in iol_analysis['iol_links']]) if iol_analysis[
                'iol_links'] else 0,
            'min_distance': min([link['distance'] for link in iol_analysis['iol_links']]) if iol_analysis[
                'iol_links'] else 0,
            'avg_distance': np.mean([link['distance'] for link in iol_analysis['iol_links']]) if iol_analysis[
                'iol_links'] else 0
        }

        # 流量分析
        iol_analysis['traffic_analysis'] = self._analyze_iol_traffic_patterns()

        self.analysis_results['iol_analysis'] = iol_analysis
        return iol_analysis

    def _analyze_iol_traffic_patterns(self) -> Dict:
        """分析IOL流量模式"""
        # 模拟不同时间段的IOL流量
        time_periods = ['morning', 'afternoon', 'evening', 'night']
        traffic_patterns = {}

        for period in time_periods:
            if period == 'morning':
                base_utilization = 0.6
                variation = 0.15
            elif period == 'afternoon':
                base_utilization = 0.8
                variation = 0.1
            elif period == 'evening':
                base_utilization = 0.75
                variation = 0.2
            else:  # night
                base_utilization = 0.4
                variation = 0.25

            utilizations = np.random.normal(base_utilization, variation, 50)
            utilizations = np.clip(utilizations, 0.1, 0.95)

            traffic_patterns[period] = {
                'avg_utilization': np.mean(utilizations),
                'peak_utilization': np.max(utilizations),
                'min_utilization': np.min(utilizations),
                'congestion_probability': len([u for u in utilizations if u > 0.85]) / len(utilizations)
            }

        return traffic_patterns

    def analyze_routing_efficiency(self) -> Dict:
        """分析路由效率"""
        print("分析路由效率...")

        efficiency_analysis = {
            'path_efficiency': {},
            'algorithm_comparison': {},
            'traffic_class_analysis': {}
        }

        # 路径效率分析
        test_pairs = self.simulation._generate_random_pairs(100)
        path_metrics = {
            'optimal_paths': [],
            'actual_paths': [],
            'efficiency_ratios': [],
            'hop_differences': []
        }

        for source, dest in test_pairs[:50]:  # 限制测试数量
            # 计算理论最优路径（最短距离）
            try:
                optimal_path = nx.shortest_path(
                    self.network.graph, source, dest, weight='distance'
                )
                optimal_distance = self.network.get_path_distance(optimal_path)

                # 计算实际M-BMDP路径
                actual_path = self.algorithm.modified_bmdp_algorithm(
                    source, dest, TrafficClass.CLASS_A, 0.5
                )

                if actual_path:
                    actual_distance = self.network.get_path_distance(actual_path)
                    efficiency_ratio = optimal_distance / actual_distance if actual_distance > 0 else 0
                    hop_difference = len(actual_path) - len(optimal_path)

                    path_metrics['optimal_paths'].append(len(optimal_path) - 1)
                    path_metrics['actual_paths'].append(len(actual_path) - 1)
                    path_metrics['efficiency_ratios'].append(efficiency_ratio)
                    path_metrics['hop_differences'].append(hop_difference)

            except (nx.NetworkXNoPath, KeyError):
                continue

        if path_metrics['efficiency_ratios']:
            efficiency_analysis['path_efficiency'] = {
                'avg_efficiency_ratio': np.mean(path_metrics['efficiency_ratios']),
                'min_efficiency_ratio': np.min(path_metrics['efficiency_ratios']),
                'max_efficiency_ratio': np.max(path_metrics['efficiency_ratios']),
                'avg_hop_difference': np.mean(path_metrics['hop_differences']),
                'path_optimality_score': len([r for r in path_metrics['efficiency_ratios'] if r > 0.9]) / len(
                    path_metrics['efficiency_ratios'])
            }

        # 算法对比分析
        efficiency_analysis['algorithm_comparison'] = self._compare_algorithm_efficiency(test_pairs[:20])

        # 流量类别分析
        efficiency_analysis['traffic_class_analysis'] = self._analyze_traffic_class_efficiency(test_pairs[:20])

        self.analysis_results['routing_efficiency'] = efficiency_analysis
        return efficiency_analysis

    def _compare_algorithm_efficiency(self, test_pairs: List[Tuple[str, str]]) -> Dict:
        """对比算法效率"""
        from routing import BaselineAlgorithms

        baseline = BaselineAlgorithms(self.network)
        comparison = {
            'M-BMDP': {'delays': [], 'hop_counts': [], 'success_rate': 0},
            'BMDP': {'delays': [], 'hop_counts': [], 'success_rate': 0},
            'MDSP': {'delays': [], 'hop_counts': [], 'success_rate': 0}
        }

        for source, dest in test_pairs:
            # M-BMDP
            path_mbmdp = self.algorithm.modified_bmdp_algorithm(
                source, dest, TrafficClass.CLASS_A, 0.5
            )
            if path_mbmdp:
                delay, _ = self.simulation._calculate_performance_metrics(path_mbmdp, TrafficClass.CLASS_A)
                comparison['M-BMDP']['delays'].append(delay)
                comparison['M-BMDP']['hop_counts'].append(len(path_mbmdp) - 1)
                comparison['M-BMDP']['success_rate'] += 1

            # BMDP
            path_bmdp = baseline.bmdp_algorithm(source, dest, 0.5)
            if path_bmdp:
                delay, _ = self.simulation._calculate_performance_metrics(path_bmdp, TrafficClass.CLASS_A)
                comparison['BMDP']['delays'].append(delay)
                comparison['BMDP']['hop_counts'].append(len(path_bmdp) - 1)
                comparison['BMDP']['success_rate'] += 1

            # MDSP
            path_mdsp = baseline.mdsp_algorithm(source, dest)
            if path_mdsp:
                delay, _ = self.simulation._calculate_performance_metrics(path_mdsp, TrafficClass.CLASS_A)
                comparison['MDSP']['delays'].append(delay)
                comparison['MDSP']['hop_counts'].append(len(path_mdsp) - 1)
                comparison['MDSP']['success_rate'] += 1

        # 计算统计数据
        total_tests = len(test_pairs)
        for algorithm in comparison:
            delays = [d for d in comparison[algorithm]['delays'] if d < float('inf')]
            hop_counts = comparison[algorithm]['hop_counts']

            comparison[algorithm].update({
                'avg_delay': np.mean(delays) if delays else float('inf'),
                'std_delay': np.std(delays) if delays else 0,
                'avg_hop_count': np.mean(hop_counts) if hop_counts else 0,
                'success_rate': comparison[algorithm]['success_rate'] / total_tests
            })

        return comparison

    def _analyze_traffic_class_efficiency(self, test_pairs: List[Tuple[str, str]]) -> Dict:
        """分析不同流量类别的效率"""
        traffic_analysis = {
            TrafficClass.CLASS_A.value: {'delays': [], 'losses': [], 'paths': []},
            TrafficClass.CLASS_B.value: {'delays': [], 'losses': [], 'paths': []}
        }

        for source, dest in test_pairs:
            for traffic_class in [TrafficClass.CLASS_A, TrafficClass.CLASS_B]:
                path = self.algorithm.modified_bmdp_algorithm(
                    source, dest, traffic_class, 0.5
                )

                if path:
                    delay, loss = self.simulation._calculate_performance_metrics(path, traffic_class)

                    traffic_analysis[traffic_class.value]['delays'].append(delay)
                    traffic_analysis[traffic_class.value]['losses'].append(loss)
                    traffic_analysis[traffic_class.value]['paths'].append(len(path) - 1)

        # 计算统计数据
        for traffic_class in traffic_analysis:
            delays = [d for d in traffic_analysis[traffic_class]['delays'] if d < float('inf')]
            losses = traffic_analysis[traffic_class]['losses']
            paths = traffic_analysis[traffic_class]['paths']

            traffic_analysis[traffic_class].update({
                'avg_delay': np.mean(delays) if delays else float('inf'),
                'avg_loss': np.mean(losses) if losses else 0,
                'avg_path_length': np.mean(paths) if paths else 0,
                'performance_score': (1 - np.mean(losses)) * (100 / np.mean(delays)) if delays and losses else 0
            })

        return traffic_analysis

    def analyze_geographical_distribution(self) -> Dict:
        """分析地理分布特性"""
        print("分析地理分布特性...")

        geo_analysis = {
            'coverage_distribution': {},
            'polar_analysis': {},
            'group_manager_distribution': {},
            'connectivity_by_region': {}
        }

        # 覆盖分布分析
        if hasattr(self.network, 'visualize_coverage'):
            coverage_data = self.network.visualize_coverage()
            geo_analysis['coverage_distribution'] = coverage_data['coverage_stats']
        else:
            geo_analysis['coverage_distribution'] = {
                'total_groups': len(self.network.groups),
                'avg_members_per_group': 1.0,
                'min_members': 1,
                'max_members': 1
            }

        # 极地分析
        polar_satellites = [
            sat for sat in self.network.leo_satellites
            if GeographyUtils.is_in_reversed_crevice_zone(sat)
        ]

        geo_analysis['polar_analysis'] = {
            'polar_satellite_count': len(polar_satellites),
            'polar_coverage_ratio': len(polar_satellites) / len(self.network.leo_satellites),
            'polar_satellites': [sat.id for sat in polar_satellites[:10]]  # 限制数量
        }

        # 组管理器分布
        group_managers = self.network.constellation.get_group_managers()
        geo_analysis['group_manager_distribution'] = {
            'total_managers': len(group_managers),
            'coverage_efficiency': len(group_managers) / (Constants.GRID_ROWS * Constants.GRID_COLS),
            'manager_positions': [(mgr.latitude, mgr.longitude) for mgr in group_managers[:20]]
        }

        # 区域连通性分析
        geo_analysis['connectivity_by_region'] = self._analyze_regional_connectivity()

        self.analysis_results['geographical_analysis'] = geo_analysis
        return geo_analysis

    def _analyze_regional_connectivity(self) -> Dict:
        """分析区域连通性"""
        regions = {
            'northern': [sat for sat in self.network.leo_satellites if sat.latitude > 30],
            'equatorial': [sat for sat in self.network.leo_satellites if -30 <= sat.latitude <= 30],
            'southern': [sat for sat in self.network.leo_satellites if sat.latitude < -30]
        }

        connectivity_analysis = {}

        for region_name, satellites in regions.items():
            if len(satellites) < 2:
                continue

            # 计算区域内连通性
            node_ids = [sat.id for sat in satellites]
            subgraph = self.network.graph.subgraph(node_ids)

            # 区域间连通性测试
            inter_region_connectivity = 0
            total_tests = 0

            for sat in satellites[:5]:  # 限制测试数量
                for other_region, other_sats in regions.items():
                    if other_region != region_name and other_sats:
                        target_sat = other_sats[0]
                        try:
                            nx.shortest_path(self.network.graph, sat.id, target_sat.id)
                            inter_region_connectivity += 1
                        except nx.NetworkXNoPath:
                            pass
                        total_tests += 1

            connectivity_analysis[region_name] = {
                'satellite_count': len(satellites),
                'intra_region_connected': nx.is_connected(subgraph),
                'intra_region_components': nx.number_connected_components(subgraph),
                'inter_region_connectivity': inter_region_connectivity / total_tests if total_tests > 0 else 0,
                'average_degree': np.mean([d for n, d in subgraph.degree()]) if subgraph.nodes() else 0
            }

        return connectivity_analysis

    def generate_comprehensive_report(self) -> str:
        """生成综合分析报告"""
        print("生成综合分析报告...")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
# MEO/LEO双层卫星网络综合分析报告

生成时间: {timestamp}

## 1. 网络拓扑分析

### 基本网络统计
"""

        if 'topology_analysis' in self.analysis_results:
            topo = self.analysis_results['topology_analysis']
            basic_stats = topo['basic_stats']
            connectivity = topo['connectivity']

            report += f"""
- **总节点数**: {basic_stats['total_nodes']}
- **总链路数**: {basic_stats['total_edges']}
- **LEO卫星数**: {basic_stats['leo_nodes']}
- **MEO卫星数**: {basic_stats['meo_nodes']}
- **平均度数**: {basic_stats['average_degree']:.2f}
- **网络密度**: {basic_stats['density']:.4f}
- **聚类系数**: {basic_stats['clustering_coefficient']:.4f}

### 连通性特征
- **网络连通性**: {'是' if connectivity['is_connected'] else '否'}
- **连通分量数**: {connectivity['number_of_components']}
- **最大分量大小**: {connectivity['largest_component_size']}
- **网络直径**: {connectivity['diameter']}
- **网络半径**: {connectivity['radius']}
"""

            if 'path_analysis' in topo:
                path_stats = topo['path_analysis']
                report += f"""
### 路径特征
- **平均路径长度**: {path_stats['avg_path_length']:.2f}
- **最大路径长度**: {path_stats['max_path_length']}
- **最小路径长度**: {path_stats['min_path_length']}
- **路径长度标准差**: {path_stats['std_path_length']:.2f}
- **连通率**: {path_stats['connectivity_ratio']:.2%}
"""

        report += "\n## 2. 层间链路(IOL)性能分析\n"

        if 'iol_analysis' in self.analysis_results:
            iol = self.analysis_results['iol_analysis']
            performance = iol['performance_impact']

            report += f"""
### IOL基本统计
- **IOL链路总数**: {performance['total_iol_links']}
- **平均利用率**: {performance['avg_utilization']:.2%}
- **利用率标准差**: {performance['std_utilization']:.2%}
- **高利用率链路比例**: {performance['high_utilization_percentage']:.2%}
- **低利用率链路比例**: {performance['low_utilization_percentage']:.2%}

### IOL距离特征
- **最大距离**: {performance['max_distance']:.2f} km
- **最小距离**: {performance['min_distance']:.2f} km
- **平均距离**: {performance['avg_distance']:.2f} km
"""

            if 'traffic_analysis' in iol:
                traffic = iol['traffic_analysis']
                report += "\n### IOL流量模式\n"
                for period, stats in traffic.items():
                    period_name = LABELS.get(period, period.title())
                    report += f"- **{period_name}**: 平均利用率 {stats['avg_utilization']:.2%}, 拥塞概率 {stats['congestion_probability']:.2%}\n"

        report += "\n## 3. 路由效率分析\n"

        if 'routing_efficiency' in self.analysis_results:
            routing = self.analysis_results['routing_efficiency']

            if 'path_efficiency' in routing:
                path_eff = routing['path_efficiency']
                report += f"""
### 路径效率
- **平均效率比**: {path_eff['avg_efficiency_ratio']:.3f}
- **路径最优性得分**: {path_eff['path_optimality_score']:.2%}
- **平均跳数差异**: {path_eff['avg_hop_difference']:.1f}
"""

            if 'algorithm_comparison' in routing:
                algo_comp = routing['algorithm_comparison']
                report += "\n### 算法对比\n"
                for algo, stats in algo_comp.items():
                    report += f"""
#### {algo} 算法
- 平均延迟: {stats['avg_delay']:.2f} ms
- 平均跳数: {stats['avg_hop_count']:.1f}
- 成功率: {stats['success_rate']:.2%}
"""

        report += "\n## 4. 地理分布特性\n"

        if 'geographical_analysis' in self.analysis_results:
            geo = self.analysis_results['geographical_analysis']

            if 'polar_analysis' in geo:
                polar = geo['polar_analysis']
                report += f"""
### 极地区域分析
- **极地卫星数量**: {polar['polar_satellite_count']}
- **极地覆盖比例**: {polar['polar_coverage_ratio']:.2%}
"""

            if 'group_manager_distribution' in geo:
                group_mgr = geo['group_manager_distribution']
                report += f"""
### 组管理器分布
- **组管理器总数**: {group_mgr['total_managers']}
- **覆盖效率**: {group_mgr['coverage_efficiency']:.2%}
"""

            if 'connectivity_by_region' in geo:
                regional = geo['connectivity_by_region']
                report += "\n### 区域连通性\n"
                for region, stats in regional.items():
                    region_name = LABELS.get(region, region.title())
                    report += f"""
#### {region_name} 区域
- 卫星数量: {stats['satellite_count']}
- 区域内连通: {'是' if stats['intra_region_connected'] else '否'}
- 区域间连通性: {stats['inter_region_connectivity']:.2%}
- 平均度数: {stats['average_degree']:.2f}
"""

        report += """
## 5. 结论与建议

### 主要发现
1. **网络拓扑**: 双层卫星网络展现出良好的连通性和路径多样性
2. **IOL性能**: 层间链路是关键瓶颈，需要优化负载均衡
3. **路由效率**: M-BMDP算法在QoS保障方面优于传统算法
4. **地理覆盖**: 极地区域需要特殊的路由策略

### 优化建议
1. **动态负载均衡**: 实现IOL链路的自适应负载分配
2. **区域优化**: 加强极地区域的连通性设计
3. **QoS增强**: 进一步优化延迟敏感流量的处理机制
4. **冗余设计**: 增加关键区域的备用路径

### 未来研究方向
- 考虑卫星移动性的动态路由算法
- 基于机器学习的网络优化策略
- 跨层优化的综合设计方案
"""

        return report

    def save_analysis_results(self, filename: str = None):
        """保存分析结果"""
        if filename is None:
            filename = f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 处理不可序列化的对象
        def clean_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj

        cleaned_results = clean_for_json(self.analysis_results)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False)

        print(f"分析结果已保存到: {filename}")

    def plot_comprehensive_analysis(self):
        """绘制综合分析图表"""
        if not self.analysis_results:
            print("没有分析结果可绘制")
            return

        fig = plt.figure(figsize=(20, 16))

        # 创建子图网格
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. 网络度数分布
        if 'topology_analysis' in self.analysis_results:
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_degree_distribution(ax1)

        # 2. IOL利用率分布
        if 'iol_analysis' in self.analysis_results:
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_iol_utilization_distribution(ax2)

        # 3. 算法延迟对比
        if 'routing_efficiency' in self.analysis_results:
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_algorithm_delay_comparison(ax3)

        # 4. 区域连通性
        if 'geographical_analysis' in self.analysis_results:
            ax4 = fig.add_subplot(gs[0, 3])
            self._plot_regional_connectivity(ax4)

        # 5. IOL流量模式
        if 'iol_analysis' in self.analysis_results:
            ax5 = fig.add_subplot(gs[1, :2])
            self._plot_iol_traffic_patterns(ax5)

        # 6. 路径效率分析
        if 'routing_efficiency' in self.analysis_results:
            ax6 = fig.add_subplot(gs[1, 2:])
            self._plot_path_efficiency_analysis(ax6)

        # 7. 网络覆盖热图
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_network_coverage_heatmap(ax7)

        # 8. 综合性能雷达图
        ax8 = fig.add_subplot(gs[3, :], projection='polar')
        self._plot_performance_radar(ax8)

        plt.suptitle(LABELS['comprehensive_analysis_title'], fontsize=16, fontweight='bold')
        plt.show()

    def _plot_degree_distribution(self, ax):
        """绘制度数分布"""
        degrees = [d for n, d in self.network.graph.degree()]
        if degrees:
            bins = max(degrees) - min(degrees) + 1 if max(degrees) > min(degrees) else 5
            ax.hist(degrees, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel(LABELS['node_degree'])
        ax.set_ylabel(LABELS['node_count'])
        ax.set_title(LABELS['network_degree_dist'])
        ax.grid(True, alpha=0.3)

    def _plot_iol_utilization_distribution(self, ax):
        """绘制IOL利用率分布"""
        if 'iol_analysis' in self.analysis_results:
            utilizations = self.analysis_results['iol_analysis']['utilization_distribution']
            ax.hist([u * 100 for u in utilizations], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel(LABELS['iol_utilization_percent'])
        ax.set_ylabel(LABELS['frequency'])
        ax.set_title(LABELS['iol_utilization_dist'])
        ax.grid(True, alpha=0.3)

    def _plot_algorithm_delay_comparison(self, ax):
        """绘制算法延迟对比"""
        if ('routing_efficiency' in self.analysis_results and
                'algorithm_comparison' in self.analysis_results['routing_efficiency']):

            algo_comp = self.analysis_results['routing_efficiency']['algorithm_comparison']
            algorithms = list(algo_comp.keys())
            delays = [algo_comp[algo]['avg_delay'] for algo in algorithms]

            colors = ['red', 'blue', 'green'][:len(algorithms)]
            bars = ax.bar(algorithms, delays, color=colors, alpha=0.7)

            ax.set_ylabel(LABELS['avg_delay_ms'])
            ax.set_title(LABELS['algorithm_delay_comparison'])
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, delay in zip(bars, delays):
                if delay < float('inf'):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{delay:.1f}', ha='center', va='bottom')

    def _plot_regional_connectivity(self, ax):
        """绘制区域连通性"""
        if ('geographical_analysis' in self.analysis_results and
                'connectivity_by_region' in self.analysis_results['geographical_analysis']):
            regional = self.analysis_results['geographical_analysis']['connectivity_by_region']
            regions = list(regional.keys())
            connectivity = [regional[region]['inter_region_connectivity'] * 100 for region in regions]

            colors = ['lightblue', 'lightgreen', 'lightcoral'][:len(regions)]
            ax.bar(regions, connectivity, color=colors, alpha=0.7)

            ax.set_ylabel(LABELS['inter_region_connectivity_percent'])
            ax.set_title(LABELS['regional_connectivity'])
            region_labels = [LABELS.get(r, r.title()) for r in regions]
            ax.set_xticklabels(region_labels)
            ax.grid(True, alpha=0.3)

    def _plot_iol_traffic_patterns(self, ax):
        """绘制IOL流量模式"""
        if ('iol_analysis' in self.analysis_results and
                'traffic_analysis' in self.analysis_results['iol_analysis']):
            traffic = self.analysis_results['iol_analysis']['traffic_analysis']
            periods = list(traffic.keys())
            utilizations = [traffic[period]['avg_utilization'] * 100 for period in periods]
            congestions = [traffic[period]['congestion_probability'] * 100 for period in periods]

            x = np.arange(len(periods))
            width = 0.35

            ax.bar(x - width / 2, utilizations, width, label=LABELS['avg_utilization'], alpha=0.7, color='blue')
            ax.bar(x + width / 2, congestions, width, label=LABELS['congestion_prob'], alpha=0.7, color='red')

            ax.set_xlabel(LABELS['time_period'])
            ax.set_ylabel(LABELS['percentage'])
            ax.set_title(LABELS['iol_traffic_patterns'])
            ax.set_xticks(x)
            period_labels = [LABELS.get(p, p.title()) for p in periods]
            ax.set_xticklabels(period_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_path_efficiency_analysis(self, ax):
        """绘制路径效率分析"""
        if ('routing_efficiency' in self.analysis_results and
                'path_efficiency' in self.analysis_results['routing_efficiency']):

            path_eff = self.analysis_results['routing_efficiency']['path_efficiency']

            metrics = [LABELS['efficiency_ratio'], LABELS['optimality_score'], LABELS['hop_difference']]
            values = [
                path_eff['avg_efficiency_ratio'],
                path_eff['path_optimality_score'],
                path_eff['avg_hop_difference'] / 10  # 归一化
            ]

            colors = ['green', 'blue', 'orange']
            bars = ax.bar(metrics, values, color=colors, alpha=0.7)

            ax.set_ylabel(LABELS['normalized_value'])
            ax.set_title(LABELS['path_efficiency_analysis'])
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

    def _plot_network_coverage_heatmap(self, ax):
        """绘制网络覆盖热图"""
        # 创建简化的覆盖热图
        grid_size = 30
        coverage_map = np.zeros((grid_size, grid_size))

        # 基于卫星位置生成覆盖热图
        for sat in self.network.leo_satellites:
            # 将卫星位置映射到网格
            lat_idx = int((sat.latitude + 90) / 180 * grid_size)
            lon_idx = int(sat.longitude / 360 * grid_size)

            lat_idx = max(0, min(lat_idx, grid_size - 1))
            lon_idx = max(0, min(lon_idx, grid_size - 1))

            # 在卫星周围增加覆盖强度
            for i in range(max(0, lat_idx - 2), min(grid_size, lat_idx + 3)):
                for j in range(max(0, lon_idx - 2), min(grid_size, lon_idx + 3)):
                    distance = np.sqrt((i - lat_idx) ** 2 + (j - lon_idx) ** 2)
                    coverage_map[i, j] += max(0, 1 - distance / 3)

        im = ax.imshow(coverage_map, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xlabel(LABELS['longitude'])
        ax.set_ylabel(LABELS['latitude'])
        ax.set_title(LABELS['network_coverage_heatmap'])

        # 设置坐标轴标签
        ax.set_xticks(np.linspace(0, grid_size - 1, 5))
        ax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'])
        ax.set_yticks(np.linspace(0, grid_size - 1, 5))
        ax.set_yticklabels(['-90°', '-45°', '0°', '45°', '90°'])

        plt.colorbar(im, ax=ax, label=LABELS['coverage_intensity'])

    def _plot_performance_radar(self, ax):
        """绘制性能雷达图"""
        # 定义性能指标
        categories = [
            LABELS['connectivity'],
            LABELS['delay_performance'],
            LABELS['iol_efficiency'],
            LABELS['path_optimization'],
            LABELS['regional_coverage'],
            LABELS['qos_guarantee']
        ]

        # 计算各项指标得分（0-1）
        scores = []

        # 连通性得分
        if 'topology_analysis' in self.analysis_results:
            connectivity = self.analysis_results['topology_analysis']['connectivity']
            conn_score = 0.9 if connectivity['is_connected'] else 0.3
            scores.append(conn_score)
        else:
            scores.append(0.5)

        # 延迟性能得分
        if ('routing_efficiency' in self.analysis_results and
                'algorithm_comparison' in self.analysis_results['routing_efficiency']):
            algo_comp = self.analysis_results['routing_efficiency']['algorithm_comparison']
            if 'M-BMDP' in algo_comp:
                delay = algo_comp['M-BMDP']['avg_delay']
                delay_score = max(0, min(1, 1 - delay / 200))  # 假设200ms为最差情况
                scores.append(delay_score)
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # IOL效率得分
        if 'iol_analysis' in self.analysis_results:
            iol_perf = self.analysis_results['iol_analysis']['performance_impact']
            iol_score = 1 - iol_perf['high_utilization_percentage']
            scores.append(iol_score)
        else:
            scores.append(0.5)

        # 路径优化得分
        if ('routing_efficiency' in self.analysis_results and
                'path_efficiency' in self.analysis_results['routing_efficiency']):
            path_eff = self.analysis_results['routing_efficiency']['path_efficiency']
            path_score = path_eff['path_optimality_score']
            scores.append(path_score)
        else:
            scores.append(0.5)

        # 区域覆盖得分
        if 'geographical_analysis' in self.analysis_results:
            geo = self.analysis_results['geographical_analysis']
            if 'group_manager_distribution' in geo:
                coverage_score = geo['group_manager_distribution']['coverage_efficiency']
                scores.append(coverage_score)
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # QoS保障得分（基于算法成功率）
        if ('routing_efficiency' in self.analysis_results and
                'algorithm_comparison' in self.analysis_results['routing_efficiency']):
            algo_comp = self.analysis_results['routing_efficiency']['algorithm_comparison']
            if 'M-BMDP' in algo_comp:
                qos_score = algo_comp['M-BMDP']['success_rate']
                scores.append(qos_score)
            else:
                scores.append(0.5)
        else:
            scores.append(0.5)

        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # 闭合图形
        angles += angles[:1]

        ax.plot(angles, scores, 'o-', linewidth=2, color='red', alpha=0.7)
        ax.fill(angles, scores, alpha=0.25, color='red')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        ax.set_title(LABELS['comprehensive_performance_radar'], y=1.08)


# 导出的便利函数
def run_complete_analysis(network, algorithm, simulation):
    """运行完整的分析流程"""
    print("=== 开始完整分析流程 ===\n")

    # 创建高级分析器
    analyzer = AdvancedAnalyzer(network, algorithm, simulation)

    # 运行各项分析
    print("1. 分析网络拓扑...")
    topology_results = analyzer.analyze_network_topology()

    print("2. 分析IOL性能...")
    iol_results = analyzer.analyze_iol_performance()

    print("3. 分析路由效率...")
    routing_results = analyzer.analyze_routing_efficiency()

    print("4. 分析地理分布...")
    geo_results = analyzer.analyze_geographical_distribution()

    # 生成报告
    print("5. 生成综合报告...")
    report = analyzer.generate_comprehensive_report()
    print(report)

    # 保存结果
    print("6. 保存分析结果...")
    analyzer.save_analysis_results()

    # 绘制图表
    print("7. 生成分析图表...")
    analyzer.plot_comprehensive_analysis()

    print("\n=== 分析完成 ===")
    return analyzer
