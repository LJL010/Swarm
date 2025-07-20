# ================================
# 文件: routing.py
# 路由算法实现
# ================================

import heapq
import random
from typing import List, Optional
from utils import *
from config import *
import networkx as nx



class QoSRoutingAlgorithm:
    """QoS路由算法实现"""

    def __init__(self, network, params: AlgorithmParameters = None):
        self.network = network
        self.params = params or AlgorithmParameters()
        self.routing_tables = {}
        self.link_states = {}

        # 初始化链路状态
        self._initialize_link_states()

    def _initialize_link_states(self):
        """初始化链路状态"""
        for u, v, data in self.network.graph.edges(data=True):
            link_id = f"{u}-{v}"
            self.link_states[link_id] = {
                'utilization': random.uniform(0.2, 0.6),
                'available_bandwidth': self.params.default_link_capacity * 0.5,
                'queue_length': random.uniform(0.1, 0.3),
                'last_update': 0
            }

    def modified_bmdp_algorithm(self, source: str, destination: str,
                                traffic_class: TrafficClass,
                                bandwidth_requirement: float) -> List[str]:
        """修改的带宽约束最小延迟路径(M-BMDP)算法"""

        # 第一步：计算主路径（最小延迟路径）
        primary_path = self._calculate_minimum_delay_path(source, destination)

        if not primary_path:
            return []

        # 第二步：检查带宽约束
        if self._check_bandwidth_constraint(primary_path, bandwidth_requirement):
            return primary_path

        # 第三步：计算备用路径
        alternative_paths = self._calculate_alternative_paths(
            source, destination, exclude_path=primary_path
        )

        # 第四步：选择满足QoS要求的最佳备用路径
        best_path = self._select_best_alternative_path(
            alternative_paths, traffic_class, bandwidth_requirement
        )

        return best_path if best_path else primary_path

    def _calculate_minimum_delay_path(self, source: str, destination: str) -> List[str]:
        """计算最小延迟路径"""
        try:
            path = nx.dijkstra_path(
                self.network.graph, source, destination,
                weight=self._delay_weight_function
            )
            return path
        except nx.NetworkXNoPath:
            return []

    def _delay_weight_function(self, u: str, v: str, edge_data: dict) -> float:
        """延迟权重函数"""
        distance = edge_data['distance']
        link_type = edge_data.get('link_type', 'unknown')
        link_id = f"{u}-{v}"

        # 传播延迟
        propagation_delay = PerformanceMetrics.calculate_propagation_delay(distance)

        # 传输延迟
        transmission_delay = PerformanceMetrics.calculate_transmission_delay(
            1000, self.params.default_link_capacity
        )

        # 队列延迟（基于当前链路状态）
        link_state = self.link_states.get(link_id, {})
        queue_length = link_state.get('queue_length', 0.1)
        queuing_delay = queue_length * 10  # 简化的队列延迟计算

        # IOL链路有额外的处理延迟
        processing_delay = 5.0 if link_type == Constants.LINK_TYPE_IOL else 1.0

        return propagation_delay + transmission_delay + queuing_delay + processing_delay

    def _check_bandwidth_constraint(self, path: List[str],
                                    bandwidth_requirement: float) -> bool:
        """检查路径是否满足带宽约束"""
        if not path or len(path) < 2:
            return False

        for i in range(len(path) - 1):
            link_id = f"{path[i]}-{path[i + 1]}"
            link_state = self.link_states.get(link_id, {})
            available_bandwidth = link_state.get(
                'available_bandwidth',
                self.params.default_link_capacity * 0.5
            )

            if available_bandwidth < bandwidth_requirement:
                return False

        return True

    def _calculate_alternative_paths(self, source: str, destination: str,
                                     exclude_path: List[str]) -> List[List[str]]:
        """计算备用路径"""
        alternative_paths = []

        # 方法1：移除主路径中的关键链路
        if len(exclude_path) >= 3:
            edges_to_remove = []
            for i in range(1, len(exclude_path) - 2):
                edges_to_remove.append((exclude_path[i], exclude_path[i + 1]))

            temp_graph = self.network.graph.copy()
            temp_graph.remove_edges_from(edges_to_remove[:len(edges_to_remove) // 2])

            try:
                alt_path = nx.dijkstra_path(
                    temp_graph, source, destination,
                    weight=self._delay_weight_function
                )
                if alt_path != exclude_path:
                    alternative_paths.append(alt_path)
            except nx.NetworkXNoPath:
                pass

        # 方法2：使用不同的路由策略（最短跳数）
        try:
            hop_path = nx.shortest_path(self.network.graph, source, destination)
            if hop_path != exclude_path:
                alternative_paths.append(hop_path)
        except nx.NetworkXNoPath:
            pass

        return alternative_paths

    def _select_best_alternative_path(self, paths: List[List[str]],
                                      traffic_class: TrafficClass,
                                      bandwidth_requirement: float) -> Optional[List[str]]:
        """选择最佳备用路径"""
        valid_paths = []

        for path in paths:
            if (is_valid_path(path, self.network) and
                    self._check_bandwidth_constraint(path, bandwidth_requirement)):
                score = self._calculate_path_score(path, traffic_class)
                valid_paths.append((path, score))

        if not valid_paths:
            return None

        # 选择得分最低（最好）的路径
        valid_paths.sort(key=lambda x: x[1])
        return valid_paths[0][0]

    def _calculate_path_score(self, path: List[str], traffic_class: TrafficClass) -> float:
        """计算路径得分"""
        if not path or len(path) < 2:
            return float('inf')

        total_delay = 0
        total_hops = len(path) - 1
        iol_count = 0

        for i in range(len(path) - 1):
            edge_data = self.network.graph[path[i]][path[i + 1]]
            total_delay += self._delay_weight_function(path[i], path[i + 1], edge_data)

            # 统计IOL链路数量
            if edge_data.get('link_type') == Constants.LINK_TYPE_IOL:
                iol_count += 1

        # 根据流量类别调整权重
        if traffic_class == TrafficClass.CLASS_A:
            # 延迟敏感流量：延迟权重高，跳数权重低
            score = total_delay * 2.0 + total_hops * 0.1 + iol_count * 1.5
        else:
            # 其他流量：平衡延迟和跳数
            score = total_delay * 1.0 + total_hops * 0.5 + iol_count * 1.0

        return score

    def improved_hop_limit_algorithm(self, source: str, destination: str,
                                     hop_limit: int, traffic_class: TrafficClass) -> List[str]:
        """基于跳数限制的改进算法 - 论文第3.4节"""

        # 首先尝试在跳数限制内找到路径
        hop_limited_path = self._find_hop_limited_path(source, destination, hop_limit)

        if hop_limited_path:
            return hop_limited_path

        # 如果找不到跳数限制内的路径
        if traffic_class == TrafficClass.CLASS_A:
            # 对于延迟敏感流量，优先考虑延迟要求
            basic_path = self._calculate_minimum_delay_path(source, destination)

            if basic_path and len(basic_path) - 1 <= hop_limit * 1.5:
                # 允许适度超出跳数限制
                return basic_path
            else:
                # 无法满足要求，返回空路径（丢弃包）
                return []
        else:
            # 对于其他流量，返回基本路径
            return self._calculate_minimum_delay_path(source, destination)

    def _find_hop_limited_path(self, source: str, destination: str,
                               hop_limit: int) -> Optional[List[str]]:
        """寻找跳数限制内的路径"""
        queue = [(0, 0, source, [source])]  # (delay, hops, current, path)
        visited = {}

        while queue:
            current_delay, hops, current_node, path = heapq.heappop(queue)

            if current_node == destination:
                return path

            if hops >= hop_limit:
                continue

            if current_node in visited:
                if visited[current_node] <= current_delay:
                    continue
            visited[current_node] = current_delay

            for neighbor in self.network.graph.neighbors(current_node):
                if neighbor not in path:
                    edge_data = self.network.graph[current_node][neighbor]
                    delay = self._delay_weight_function(current_node, neighbor, edge_data)
                    new_delay = current_delay + delay
                    new_path = path + [neighbor]

                    heapq.heappush(queue, (new_delay, hops + 1, neighbor, new_path))

        return None

    def special_area_routing(self, source: str, destination: str,
                             traffic_class: TrafficClass) -> List[str]:
        """反向裂隙区域特殊路由 - 论文第3.5节"""
        source_sat = self.network.satellites[source]
        dest_sat = self.network.satellites[destination]

        # 检查是否涉及反向裂隙区域
        source_in_rcz = GeographyUtils.is_in_reversed_crevice_zone(source_sat)
        dest_in_rcz = GeographyUtils.is_in_reversed_crevice_zone(dest_sat)

        if not source_in_rcz and not dest_in_rcz:
            # 不涉及特殊区域，使用标准算法
            return self.modified_bmdp_algorithm(source, destination, traffic_class, 0.5)

        # 涉及反向裂隙区域的特殊处理
        if source_in_rcz and dest_in_rcz:
            # 两端都在特殊区域，需要通过MEO层中继
            return self._route_through_meo_layer(source, destination, traffic_class)
        elif source_in_rcz or dest_in_rcz:
            # 一端在特殊区域，优化路径选择
            return self._route_with_rcz_optimization(source, destination, traffic_class)

        return []

    def _route_through_meo_layer(self, source: str, destination: str,
                                 traffic_class: TrafficClass) -> List[str]:
        """通过MEO层进行路由"""
        source_sat = self.network.satellites[source]
        dest_sat = self.network.satellites[destination]

        # 寻找合适的MEO中继卫星
        best_meo_relay = None
        best_total_distance = float('inf')

        for meo_sat in self.network.meo_satellites:
            # 计算通过该MEO卫星的总距离
            dist1 = source_sat.distance_to(meo_sat, self.network.params.earth_radius)
            dist2 = meo_sat.distance_to(dest_sat, self.network.params.earth_radius)
            total_dist = dist1 + dist2

            if total_dist < best_total_distance:
                best_total_distance = total_dist
                best_meo_relay = meo_sat

        if best_meo_relay:
            try:
                # 构建通过MEO中继的路径
                path1 = nx.shortest_path(self.network.graph, source, best_meo_relay.id)
                path2 = nx.shortest_path(self.network.graph, best_meo_relay.id, destination)

                # 合并路径
                combined_path = path1 + path2[1:]
                return combined_path
            except nx.NetworkXNoPath:
                pass

        # 如果失败，回退到标准算法
        return self._calculate_minimum_delay_path(source, destination)

    def _route_with_rcz_optimization(self, source: str, destination: str,
                                     traffic_class: TrafficClass) -> List[str]:
        """带RCZ优化的路由"""
        # 首先尝试标准路由
        standard_path = self.modified_bmdp_algorithm(source, destination, traffic_class, 0.5)

        if standard_path:
            path_delay = self._calculate_path_delay(standard_path)

            # 如果延迟可接受，直接返回
            if path_delay <= self.params.delay_threshold * 1.2:
                return standard_path

        # 否则尝试通过最近的非RCZ卫星中继
        return self._find_non_rcz_relay_path(source, destination, traffic_class)

    def _find_non_rcz_relay_path(self, source: str, destination: str,
                                 traffic_class: TrafficClass) -> List[str]:
        """寻找通过非RCZ卫星的中继路径"""
        # 寻找非RCZ的LEO卫星作为中继
        non_rcz_satellites = [
            sat for sat in self.network.leo_satellites
            if not GeographyUtils.is_in_reversed_crevice_zone(sat)
        ]

        best_path = None
        best_score = float('inf')

        for relay_sat in non_rcz_satellites[:10]:  # 限制搜索范围
            try:
                path1 = nx.shortest_path(self.network.graph, source, relay_sat.id)
                path2 = nx.shortest_path(self.network.graph, relay_sat.id, destination)

                combined_path = path1 + path2[1:]
                score = self._calculate_path_score(combined_path, traffic_class)

                if score < best_score:
                    best_score = score
                    best_path = combined_path
            except nx.NetworkXNoPath:
                continue

        return best_path if best_path else []

    def _calculate_path_delay(self, path: List[str]) -> float:
        """计算路径总延迟"""
        if len(path) < 2:
            return float('inf')

        total_delay = 0
        for i in range(len(path) - 1):
            edge_data = self.network.graph[path[i]][path[i + 1]]
            total_delay += self._delay_weight_function(path[i], path[i + 1], edge_data)

        return total_delay


# 对比算法实现
class BaselineAlgorithms:
    """基线对比算法"""

    def __init__(self, network):
        self.network = network

    def mdsp_algorithm(self, source: str, destination: str) -> List[str]:
        """多层Dijkstra最短路径算法"""
        try:
            return nx.shortest_path(self.network.graph, source, destination, weight='distance')
        except nx.NetworkXNoPath:
            return []

    def bmdp_algorithm(self, source: str, destination: str,
                       bandwidth_requirement: float) -> List[str]:
        """传统BMDP算法（简化版本）"""
        try:
            # 移除不满足带宽要求的链路
            filtered_graph = self.network.graph.copy()
            edges_to_remove = []

            for u, v, data in filtered_graph.edges(data=True):
                # 模拟带宽检查
                available_bw = 100 * (1 - random.uniform(0.3, 0.8))
                if available_bw < bandwidth_requirement:
                    edges_to_remove.append((u, v))

            filtered_graph.remove_edges_from(edges_to_remove)

            return nx.dijkstra_path(filtered_graph, source, destination, weight='distance')
        except nx.NetworkXNoPath:
            return []