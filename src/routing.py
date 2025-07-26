from typing import Dict, List, Tuple, Optional, Set
import math
import heapq
from collections import deque, defaultdict

from satellites import LEOSatellite, MEOSatellite
from environment import get_leo_by_id, get_meo_by_id
from rl_agent import RLAgent


def calculate_geographic_distance(leo1: LEOSatellite, leo2: LEOSatellite) -> float:
    """计算两个LEO卫星之间的地理距离"""
    return math.sqrt(
        (leo1.latitude - leo2.latitude) ** 2 +
        (leo1.longitude - leo2.longitude) ** 2 +
        (leo1.altitude - leo2.altitude) ** 2
    )


def check_cluster_connectivity(src_leo_id: int, dst_leo_id: int,
                               cluster_leos: List[int], leos: Dict[int, LEOSatellite]) -> bool:
    """
    使用BFS检查cluster内两个LEO之间是否有连通性
    """
    if src_leo_id == dst_leo_id:
        return True

    if src_leo_id not in cluster_leos or dst_leo_id not in cluster_leos:
        return False

    visited = {src_leo_id}
    queue = deque([src_leo_id])
    cluster_set = set(cluster_leos)

    while queue:
        current = queue.popleft()

        if current == dst_leo_id:
            return True

        if current not in leos:
            continue

        current_leo = leos[current]

        for neighbor in current_leo.neighbors:
            if neighbor in cluster_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


def find_optimal_edge_nodes_enhanced(
        src_cluster_leos: List[int],
        dst_cluster_leos: List[int],
        leos: Dict[int, LEOSatellite],
        num_candidates: int = 3,
        load_weight: float = 0.3,
        distance_weight: float = 0.4,
        connectivity_weight: float = 0.3
) -> List[Tuple[int, int]]:
    """
    找到多个最优边缘节点对，考虑距离、负载和连通性
    """
    if not src_cluster_leos or not dst_cluster_leos:
        return []

    candidates = []
    src_set = set(src_cluster_leos)
    dst_set = set(dst_cluster_leos)

    # 计算归一化参数
    distances = []
    loads = []
    connectivities = []

    for src_leo_id in src_cluster_leos:
        if src_leo_id not in leos:
            continue
        for dst_leo_id in dst_cluster_leos:
            if dst_leo_id not in leos:
                continue

            src_leo = leos[src_leo_id]
            dst_leo = leos[dst_leo_id]

            distance = calculate_geographic_distance(src_leo, dst_leo)
            load = src_leo.load + dst_leo.load
            connectivity = len(src_leo.neighbors) + len(dst_leo.neighbors)

            distances.append(distance)
            loads.append(load)
            connectivities.append(connectivity)

    if not distances:
        return []

    max_distance = max(distances) if distances else 1
    max_load = max(loads) if loads else 1
    max_connectivity = max(connectivities) if connectivities else 1

    # 评估所有边缘节点对
    for src_leo_id in src_cluster_leos:
        if src_leo_id not in leos:
            continue
        src_leo = leos[src_leo_id]

        # 确保源节点有足够的连通性
        if len(src_leo.neighbors) < 2:
            continue

        for dst_leo_id in dst_cluster_leos:
            if dst_leo_id not in leos:
                continue
            dst_leo = leos[dst_leo_id]

            # 确保目标节点有足够的连通性
            if len(dst_leo.neighbors) < 2:
                continue

            # 计算各项指标
            distance = calculate_geographic_distance(src_leo, dst_leo)
            load = src_leo.load + dst_leo.load
            connectivity = len(src_leo.neighbors) + len(dst_leo.neighbors)

            # 归一化
            norm_distance = distance / max_distance if max_distance > 0 else 0
            norm_load = load / max_load if max_load > 0 else 0
            norm_connectivity = connectivity / max_connectivity if max_connectivity > 0 else 0

            # 计算综合得分（越小越好）
            score = (distance_weight * norm_distance +
                     load_weight * norm_load +
                     connectivity_weight * (1 - norm_connectivity))  # 连通性越高越好

            candidates.append((score, src_leo_id, dst_leo_id))

    # 按得分排序并返回前N个
    candidates.sort(key=lambda x: x[0])
    return [(src_id, dst_id) for _, src_id, dst_id in candidates[:num_candidates]]


def dijkstra_path(start_leo_id: int, end_leo_id: int,
                  available_leos: Set[int], leos: Dict[int, LEOSatellite],
                  max_hops: int = 20) -> Optional[List[int]]:
    """
    使用Dijkstra算法找到最短路径，考虑负载作为权重
    """
    if start_leo_id == end_leo_id:
        return [start_leo_id]

    if start_leo_id not in available_leos or end_leo_id not in available_leos:
        return None

    # 初始化距离和前驱节点
    distances = {leo_id: float('inf') for leo_id in available_leos}
    distances[start_leo_id] = 0
    previous = {}
    visited = set()

    # 优先队列：(distance, leo_id)
    pq = [(0, start_leo_id)]

    while pq:
        current_dist, current_leo = heapq.heappop(pq)

        if current_leo in visited:
            continue

        visited.add(current_leo)

        if current_leo == end_leo_id:
            # 重构路径
            path = []
            node = end_leo_id
            while node is not None:
                path.append(node)
                node = previous.get(node)
            return path[::-1]

        if current_leo not in leos:
            continue

        current_satellite = leos[current_leo]

        for neighbor in current_satellite.neighbors:
            if neighbor not in available_leos or neighbor in visited:
                continue

            # 计算边权重（考虑负载）
            neighbor_satellite = leos.get(neighbor)
            if neighbor_satellite is None:
                continue

            # 权重 = 1 + 负载因子
            weight = 1 + neighbor_satellite.load * 0.1
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_leo
                heapq.heappush(pq, (new_dist, neighbor))

        # 防止路径过长
        if len(visited) > max_hops:
            break

    return None  # 没有找到路径


def agent_generate_path_improved(
        start_leo_id: int,
        end_leo_id: int,
        available_leos: Set[int],
        leos: Dict[int, LEOSatellite],
        meos: Dict[int, MEOSatellite],
        agent: RLAgent,
        max_hops: int = 15
) -> Optional[List[int]]:
    """
    改进的基于agent的路径生成，避免死循环和提高效率
    """
    if start_leo_id == end_leo_id:
        return [start_leo_id]

    if start_leo_id not in available_leos or end_leo_id not in available_leos:
        return None

    path = [start_leo_id]
    current = start_leo_id
    visited = {start_leo_id}
    steps_without_progress = 0
    max_steps_without_progress = 5

    for hop in range(max_hops):
        if current == end_leo_id:
            break

        if current not in leos:
            return None

        current_leo = leos[current]

        # 获取可用的邻居节点
        available_neighbors = [
            n for n in current_leo.neighbors
            if n in available_leos and n not in visited
        ]

        if not available_neighbors:
            # 没有未访问的邻居，允许回访但限制次数
            available_neighbors = [
                n for n in current_leo.neighbors
                if n in available_leos
            ]
            if not available_neighbors:
                return None  # 真的没有邻居了

        # 使用agent选择下一个节点
        next_node = agent.choose_action(
            current, end_leo_id, available_neighbors, leos, meos
        )

        # 计算距离目标的进展
        if current in leos and end_leo_id in leos and next_node in leos:
            current_dist = calculate_geographic_distance(leos[current], leos[end_leo_id])
            next_dist = calculate_geographic_distance(leos[next_node], leos[end_leo_id])

            if next_dist >= current_dist:
                steps_without_progress += 1
            else:
                steps_without_progress = 0

        # 如果太久没有进展，使用贪心策略
        if steps_without_progress >= max_steps_without_progress:
            if end_leo_id in leos:
                target_leo = leos[end_leo_id]
                # 选择距离目标最近的邻居
                best_neighbor = min(
                    available_neighbors,
                    key=lambda n: calculate_geographic_distance(leos[n], target_leo)
                    if n in leos else float('inf')
                )
                next_node = best_neighbor
                steps_without_progress = 0

        path.append(next_node)

        # 计算奖励并更新agent
        if next_node == end_leo_id:
            reward = 10.0  # 到达目标
        else:
            # 基于距离目标的接近程度给予奖励
            if next_node in leos and end_leo_id in leos:
                distance_to_target = calculate_geographic_distance(leos[next_node], leos[end_leo_id])
                reward = -distance_to_target * 0.01  # 距离越近奖励越高
            else:
                reward = -1.0  # 惩罚无效移动

        # 更新agent
        next_neighbors = []
        if next_node in leos:
            next_neighbors = [
                n for n in leos[next_node].neighbors
                if n in available_leos
            ]

        agent.update(
            current, end_leo_id, next_node, reward,
            next_node, end_leo_id, next_neighbors, leos, meos,
            done=(next_node == end_leo_id)
        )

        visited.add(next_node)
        current = next_node

    return path if current == end_leo_id else None


def generate_k_paths_robust(
        start_leo_id: int,
        end_leo_id: int,
        available_leos: Set[int],
        leos: Dict[int, LEOSatellite],
        meos: Dict[int, MEOSatellite],
        agent: RLAgent,
        k: int = 3,
        max_hops: int = 15
) -> List[List[int]]:
    """
    生成k条不同的路径，使用多种策略确保路径多样性
    """
    paths = []

    # 方法1：使用agent生成路径
    for attempt in range(k * 2):
        if len(paths) >= k:
            break

        # 避免使用已有路径的中间节点
        avoid_nodes = set()
        for existing_path in paths:
            if len(existing_path) > 2:
                avoid_nodes.update(existing_path[1:-1])

        # 创建可用节点集合
        current_available = available_leos - avoid_nodes
        if start_leo_id not in current_available:
            current_available.add(start_leo_id)
        if end_leo_id not in current_available:
            current_available.add(end_leo_id)

        if len(current_available) < 2:
            current_available = available_leos  # 回退到全部可用节点

        path = agent_generate_path_improved(
            start_leo_id, end_leo_id, current_available, leos, meos, agent, max_hops
        )

        if path and path not in paths and len(path) <= max_hops + 1:
            paths.append(path)

    # 方法2：如果agent生成的路径不够，使用Dijkstra作为补充
    while len(paths) < k:
        # 避免已有路径
        avoid_nodes = set()
        for existing_path in paths:
            if len(existing_path) > 2:
                # 只避免部分中间节点，保持一定的重叠可能性
                middle_idx = len(existing_path) // 2
                avoid_nodes.add(existing_path[middle_idx])

        current_available = available_leos - avoid_nodes
        if start_leo_id not in current_available:
            current_available.add(start_leo_id)
        if end_leo_id not in current_available:
            current_available.add(end_leo_id)

        dijkstra_path_result = dijkstra_path(
            start_leo_id, end_leo_id, current_available, leos, max_hops
        )

        if dijkstra_path_result and dijkstra_path_result not in paths:
            paths.append(dijkstra_path_result)
        else:
            break  # 无法生成更多不同的路径

    return paths


def calculate_path_score_enhanced(path: List[int], leos: Dict[int, LEOSatellite],
                                  load_weight: float = 0.3, delay_weight: float = 0.4,
                                  reliability_weight: float = 0.3) -> float:
    """
    增强的路径评分函数，考虑负载、延迟和可靠性
    """
    if len(path) < 2:
        return float('inf')  # 无效路径

    total_score = 0.0

    # 延迟得分（路径长度）
    delay_score = len(path) - 1

    # 负载得分
    load_score = 0.0
    reliability_score = 0.0

    for leo_id in path:
        if leo_id not in leos:
            return float('inf')  # 路径包含不存在的节点

        leo = leos[leo_id]
        load_score += leo.load

        # 可靠性得分（基于连通性）
        reliability_score += 1.0 / (1.0 + len(leo.neighbors))  # 邻居越多越可靠

    # 归一化
    avg_load = load_score / len(path)
    avg_reliability = reliability_score / len(path)

    # 综合得分（越小越好）
    total_score = (delay_weight * delay_score +
                   load_weight * avg_load +
                   reliability_weight * avg_reliability)

    return total_score


def route_request_optimized(
        src_leo_id: int,
        dst_leo_id: int,
        leos: Dict[int, LEOSatellite],
        meos: Dict[int, MEOSatellite],
        agent: RLAgent,
        k_paths: int = 3,
        max_hops: int = 25
) -> Tuple[List[int], Dict[str, any]]:
    """
    优化的路由请求处理函数
    """
    # 输入验证
    if src_leo_id not in leos or dst_leo_id not in leos:
        return [], {'success': False, 'error': 'Source or destination not found'}

    if src_leo_id == dst_leo_id:
        return [src_leo_id], {'success': True, 'routing_strategy': 'direct', 'total_hops': 0}

    src_leo = leos[src_leo_id]
    dst_leo = leos[dst_leo_id]

    routing_stats = {
        'total_hops': 0,
        'routing_strategy': 'unknown',
        'success': False,
        'paths_evaluated': 0,
        'edge_nodes_used': []
    }

    # 检查是否为同集群路由
    if src_leo.meo_id == dst_leo.meo_id:
        routing_stats['routing_strategy'] = 'intra_cluster'

        # 获取集群内的LEO节点
        if src_leo.meo_id in meos:
            cluster_leos = set(meos[src_leo.meo_id].cluster_leos)

            # 检查集群内连通性
            if check_cluster_connectivity(src_leo_id, dst_leo_id, list(cluster_leos), leos):
                # 生成k条路径
                paths = generate_k_paths_robust(
                    src_leo_id, dst_leo_id, cluster_leos, leos, meos, agent, k_paths, max_hops
                )

                if paths:
                    # 选择最优路径
                    best_path = min(paths, key=lambda p: calculate_path_score_enhanced(p, leos))

                    routing_stats['total_hops'] = len(best_path) - 1
                    routing_stats['success'] = True
                    routing_stats['paths_evaluated'] = len(paths)

                    return best_path, routing_stats

        # 集群内路由失败，回退到全网络路由
        routing_stats['routing_strategy'] = 'intra_cluster_fallback'
        all_leos = set(leos.keys())
        paths = generate_k_paths_robust(
            src_leo_id, dst_leo_id, all_leos, leos, meos, agent, k_paths, max_hops
        )

        if paths:
            best_path = min(paths, key=lambda p: calculate_path_score_enhanced(p, leos))
            routing_stats['total_hops'] = len(best_path) - 1
            routing_stats['success'] = True
            routing_stats['paths_evaluated'] = len(paths)
            return best_path, routing_stats

    else:
        # 跨集群路由
        routing_stats['routing_strategy'] = 'inter_cluster'

        if src_leo.meo_id not in meos or dst_leo.meo_id not in meos:
            return [], {'success': False, 'error': 'MEO satellites not found'}

        src_meo = meos[src_leo.meo_id]
        dst_meo = meos[dst_leo.meo_id]

        # 寻找边缘节点对
        edge_candidates = find_optimal_edge_nodes_enhanced(
            src_meo.cluster_leos, dst_meo.cluster_leos, leos, num_candidates=k_paths
        )

        if not edge_candidates:
            # 边缘节点选择失败，使用全网络路由
            routing_stats['routing_strategy'] = 'inter_cluster_direct'
            all_leos = set(leos.keys())
            paths = generate_k_paths_robust(
                src_leo_id, dst_leo_id, all_leos, leos, meos, agent, k_paths, max_hops
            )

            if paths:
                best_path = min(paths, key=lambda p: calculate_path_score_enhanced(p, leos))
                routing_stats['total_hops'] = len(best_path) - 1
                routing_stats['success'] = True
                routing_stats['paths_evaluated'] = len(paths)
                return best_path, routing_stats

        else:
            # 尝试两段式路由
            best_complete_path = None
            best_score = float('inf')

            for edge_src, edge_dst in edge_candidates:
                routing_stats['edge_nodes_used'].append((edge_src, edge_dst))

                # 第一段：源到边缘源
                if src_leo_id == edge_src:
                    segment1 = [src_leo_id]
                else:
                    src_cluster = set(src_meo.cluster_leos)
                    segment1_paths = generate_k_paths_robust(
                        src_leo_id, edge_src, src_cluster, leos, meos, agent, 2, max_hops // 2
                    )
                    if not segment1_paths:
                        continue
                    segment1 = min(segment1_paths, key=lambda p: calculate_path_score_enhanced(p, leos))

                # 第二段：边缘目标到目标
                if edge_dst == dst_leo_id:
                    segment2 = [dst_leo_id]
                else:
                    dst_cluster = set(dst_meo.cluster_leos)
                    segment2_paths = generate_k_paths_robust(
                        edge_dst, dst_leo_id, dst_cluster, leos, meos, agent, 2, max_hops // 2
                    )
                    if not segment2_paths:
                        continue
                    segment2 = min(segment2_paths, key=lambda p: calculate_path_score_enhanced(p, leos))

                # 组合完整路径
                if edge_src == edge_dst:
                    complete_path = segment1 + segment2[1:]
                else:
                    complete_path = segment1 + [edge_dst] + segment2[1:]

                # 评估路径
                path_score = calculate_path_score_enhanced(complete_path, leos)
                if path_score < best_score:
                    best_score = path_score
                    best_complete_path = complete_path

            if best_complete_path:
                routing_stats['total_hops'] = len(best_complete_path) - 1
                routing_stats['success'] = True
                routing_stats['paths_evaluated'] = len(edge_candidates)
                return best_complete_path, routing_stats

    # 所有方法都失败
    routing_stats['success'] = False
    return [], routing_stats


# 向后兼容的接口
def route_request_with_intelligent_edge_selection(
        src_leo_id: int,
        dst_leo_id: int,
        leos: Dict[int, LEOSatellite],
        meos: Dict[int, MEOSatellite],
        agent: RLAgent,
        max_hops: int = 25,
        max_retries: int = 3,
        load_weight: float = 0.25,
        distance_weight: float = 0.35,
) -> Tuple[List[int], Dict[str, any]]:
    """
    向后兼容的路由接口
    """
    return route_request_optimized(src_leo_id, dst_leo_id, leos, meos, agent, max_retries, max_hops)