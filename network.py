"""
网络拓扑生成模块
负责创建MEO/LEO双层卫星网络拓扑
修复版本 - 支持灵活的网络规模参数
"""

import math
import networkx as nx
from typing import Dict, List, Tuple, Optional

from config import NetworkParameters, Constants
from satellite import Satellite, SatelliteGroup, ConstellationManager
from utils import DistanceCalculator


class SatelliteNetwork:
    """卫星网络拓扑"""

    def __init__(self, params: NetworkParameters):
        self.params = params
        self.constellation = ConstellationManager()
        self.graph = nx.Graph()
        self.logical_areas = {}  # 逻辑区域映射 (p,q) -> satellite_id
        self.groups = {}  # 组管理器映射 (p,q) -> SatelliteGroup
        self.distance_calc = DistanceCalculator(params.earth_radius)

        # 生成网络
        self._generate_constellation()
        self._establish_links()
        self._apply_virtual_node_algorithm()

    @property
    def satellites(self) -> Dict[str, Satellite]:
        return {sat.id: sat for sat in self.constellation.satellites}

    @property
    def leo_satellites(self) -> List[Satellite]:
        return self.constellation.leo_satellites

    @property
    def meo_satellites(self) -> List[Satellite]:
        return self.constellation.meo_satellites

    def _generate_constellation(self):
        """生成卫星星座"""
        self._generate_leo_constellation()
        self._generate_meo_constellation()
        print(f"星座生成完成: {self.constellation}")

    def _generate_leo_constellation(self):
        """生成LEO星座"""
        for orbit in range(self.params.leo_orbits):
            for pos in range(self.params.leo_satellites_per_orbit):
                satellite = Satellite(
                    id=f"LEO_{orbit}_{pos}",
                    layer="LEO",
                    orbit=orbit,
                    position_in_orbit=pos,
                    altitude=self.params.leo_altitude
                )
                self._calculate_leo_position(satellite)
                self.constellation.add_satellite(satellite)
                self.graph.add_node(satellite.id, satellite=satellite)

    def _generate_meo_constellation(self):
        """生成MEO星座"""
        for orbit in range(self.params.meo_orbits):
            inclination = (self.params.meo_inclination_1 if orbit == 0
                           else self.params.meo_inclination_2)

            for pos in range(self.params.meo_satellites_per_orbit):
                satellite = Satellite(
                    id=f"MEO_{orbit}_{pos}",
                    layer="MEO",
                    orbit=orbit,
                    position_in_orbit=pos,
                    altitude=self.params.meo_altitude
                )
                self._calculate_meo_position(satellite, inclination)
                self.constellation.add_satellite(satellite)
                self.graph.add_node(satellite.id, satellite=satellite)

    def _calculate_leo_position(self, satellite: Satellite):
        """计算LEO卫星位置"""
        angle_per_satellite = 360.0 / self.params.leo_satellites_per_orbit
        longitude = satellite.position_in_orbit * angle_per_satellite
        latitude = self.params.leo_inclination * math.sin(math.radians(longitude))

        satellite.longitude = longitude
        satellite.latitude = latitude

    def _calculate_meo_position(self, satellite: Satellite, inclination: float):
        """计算MEO卫星位置"""
        angle_per_satellite = 360.0 / self.params.meo_satellites_per_orbit
        longitude = satellite.position_in_orbit * angle_per_satellite
        latitude = inclination * math.sin(math.radians(longitude))

        satellite.longitude = longitude
        satellite.latitude = latitude

    def _establish_links(self):
        """建立卫星间链路(ISL)"""
        print("建立卫星间链路...")

        leo_links = self._establish_leo_links()
        print(f"LEO层内链路: {leo_links}条")

        meo_links = self._establish_meo_links()
        print(f"MEO层内链路: {meo_links}条")

        iol_links = self._establish_inter_orbit_links()
        print(f"层间链路(IOL): {iol_links}条")

        total_links = self.graph.number_of_edges()
        print(f"总链路数: {total_links}")

    def _establish_leo_links(self) -> int:
        """建立LEO层内链路"""
        link_count = 0

        for orbit in range(self.params.leo_orbits):
            # 轨道内链路（环形连接）
            for pos in range(self.params.leo_satellites_per_orbit):
                current_sat = f"LEO_{orbit}_{pos}"
                next_pos = (pos + 1) % self.params.leo_satellites_per_orbit
                next_sat = f"LEO_{orbit}_{next_pos}"

                if self._satellite_exists(current_sat) and self._satellite_exists(next_sat):
                    self._add_link(current_sat, next_sat, Constants.LINK_TYPE_LEO_INTRA)
                    link_count += 1

            # 轨道间链路（相邻轨道）
            if orbit < self.params.leo_orbits - 1:
                for pos in range(self.params.leo_satellites_per_orbit):
                    current_sat = f"LEO_{orbit}_{pos}"
                    neighbor_sat = f"LEO_{orbit + 1}_{pos}"

                    if self._satellite_exists(current_sat) and self._satellite_exists(neighbor_sat):
                        self._add_link(current_sat, neighbor_sat, Constants.LINK_TYPE_LEO_INTER)
                        link_count += 1

        return link_count

    def _establish_meo_links(self) -> int:
        """建立MEO层内链路"""
        link_count = 0

        # 轨道内链路
        for orbit in range(self.params.meo_orbits):
            for pos in range(self.params.meo_satellites_per_orbit):
                current_sat = f"MEO_{orbit}_{pos}"
                next_pos = (pos + 1) % self.params.meo_satellites_per_orbit
                next_sat = f"MEO_{orbit}_{next_pos}"

                if self._satellite_exists(current_sat) and self._satellite_exists(next_sat):
                    self._add_link(current_sat, next_sat, Constants.LINK_TYPE_MEO_INTRA)
                    link_count += 1

        # MEO轨道间链路（只有当存在多个轨道时才建立）
        if self.params.meo_orbits >= 2:
            for pos in range(min(self.params.meo_satellites_per_orbit,
                                 len([s for s in self.meo_satellites if s.orbit == 0]),
                                 len([s for s in self.meo_satellites if s.orbit == 1]))):
                sat1 = f"MEO_0_{pos}"
                sat2 = f"MEO_1_{pos}"

                if self._satellite_exists(sat1) and self._satellite_exists(sat2):
                    self._add_link(sat1, sat2, Constants.LINK_TYPE_MEO_INTER)
                    link_count += 1

        return link_count

    def _establish_inter_orbit_links(self) -> int:
        """建立LEO-MEO层间链路(IOL)"""
        link_count = 0

        if not self.meo_satellites:
            print("警告: 没有MEO卫星，无法建立IOL链路")
            return 0

        for leo_sat in self.leo_satellites:
            closest_meo = self._find_closest_meo_satellite(leo_sat)
            if closest_meo:
                self._add_link(leo_sat.id, closest_meo.id, Constants.LINK_TYPE_IOL)
                link_count += 1

        return link_count

    def _satellite_exists(self, sat_id: str) -> bool:
        """检查卫星是否存在"""
        return sat_id in self.satellites

    def _add_link(self, sat1_id: str, sat2_id: str, link_type: str):
        """添加链路到图中"""
        # 检查卫星是否存在
        if not self._satellite_exists(sat1_id) or not self._satellite_exists(sat2_id):
            print(f"警告: 尝试在不存在的卫星间建立链路: {sat1_id} <-> {sat2_id}")
            return

        sat1 = self.satellites[sat1_id]
        sat2 = self.satellites[sat2_id]

        # 计算距离
        try:
            if link_type in [Constants.LINK_TYPE_LEO_INTRA, Constants.LINK_TYPE_LEO_INTER]:
                distance = self.distance_calc.calculate_leo_distance(sat1, sat2)
            else:
                distance = self.distance_calc.calculate_3d_distance(sat1, sat2)
        except Exception as e:
            print(f"警告: 计算距离失败 {sat1_id} <-> {sat2_id}: {e}")
            distance = 1000.0  # 使用默认距离

        self.graph.add_edge(sat1_id, sat2_id,
                            weight=distance,
                            link_type=link_type,
                            distance=distance)

    def _find_closest_meo_satellite(self, leo_sat: Satellite) -> Optional[Satellite]:
        """为LEO卫星找到最近的MEO卫星"""
        if not self.meo_satellites:
            return None

        min_distance = float('inf')
        closest_meo = None

        for meo_sat in self.meo_satellites:
            try:
                distance = self.distance_calc.calculate_3d_distance(leo_sat, meo_sat)
                if distance < min_distance:
                    min_distance = distance
                    closest_meo = meo_sat
            except Exception as e:
                print(f"警告: 计算LEO-MEO距离失败: {e}")
                continue

        return closest_meo

    def _apply_virtual_node_algorithm(self):
        """应用虚拟节点算法 - 按照论文第3节描述"""
        print("应用虚拟节点算法...")

        if not self.leo_satellites:
            print("警告: 没有LEO卫星，无法应用虚拟节点算法")
            return

        # 根据LEO卫星数量调整逻辑区域大小
        total_leo_sats = len(self.leo_satellites)
        if total_leo_sats < 12:
            # 小规模网络，减少逻辑区域数量
            grid_rows = min(2, max(1, total_leo_sats // 6))
            grid_cols = min(6, max(1, total_leo_sats // grid_rows))
        else:
            # 标准网络
            grid_rows, grid_cols = Constants.GRID_ROWS, Constants.GRID_COLS

        group_count = 0

        for p in range(grid_rows):
            for q in range(grid_cols):
                # 计算逻辑区域中心
                center_lat = (p - grid_rows / 2) * (180 / grid_rows)
                center_lon = q * (360 / grid_cols)

                # 为每个逻辑区域选择最近的LEO卫星作为组管理器
                closest_leo = self.constellation.find_nearest_satellite(
                    center_lat, center_lon, "LEO"
                )

                if closest_leo and not closest_leo.is_group_manager:
                    closest_leo.logical_area = (p, q)
                    closest_leo.is_group_manager = True

                    # 创建卫星组
                    group = SatelliteGroup(closest_leo, (p, q))
                    self.groups[(p, q)] = group
                    self.logical_areas[(p, q)] = closest_leo.id
                    self.constellation.groups.append(group)
                    group_count += 1

        print(f"虚拟节点算法完成，创建了{group_count}个组")

    def get_iol_links(self) -> List[Tuple[str, str]]:
        """获取所有IOL链路"""
        iol_links = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('link_type') == Constants.LINK_TYPE_IOL:
                iol_links.append((u, v))
        return iol_links

    def get_path_distance(self, path: List[str]) -> float:
        """计算路径总距离"""
        if len(path) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i + 1]):
                edge_data = self.graph[path[i]][path[i + 1]]
                total_distance += edge_data['distance']
            else:
                # 如果边不存在，返回无穷大
                return float('inf')

        return total_distance

    def has_link(self, sat1_id: str, sat2_id: str) -> bool:
        """检查两颗卫星间是否有直接链路"""
        return self.graph.has_edge(sat1_id, sat2_id)

    def get_link_info(self, sat1_id: str, sat2_id: str) -> Optional[Dict]:
        """获取链路信息"""
        if self.has_link(sat1_id, sat2_id):
            return self.graph[sat1_id][sat2_id]
        return None

    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        stats = {
            'total_satellites': len(self.satellites),
            'leo_satellites': len(self.leo_satellites),
            'meo_satellites': len(self.meo_satellites),
            'total_edges': self.graph.number_of_edges(),
            'leo_orbits': self.params.leo_orbits,
            'meo_orbits': self.params.meo_orbits,
            'logical_areas': len(self.logical_areas),
            'group_managers': len([s for s in self.constellation.satellites if s.is_group_manager])
        }

        if self.graph.number_of_nodes() > 0:
            stats.update({
                'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_connected(self.graph)
            })

        # 链路类型统计
        link_types = {}
        for _, _, data in self.graph.edges(data=True):
            link_type = data.get('link_type', 'unknown')
            link_types[link_type] = link_types.get(link_type, 0) + 1

        stats['link_types'] = link_types

        return stats

    def visualize_coverage(self) -> Dict:
        """可视化覆盖信息（返回数据用于绘图）"""
        coverage_data = {
            'logical_areas': [],
            'group_managers': [],
            'coverage_stats': {}
        }

        for (p, q), group in self.groups.items():
            # 根据实际网格大小计算中心
            grid_rows = max(1, len(set(area[0] for area in self.groups.keys())))
            grid_cols = max(1, len(set(area[1] for area in self.groups.keys())))

            center_lat = (p - grid_rows / 2) * (180 / grid_rows)
            center_lon = q * (360 / grid_cols)

            coverage_data['logical_areas'].append({
                'area': (p, q),
                'center': (center_lat, center_lon),
                'manager': group.manager.id,
                'member_count': group.get_member_count()
            })

            coverage_data['group_managers'].append({
                'id': group.manager.id,
                'position': (group.manager.latitude, group.manager.longitude),
                'area': (p, q)
            })

        # 覆盖统计
        member_counts = [group.get_member_count() for group in self.groups.values()]
        if member_counts:
            coverage_data['coverage_stats'] = {
                'total_groups': len(self.groups),
                'avg_members_per_group': sum(member_counts) / len(member_counts),
                'min_members': min(member_counts),
                'max_members': max(member_counts)
            }
        else:
            coverage_data['coverage_stats'] = {
                'total_groups': 0,
                'avg_members_per_group': 0,
                'min_members': 0,
                'max_members': 0
            }

        return coverage_data