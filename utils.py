# ================================
# 文件: utils.py
# 工具函数
# ================================
import math
from typing import List
from satellite import Satellite

import numpy as np


class DistanceCalculator:
    """距离计算器 - 实现论文中的距离计算公式"""

    def __init__(self, earth_radius: float):
        self.earth_radius = earth_radius

    def calculate_leo_distance(self, sat1: Satellite, sat2: Satellite) -> float:
        """计算LEO层内卫星距离 - 按照论文公式(1)(2)(3)"""
        if sat1.layer != "LEO" or sat2.layer != "LEO":
            raise ValueError("Both satellites must be in LEO layer")

        if sat1.is_same_orbit(sat2):
            return self._calculate_intra_orbit_distance(sat1, sat2)
        elif sat1.is_adjacent_orbit(sat2):
            return self._calculate_inter_orbit_distance(sat1, sat2)
        else:
            return self.calculate_3d_distance(sat1, sat2)

    def _calculate_intra_orbit_distance(self, sat1: Satellite, sat2: Satellite) -> float:
        """轨道内距离计算 - 论文公式(1)"""
        angle_diff = abs(sat1.longitude - sat2.longitude)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        R = self.earth_radius + sat1.altitude
        distance = math.sqrt(2 * R * R * (1 - math.cos(math.radians(angle_diff))))
        return distance

    def _calculate_inter_orbit_distance(self, sat1: Satellite, sat2: Satellite) -> float:
        """轨道间距离计算 - 论文公式(2)(3)"""
        lat = min(abs(sat1.latitude), abs(sat2.latitude))
        lat_rad = math.radians(lat)

        alpha = math.sqrt(2 * self.earth_radius *
                          (1 - math.cos(math.radians(360 / (2 * 6)))))
        distance = alpha * math.cos(lat_rad)
        return distance

    def calculate_3d_distance(self, sat1: Satellite, sat2: Satellite) -> float:
        """计算3D直线距离"""
        x1, y1, z1 = sat1.get_3d_position(self.earth_radius)
        x2, y2, z2 = sat2.get_3d_position(self.earth_radius)

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance


class PerformanceMetrics:
    """性能指标计算"""

    @staticmethod
    def calculate_propagation_delay(distance: float, speed: float = 3e5) -> float:
        """计算传播延迟 (km/s -> ms)"""
        return (distance / speed) * 1000

    @staticmethod
    def calculate_transmission_delay(packet_size: int, bandwidth: float) -> float:
        """计算传输延迟 (bits, Mbps -> ms)"""
        return (packet_size / (bandwidth * 1e6)) * 1000

    @staticmethod
    def calculate_end_to_end_delay(path_distances: List[float],
                                   packet_size: int = 1000,
                                   bandwidth: float = 100.0,
                                   processing_delay: float = 1.0) -> float:
        """计算端到端延迟"""
        if not path_distances:
            return float('inf')

        total_delay = 0.0
        for distance in path_distances:
            prop_delay = PerformanceMetrics.calculate_propagation_delay(distance)
            trans_delay = PerformanceMetrics.calculate_transmission_delay(packet_size, bandwidth)
            total_delay += prop_delay + trans_delay + processing_delay

        return total_delay


class GeographyUtils:
    """地理工具类"""

    @staticmethod
    def is_in_reversed_crevice_zone(satellite: Satellite, threshold: float = 75.0) -> bool:
        """判断卫星是否在反向裂隙区域（极地区域）"""
        return abs(satellite.latitude) > threshold


def calculate_path_delay(path: List[str], network, packet_size: int = 1000) -> float:
    """计算路径延迟"""
    if len(path) < 2:
        return float('inf')

    distances = []
    for i in range(len(path) - 1):
        edge_data = network.graph[path[i]][path[i + 1]]
        distances.append(edge_data['distance'])

    return PerformanceMetrics.calculate_end_to_end_delay(distances, packet_size)


def is_valid_path(path: List[str], network) -> bool:
    """验证路径有效性"""
    if len(path) < 2:
        return False

    for i in range(len(path) - 1):
        if not network.has_link(path[i], path[i + 1]):
            return False

    return True