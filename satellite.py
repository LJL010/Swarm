# ================================
# 文件: satellite.py
# 卫星类定义
# ================================

import math
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class Satellite:
    """卫星节点"""
    id: str
    layer: str  # "LEO" or "MEO"
    orbit: int
    position_in_orbit: int
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    logical_area: Tuple[int, int] = (0, 0)
    is_group_manager: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.layer}_{self.orbit}_{self.position_in_orbit}"

    def get_3d_position(self, earth_radius: float) -> Tuple[float, float, float]:
        """获取卫星的3D坐标位置"""
        R = earth_radius + self.altitude
        lat_rad = math.radians(self.latitude)
        lon_rad = math.radians(self.longitude)

        x = R * math.cos(lat_rad) * math.cos(lon_rad)
        y = R * math.cos(lat_rad) * math.sin(lon_rad)
        z = R * math.sin(lat_rad)

        return (x, y, z)

    def distance_to(self, other: 'Satellite', earth_radius: float) -> float:
        """计算到另一颗卫星的距离"""
        x1, y1, z1 = self.get_3d_position(earth_radius)
        x2, y2, z2 = other.get_3d_position(earth_radius)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def is_same_orbit(self, other: 'Satellite') -> bool:
        return self.layer == other.layer and self.orbit == other.orbit

    def is_adjacent_orbit(self, other: 'Satellite') -> bool:
        if self.layer != other.layer:
            return False
        return abs(self.orbit - other.orbit) == 1

    def __str__(self) -> str:
        return f"{self.id}({self.latitude:.1f}°, {self.longitude:.1f}°)"


class SatelliteGroup:
    """卫星组（用于虚拟节点算法）"""

    def __init__(self, manager: Satellite, logical_area: Tuple[int, int]):
        self.manager = manager
        self.logical_area = logical_area
        self.members: List[Satellite] = [manager]
        self.area_center = self._calculate_area_center()

    def _calculate_area_center(self) -> Tuple[float, float]:
        p, q = self.logical_area
        center_lat = (p - 3) * 30
        center_lon = q * 30
        return (center_lat, center_lon)

    def add_member(self, satellite: Satellite):
        """添加组成员"""
        if satellite not in self.members:
            self.members.append(satellite)
            satellite.logical_area = self.logical_area

    def get_member_count(self) -> int:
        """获取组成员数量"""
        return len(self.members)

    def remove_member(self, satellite: Satellite):
        """移除组成员"""
        if satellite in self.members and satellite != self.manager:
            self.members.remove(satellite)

    def is_member(self, satellite: Satellite) -> bool:
        """检查是否为组成员"""
        return satellite in self.members

    def get_member_ids(self) -> List[str]:
        """获取所有成员ID列表"""
        return [member.id for member in self.members]

    def __str__(self) -> str:
        return f"Group_{self.logical_area}(Manager: {self.manager.id}, Members: {len(self.members)})"


class ConstellationManager:
    """星座管理器"""

    def __init__(self):
        self.satellites: List[Satellite] = []
        self.leo_satellites: List[Satellite] = []
        self.meo_satellites: List[Satellite] = []
        self.groups: List[SatelliteGroup] = []

    def add_satellite(self, satellite: Satellite):
        """添加卫星到星座"""
        self.satellites.append(satellite)
        if satellite.layer == "LEO":
            self.leo_satellites.append(satellite)
        elif satellite.layer == "MEO":
            self.meo_satellites.append(satellite)

    def get_satellite_by_id(self, sat_id: str) -> Optional[Satellite]:
        """根据ID获取卫星"""
        for sat in self.satellites:
            if sat.id == sat_id:
                return sat
        return None

    def find_nearest_satellite(self, target_lat: float, target_lon: float,
                               layer: Optional[str] = None) -> Optional[Satellite]:
        """找到最近的卫星"""
        candidates = self.satellites
        if layer:
            candidates = self.leo_satellites if layer == "LEO" else self.meo_satellites

        if not candidates:
            return None

        min_distance = float('inf')
        nearest_sat = None

        for sat in candidates:
            lat_diff = abs(sat.latitude - target_lat)
            lon_diff = abs(sat.longitude - target_lon)
            if lon_diff > 180:
                lon_diff = 360 - lon_diff

            distance = math.sqrt(lat_diff ** 2 + lon_diff ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_sat = sat

        return nearest_sat

    def get_group_managers(self) -> List[Satellite]:
        """获取所有组管理器"""
        return [sat for sat in self.satellites if sat.is_group_manager]

    def get_group_by_area(self, logical_area: Tuple[int, int]) -> Optional[SatelliteGroup]:
        """根据逻辑区域获取组"""
        for group in self.groups:
            if group.logical_area == logical_area:
                return group
        return None

    def get_total_groups(self) -> int:
        """获取组总数"""
        return len(self.groups)

    def get_constellation_stats(self) -> dict:
        """获取星座统计信息"""
        return {
            'total_satellites': len(self.satellites),
            'leo_satellites': len(self.leo_satellites),
            'meo_satellites': len(self.meo_satellites),
            'total_groups': len(self.groups),
            'group_managers': len(self.get_group_managers())
        }

    def __str__(self) -> str:
        return (f"Constellation(Satellites: {len(self.satellites)}, "
                f"LEO: {len(self.leo_satellites)}, MEO: {len(self.meo_satellites)}, "
                f"Groups: {len(self.groups)})")