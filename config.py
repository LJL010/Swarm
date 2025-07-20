"""
配置参数文件
包含网络参数、算法参数和仿真参数的配置
"""
# ================================
# 文件: config.py
# 配置参数文件
# ================================

from dataclasses import dataclass
from enum import Enum


class TrafficClass(Enum):
    """流量分类"""
    CLASS_A = "delay_sensitive"  # 延迟敏感流量 (如VoIP)
    CLASS_B = "other"  # 其他类型流量


@dataclass
class NetworkParameters:
    """网络参数 - 严格按照论文Table 1"""
    # LEO层参数
    leo_altitude: float = 780  # km
    leo_orbits: int = 6
    leo_satellites_per_orbit: int = 12
    leo_max_isls: int = 5
    leo_inclination: float = 86.4  # degrees

    # MEO层参数
    meo_altitude: float = 10355  # km
    meo_orbits: int = 2
    meo_satellites_per_orbit: int = 5
    meo_max_isls: int = 3
    meo_inclination_1: float = 45  # degrees
    meo_inclination_2: float = 135  # degrees

    # 地球参数
    earth_radius: float = 6371  # km


@dataclass
class AlgorithmParameters:
    """算法参数配置"""
    # 路由算法参数
    packet_loss_threshold: float = 0.1  # 包丢失率阈值
    delay_threshold: float = 100  # 延迟阈值(ms)
    hop_limit_default: int = 5  # 默认跳数限制

    # 带宽参数
    default_link_capacity: float = 100  # Mbps
    bandwidth_utilization_threshold: float = 0.8  # 带宽利用率阈值

    # 队列参数
    queue_length_threshold: float = 0.7  # 队列长度阈值


@dataclass
class SimulationParameters:
    """仿真参数配置"""
    # 流量参数
    traffic_rate_min: int = 540  # kbps
    traffic_rate_max: int = 620  # kbps
    traffic_rate_step: int = 10  # kbps
    packet_size: int = 1000  # bits

    # 仿真参数
    default_iterations: int = 100
    random_seed: int = 42

    # 性能评估参数
    congestion_threshold: float = 0.8
    high_utilization_threshold: float = 0.8


# 常量定义
class Constants:
    """系统常量"""
    LIGHT_SPEED = 3e5  # km/s 光速
    GRID_ROWS = 6  # 逻辑区域行数
    GRID_COLS = 12  # 逻辑区域列数
    AREA_SIZE = 30  # 每个逻辑区域大小(度)

    # 链路类型
    LINK_TYPE_LEO_INTRA = "LEO_intra"  # LEO轨道内
    LINK_TYPE_LEO_INTER = "LEO_inter"  # LEO轨道间
    LINK_TYPE_MEO_INTRA = "MEO_intra"  # MEO轨道内
    LINK_TYPE_MEO_INTER = "MEO_inter"  # MEO轨道间
    LINK_TYPE_IOL = "IOL"  # 层间链路


# 默认配置实例
DEFAULT_NETWORK_PARAMS = NetworkParameters()
DEFAULT_ALGORITHM_PARAMS = AlgorithmParameters()
DEFAULT_SIMULATION_PARAMS = SimulationParameters()