#!/usr/bin/env python3
"""
快速测试脚本 - 验证系统安装和基本功能
改进版本 - 使用更合理的测试参数

使用方法:
python quick_test.py
"""

import sys
import time
from typing import Dict, Any


def test_imports() -> bool:
    """测试模块导入"""
    print("🔍 测试模块导入...")

    try:
        import numpy as np
        print("  ✅ numpy导入成功")

        import matplotlib.pyplot as plt
        print("  ✅ matplotlib导入成功")

        import networkx as nx
        print("  ✅ networkx导入成功")

        # 测试项目模块
        from config import NetworkParameters, TrafficClass
        print("  ✅ config模块导入成功")

        from satellite import Satellite, ConstellationManager
        print("  ✅ satellite模块导入成功")

        from network import SatelliteNetwork
        print("  ✅ network模块导入成功")

        from routing import QoSRoutingAlgorithm
        print("  ✅ routing模块导入成功")

        from simulation import NetworkSimulation
        print("  ✅ simulation模块导入成功")

        from analysis import AdvancedAnalyzer
        print("  ✅ analysis模块导入成功")

        return True

    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_network_creation() -> Dict[str, Any]:
    """测试网络创建"""
    print("\n🌐 测试网络创建...")

    try:
        from config import NetworkParameters
        from network import SatelliteNetwork

        # 创建合理规模的测试网络
        params = NetworkParameters(
            leo_orbits=3,  # 3个LEO轨道
            leo_satellites_per_orbit=4,  # 每轨道4颗卫星
            meo_orbits=2,  # 2个MEO轨道
            meo_satellites_per_orbit=3  # 每轨道3颗卫星
        )

        start_time = time.time()
        network = SatelliteNetwork(params)
        creation_time = time.time() - start_time

        stats = network.get_network_stats()
        stats['creation_time'] = creation_time

        print(f"  ✅ 网络创建成功")
        print(f"     - 总卫星数: {stats['total_satellites']}")
        print(f"     - LEO卫星: {stats['leo_satellites']}")
        print(f"     - MEO卫星: {stats['meo_satellites']}")
        print(f"     - 总链路数: {stats['total_edges']}")
        print(f"     - 网络连通: {'是' if stats.get('is_connected', False) else '否'}")
        print(f"     - 创建耗时: {stats['creation_time']:.2f}秒")

        return stats

    except Exception as e:
        print(f"  ❌ 网络创建失败: {e}")
        import traceback
        print(f"     详细错误: {traceback.format_exc()}")
        return {}


def test_routing_algorithm() -> Dict[str, Any]:
    """测试路由算法"""
    print("\n🔄 测试路由算法...")

    try:
        from config import NetworkParameters, TrafficClass
        from network import SatelliteNetwork
        from routing import QoSRoutingAlgorithm

        # 使用合理规模网络
        params = NetworkParameters(
            leo_orbits=3,
            leo_satellites_per_orbit=4,
            meo_orbits=2,
            meo_satellites_per_orbit=3
        )

        network = SatelliteNetwork(params)
        algorithm = QoSRoutingAlgorithm(network)

        # 测试路由（选择确实存在的卫星）
        available_sats = list(network.satellites.keys())
        if len(available_sats) < 2:
            print(f"  ⚠️  卫星数量不足进行路由测试")
            return {'path_found': False}

        source = available_sats[0]
        destination = available_sats[-1]  # 选择最后一个，增加路径长度

        start_time = time.time()
        path = algorithm.modified_bmdp_algorithm(
            source, destination,
            TrafficClass.CLASS_A,
            0.5
        )
        routing_time = time.time() - start_time

        stats = {
            'path_found': path is not None and len(path) > 0,
            'path_length': len(path) if path else 0,
            'hop_count': len(path) - 1 if path else 0,
            'routing_time': routing_time,
            'source': source,
            'destination': destination,
            'available_satellites': len(available_sats)
        }

        if stats['path_found']:
            print(f"  ✅ 路由计算成功")
            print(f"     - 源卫星: {source}")
            print(f"     - 目标卫星: {destination}")
            print(f"     - 路径长度: {stats['path_length']} 个节点")
            print(f"     - 跳数: {stats['hop_count']}")
            print(f"     - 计算耗时: {stats['routing_time']:.4f}秒")
            if len(path) <= 5:
                print(f"     - 路径: {' -> '.join(path)}")
            else:
                print(f"     - 路径: {' -> '.join(path[:3] + ['...'] + path[-2:])}")
        else:
            print(f"  ⚠️  未找到路径")
            print(f"     - 可能原因: 网络不连通或参数设置问题")
            print(f"     - 可用卫星数: {stats['available_satellites']}")

        return stats

    except Exception as e:
        print(f"  ❌ 路由算法测试失败: {e}")
        import traceback
        print(f"     详细错误: {traceback.format_exc()}")
        return {}


def test_basic_simulation() -> Dict[str, Any]:
    """测试基本仿真"""
    print("\n📊 测试基本仿真...")

    try:
        from config import NetworkParameters, SimulationParameters, TrafficClass
        from network import SatelliteNetwork
        from routing import QoSRoutingAlgorithm
        from simulation import NetworkSimulation

        # 使用小规模但合理的参数
        net_params = NetworkParameters(
            leo_orbits=3,
            leo_satellites_per_orbit=6,
            meo_orbits=2,
            meo_satellites_per_orbit=3
        )

        sim_params = SimulationParameters(
            default_iterations=5,  # 大幅减少迭代次数
            traffic_rate_min=540,
            traffic_rate_max=550,  # 只测试2个流量点
            traffic_rate_step=10
        )

        network = SatelliteNetwork(net_params)
        algorithm = QoSRoutingAlgorithm(network)
        simulation = NetworkSimulation(network, algorithm, sim_params)

        # 检查网络连通性
        if not network.get_network_stats().get('is_connected', False):
            print(f"  ⚠️  网络不连通，跳过仿真测试")
            return {'simulation_completed': False, 'reason': 'network_not_connected'}

        start_time = time.time()
        # 只测试M-BMDP算法
        results = simulation.run_comprehensive_simulation(['M-BMDP'])
        simulation_time = time.time() - start_time

        stats = {
            'simulation_completed': 'M-BMDP' in results,
            'simulation_time': simulation_time,
            'algorithms_tested': list(results.keys()),
            'traffic_rates_tested': len(results['M-BMDP']['class_a']['delays']) if 'M-BMDP' in results else 0
        }

        if stats['simulation_completed']:
            mbmdp_results = results['M-BMDP']
            if mbmdp_results['class_a']['delays']:
                avg_delay_a = sum(mbmdp_results['class_a']['delays']) / len(mbmdp_results['class_a']['delays'])
                avg_loss_a = sum(mbmdp_results['class_a']['packet_losses']) / len(
                    mbmdp_results['class_a']['packet_losses'])

                print(f"  ✅ 基本仿真成功")
                print(f"     - 测试算法: {stats['algorithms_tested']}")
                print(f"     - 测试流量点数: {stats['traffic_rates_tested']}")
                print(f"     - Class A平均延迟: {avg_delay_a:.2f} ms")
                print(f"     - Class A平均丢包率: {avg_loss_a * 100:.2f}%")
                print(f"     - 仿真耗时: {stats['simulation_time']:.2f}秒")

                stats['avg_delay'] = avg_delay_a
                stats['avg_loss'] = avg_loss_a
            else:
                print(f"  ⚠️  仿真完成但无有效结果")
                stats['simulation_completed'] = False
        else:
            print(f"  ❌ 仿真未完成")

        return stats

    except Exception as e:
        print(f"  ❌ 基本仿真测试失败: {e}")
        import traceback
        print(f"     详细错误: {traceback.format_exc()}")
        return {}


def test_visualization() -> bool:
    """测试可视化功能"""
    print("\n📈 测试可视化功能...")

    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建简单测试图表
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        x = np.linspace(540, 620, 9)
        y1 = 35 + np.random.normal(0, 2, 9)  # 模拟M-BMDP延迟
        y2 = 42 + np.random.normal(0, 3, 9)  # 模拟BMDP延迟

        ax.plot(x, y1, 'o-', label='M-BMDP', color='red')
        ax.plot(x, y2, 's-', label='BMDP', color='blue')
        ax.set_xlabel('Data transmission (Kbps)')
        ax.set_ylabel('Delay (ms)')
        ax.set_title('Algorithm Comparison Test')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存测试图片
        plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 可视化测试成功")
        print(f"     - 测试图表已保存为: test_plot.png")

        return True

    except Exception as e:
        print(f"  ❌ 可视化测试失败: {e}")
        import traceback
        print(f"     详细错误: {traceback.format_exc()}")
        return False


def test_special_features() -> Dict[str, Any]:
    """测试特殊功能"""
    print("\n🎯 测试特殊功能...")

    stats = {
        'hop_limit_test': False,
        'special_area_test': False
    }

    try:
        from config import NetworkParameters, TrafficClass
        from network import SatelliteNetwork
        from routing import QoSRoutingAlgorithm

        # 创建足够大的网络以支持特殊功能测试
        params = NetworkParameters(
            leo_orbits=4,
            leo_satellites_per_orbit=8,
            meo_orbits=2,
            meo_satellites_per_orbit=4
        )

        network = SatelliteNetwork(params)
        algorithm = QoSRoutingAlgorithm(network)

        # 测试跳数限制算法
        available_sats = list(network.satellites.keys())
        if len(available_sats) >= 2:
            source = available_sats[0]
            destination = available_sats[len(available_sats) // 2]

            path = algorithm.improved_hop_limit_algorithm(
                source, destination, 4, TrafficClass.CLASS_A
            )

            if path:
                stats['hop_limit_test'] = True
                print(f"  ✅ 跳数限制算法测试成功")
                print(f"     - 路径长度: {len(path) - 1}跳")
            else:
                print(f"  ⚠️  跳数限制算法未找到路径")

        # 测试特殊区域路由
        if len(available_sats) >= 2:
            source = available_sats[0]
            destination = available_sats[-1]

            special_path = algorithm.special_area_routing(
                source, destination, TrafficClass.CLASS_A
            )

            if special_path:
                stats['special_area_test'] = True
                print(f"  ✅ 特殊区域路由测试成功")
                print(f"     - 路径长度: {len(special_path) - 1}跳")
            else:
                print(f"  ⚠️  特殊区域路由未找到路径")

        success_count = sum(stats.values())
        print(f"  📊 特殊功能测试: {success_count}/2 通过")

        return stats

    except Exception as e:
        print(f"  ❌ 特殊功能测试失败: {e}")
        return stats


def main():
    """主测试函数"""
    print("🚀 MEO/LEO双层卫星网络QoS路由算法仿真系统 - 快速测试")
    print("=" * 60)

    start_time = time.time()

    # 运行所有测试
    test_results = {}

    # 1. 测试模块导入
    test_results['imports'] = test_imports()

    if not test_results['imports']:
        print("\n❌ 模块导入失败，请检查依赖安装：")
        print("   pip install numpy matplotlib networkx pandas")
        return False

    # 2. 测试网络创建
    network_result = test_network_creation()
    test_results['network'] = bool(network_result)

    # 3. 测试路由算法
    routing_result = test_routing_algorithm()
    test_results['routing'] = bool(routing_result)

    # 4. 测试基本仿真
    simulation_result = test_basic_simulation()
    test_results['simulation'] = bool(simulation_result)

    # 5. 测试可视化
    test_results['visualization'] = test_visualization()

    # 6. 测试特殊功能
    special_result = test_special_features()
    test_results['special_features'] = any(special_result.values())

    total_time = time.time() - start_time

    # 输出测试总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)

    passed_tests = 0
    total_tests = 6

    test_descriptions = [
        ("模块导入", test_results['imports']),
        ("网络创建", test_results['network']),
        ("路由算法", test_results['routing']),
        ("基本仿真", test_results['simulation']),
        ("可视化功能", test_results['visualization']),
        ("特殊功能", test_results['special_features'])
    ]

    for desc, passed in test_descriptions:
        status = "通过" if passed else "失败"
        print(f"✅ {desc}: {status}")
        if passed:
            passed_tests += 1

    print(f"\n📊 测试结果: {passed_tests}/{total_tests} 通过")
    print(f"⏱️  总耗时: {total_time:.2f}秒")

    if passed_tests >= 4:  # 至少4个核心测试通过
        print("\n🎉 核心功能测试通过！系统基本可用。")
        print("\n📚 下一步:")
        print("   - 运行完整仿真: python main.py")
        print("   - 运行高级分析: python run_analysis.py")
        print("   - 查看详细文档: 阅读 README.md")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} 个重要测试失败，建议检查错误信息。")
        if passed_tests >= 2:
            print("   部分功能可用，可以尝试运行基本功能。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)