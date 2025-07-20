# ================================
# 文件: main.py
# 主程序入口
# ================================

from  config import NetworkParameters
from  config import AlgorithmParameters
from  config import SimulationParameters
from network import SatelliteNetwork
from routing import QoSRoutingAlgorithm
from simulation import NetworkSimulation
import numpy as np


def main():
    """主函数 - 运行完整的仿真实验"""
    print("=== MEO/LEO双层卫星网络QoS路由算法仿真 ===\n")

    print("1. 初始化网络参数...")
    params = NetworkParameters()

    print("2. 生成卫星网络拓扑...")
    network = SatelliteNetwork(params)
    print(f"网络生成完成: {len(network.leo_satellites)} LEO卫星, {len(network.meo_satellites)} MEO卫星")
    print(f"网络连接: {network.graph.number_of_edges()} 条链路")

    print("\n3. 初始化路由算法...")
    algorithm_params = AlgorithmParameters()
    algorithm = QoSRoutingAlgorithm(network, algorithm_params)
    print("M-BMDP路由算法初始化完成")

    print("\n4. 初始化仿真器...")
    simulation_params = SimulationParameters()
    simulation = NetworkSimulation(network, algorithm, simulation_params)

    print("\n5. 运行综合仿真实验...")
    traffic_rates = list(range(540, 621, 10))

    # 运行基本仿真
    print("运行基本性能仿真...")
    results = simulation.run_comprehensive_simulation(['M-BMDP', 'BMDP', 'MDSP'])

    print("\n6. 生成性能对比图表...")
    simulation.plot_comprehensive_results(results)

    # 运行跳数限制分析
    print("\n7. 运行跳数限制分析...")
    hop_results = simulation.run_hop_limit_analysis([3, 4, 5, 6, 7])
    simulation.plot_hop_limit_analysis(hop_results)

    # 运行特殊区域分析
    print("\n8. 运行特殊区域分析...")
    special_results = simulation.run_special_area_analysis()
    if special_results:
        simulation.plot_special_area_analysis(special_results)

    # 输出关键结果
    print("\n=== 仿真结果摘要 ===")
    if 'M-BMDP' in results:
        mbmdp_results = results['M-BMDP']
        print(f"M-BMDP算法性能:")
        print(f"  Class A 平均延迟: {np.mean(mbmdp_results['class_a']['delays']):.2f} ms")
        print(f"  Class B 平均延迟: {np.mean(mbmdp_results['class_b']['delays']):.2f} ms")
        print(f"  Class A 平均丢包率: {np.mean(mbmdp_results['class_a']['packet_losses']) * 100:.2f}%")
        print(f"  Class B 平均丢包率: {np.mean(mbmdp_results['class_b']['packet_losses']) * 100:.2f}%")
        print(f"  IOL 平均利用率: {np.mean(mbmdp_results['iol_utilization']) * 100:.2f}%")

    print("\n=== 仿真完成 ===")
    return network, algorithm, simulation, results


if __name__ == "__main__":
    network, algorithm, simulation, results = main()