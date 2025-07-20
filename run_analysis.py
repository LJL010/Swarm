# ================================
# 文件: run_analysis.py
# 运行高级分析
# ================================
from network import SatelliteNetwork
from simulation import *
from config import *


def run_advanced_analysis():
    """运行高级分析"""
    print("=== 开始高级分析 ===\n")

    # 1. 初始化系统
    print("1. 初始化网络系统...")
    params = NetworkParameters()
    network = SatelliteNetwork(params)
    algorithm = QoSRoutingAlgorithm(network)
    simulation = NetworkSimulation(network, algorithm)

    # 2. 创建高级分析器
    from analysis import AdvancedAnalyzer
    analyzer = AdvancedAnalyzer(network, algorithm, simulation)

    # 3. 运行各项分析
    print("\n2. 分析网络拓扑...")
    topology_results = analyzer.analyze_network_topology()

    print("\n3. 分析IOL性能...")
    iol_results = analyzer.analyze_iol_performance()

    print("\n4. 分析路由效率...")
    routing_results = analyzer.analyze_routing_efficiency()

    print("\n5. 分析地理分布...")
    geo_results = analyzer.analyze_geographical_distribution()

    # 4. 生成报告
    print("\n6. 生成综合报告...")
    report = analyzer.generate_comprehensive_report()
    print(report)

    # 5. 保存结果
    print("\n7. 保存分析结果...")
    analyzer.save_analysis_results()

    # 6. 绘制图表
    print("\n8. 生成分析图表...")
    analyzer.plot_comprehensive_analysis()

    print("\n=== 高级分析完成 ===")
    return analyzer


if __name__ == "__main__":
    analyzer = run_advanced_analysis()