# MEO/LEO双层卫星网络QoS路由算法仿真

基于论文《Swarm and Location-Based QoS Routing Algorithm in MEO/LEO Double-Layered Satellite Networks》的完整实现。

## 项目特性

- ✅ M-BMDP路由算法（修改的带宽约束最小延迟路径算法）
- ✅ 双层卫星网络拓扑（LEO: 72颗，MEO: 10颗卫星）
- ✅ QoS流量分类（延迟敏感 vs 其他类型）
- ✅ 算法性能对比（M-BMDP vs BMDP vs MDSP）
- ✅ 可视化分析和报告生成

## 快速开始

### 1. 安装依赖
```bash
pip install numpy matplotlib networkx pandas
```

### 2. 验证安装
```bash
python quick_test.py
```

### 3. 运行仿真
```bash
# 基本仿真（推荐）
python main.py

# 高级分析
python run_analysis.py
```

## 文件结构

```
├── config.py          # 配置参数
├── satellite.py       # 卫星类定义
├── network.py         # 网络拓扑生成
├── routing.py         # 路由算法实现
├── simulation.py      # 网络仿真
├── analysis.py        # 高级分析工具
├── utils.py           # 工具函数
├── main.py            # 主程序入口
└── run_analysis.py    # 高级分析入口
```

## 输出结果

运行后将生成：
- **性能对比图表**：延迟、丢包率、IOL利用率分析
- **控制台报告**：算法性能摘要
- **JSON数据文件**：详细分析结果

### 示例输出
```
=== 仿真结果摘要 ===
M-BMDP算法性能:
  Class A 平均延迟: 45.67 ms
  Class B 平均延迟: 52.34 ms
  Class A 平均丢包率: 2.45%
  Class B 平均丢包率: 1.89%
  IOL 平均利用率: 67.23%
```

## 参数调整

在 `config.py` 中可以调整：

```python
# 网络规模
leo_orbits = 6              # LEO轨道数
leo_satellites_per_orbit = 12   # 每轨道卫星数

# 仿真参数  
default_iterations = 100    # 仿真次数
traffic_rate_min = 540      # 流量范围
traffic_rate_max = 620
```

## 自定义使用

```python
from config import NetworkParameters, TrafficClass
from network import SatelliteNetwork
from routing import QoSRoutingAlgorithm

# 创建网络
network = SatelliteNetwork(NetworkParameters())
algorithm = QoSRoutingAlgorithm(network)

# 测试路由
path = algorithm.modified_bmdp_algorithm(
    "LEO_0_0", "LEO_3_6", 
    TrafficClass.CLASS_A, 0.5
)
```

## 系统要求

- Python 3.7+
- numpy, matplotlib, networkx, pandas
- 内存: 建议2GB以上
- 运行时间: 完整仿真约2-5分钟

---

🚀 **开始您的卫星网络QoS路由研究！**