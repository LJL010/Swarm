"""MEO-LEO集群路由系统主程序 - 支持动态MEO"""
import random
import argparse
import os
import json
from config import Config
from trainer import TrainingEnvironment
from inferencer import ModelInferencer

def validate_dynamic_meo_setup(config: Config) -> bool:
    """验证动态MEO设置的完整性"""
    print("=== 验证动态MEO设置 ===")

    # 检查数据文件
    data_file = config.get('data.data_file', 'data/data.json')
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        # 检查是否有动态MEO位置数据
        has_dynamic_meo = 'meo_positions_per_slot' in data

        if has_dynamic_meo:
            print("✅ 检测到动态MEO位置数据")
            meo_slots = len(data['meo_positions_per_slot'])
            leo_slots = len(data.get('sat_positions_per_slot', []))
            print(f"   MEO时间槽数: {meo_slots}")
            print(f"   LEO时间槽数: {leo_slots}")

            if meo_slots != leo_slots:
                print(f"⚠️  警告: MEO和LEO时间槽数不匹配")
                return False

        else:
            print("❌ 未找到MEO位置数据")
            return False

        # 检查LEO-MEO分配数据
        if 'MEO_per_slot' in data:
            print("✅ 检测到动态LEO-MEO分配数据")
        else:
            print("⚠️  警告: 未找到动态LEO-MEO分配数据")

        # 检查查询数据
        train_queries = data.get('train_queries', [])
        predict_queries = data.get('predict_queries', [])
        print(f"   训练查询数: {len(train_queries)}")
        print(f"   预测查询数: {len(predict_queries)}")

        if not train_queries and not predict_queries:
            print("⚠️  警告: 未找到查询数据")

        return True

    except Exception as e:
        print(f"❌ 数据文件验证失败: {e}")
        return False


def print_dynamic_meo_info(config: Config):
    """打印动态MEO配置信息"""
    print("\n=== 动态MEO配置信息 ===")

    # 网络配置
    dynamic_meo_enabled = config.get('network.enable_dynamic_meo', True)
    reassignment_enabled = config.get('network.enable_dynamic_meo_reassignment', False)
    reassignment_interval = config.get('network.meo_reassignment_interval', 5)

    print(f"动态MEO支持: {'启用' if dynamic_meo_enabled else '禁用'}")
    print(f"动态重分配: {'启用' if reassignment_enabled else '禁用'}")
    if reassignment_enabled:
        print(f"重分配间隔: 每 {reassignment_interval} 个时间槽")

    # 路由配置
    inter_cluster_enabled = config.get('routing.inter_cluster_routing_enabled', True)
    k_paths = config.get('routing.k_paths', 3)
    edge_strategy = config.get('routing.edge_node_selection_strategy', 'advanced')

    print(f"跨集群路由: {'启用' if inter_cluster_enabled else '禁用'}")
    print(f"K路径数量: {k_paths}")
    print(f"边缘节点选择策略: {edge_strategy}")

    # 奖励配置
    inter_cluster_reward = config.get('environment.reward_inter_cluster_success', 2.0)
    meo_adaptation_reward = config.get('environment.reward_meo_adaptation', 0.5)

    print(f"跨集群成功奖励: {inter_cluster_reward}")
    print(f"MEO适应奖励: {meo_adaptation_reward}")

    # 分析配置
    topology_analysis = config.get('training.enable_topology_analysis', True)
    dynamic_viz = config.get('output.enable_dynamic_visualization', True)

    print(f"拓扑分析: {'启用' if topology_analysis else '禁用'}")
    print(f"动态可视化: {'启用' if dynamic_viz else '禁用'}")


def generate_sample_dynamic_meo_data(config: Config):
    """生成示例动态MEO数据"""
    print("\n=== 生成示例动态MEO数据 ===")

    data_file = config.get('data.data_file', 'data/data.json')

    try:
        # 检查是否需要生成示例数据
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)

            if 'meo_positions_per_slot' in data:
                print("✅ 已存在动态MEO数据，跳过生成")
                return True
        else:
            print("❌ 数据文件不存在，无法生成示例数据")
            return False

        # 生成动态MEO数据
        from data.data_loader import generate_sample_dynamic_meo_data

        num_slots = len(data.get('sat_positions_per_slot', []))
        num_meos = data.get('num_meo_satellites', 3)

        if num_slots == 0:
            print("❌ 没有LEO位置数据，无法生成MEO数据")
            return False

        print(f"为 {num_slots} 个时间槽生成 {num_meos} 个MEO的动态位置数据...")

        meo_positions_per_slot = generate_sample_dynamic_meo_data(num_slots, num_meos)
        data['meo_positions_per_slot'] = meo_positions_per_slot

        # 备份原文件
        backup_file = data_file + '.backup'
        if os.path.exists(data_file):
            import shutil
            shutil.copy2(data_file, backup_file)
            print(f"原文件已备份到: {backup_file}")

        # 保存新数据
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ 动态MEO数据已生成并保存到: {data_file}")
        return True

    except Exception as e:
        print(f"❌ 生成示例数据失败: {e}")
        return False


def run_benchmark_comparison(config: Config, args):
    """运行动态MEO与静态MEO的基准对比"""
    print("\n=== 运行基准对比 ===")

    if not config.get('analysis.benchmark_against_static_meo', False):
        print("基准对比功能未启用")
        return

    print("此功能需要额外的实现...")
    # 这里可以实现动态MEO vs 静态MEO的性能对比


def main():
    """主函数 - 支持动态MEO"""
    parser = argparse.ArgumentParser(description='MEO-LEO集群路由系统 - 动态MEO版本')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['train', 'inference', 'evaluate', 'data', 'setup'], default='train',
                        help='运行模式: train=训练, inference=推理, evaluate=评估, data=数据加载演示, setup=设置向导')
    parser.add_argument('--model', help='模型文件路径（推理和评估模式需要）')
    parser.add_argument('--use-predict-data', action='store_true', default=True,
                        help='推理时使用预测数据集（默认）')
    parser.add_argument('--use-train-data', action='store_true',
                        help='推理时使用训练数据集')
    parser.add_argument('--output-dir', help='结果输出目录')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='生成结果图表（默认开启）')
    parser.add_argument('--generate-sample-data', action='store_true',
                        help='生成示例动态MEO数据')
    parser.add_argument('--force-dynamic-meo', action='store_true',
                        help='强制启用动态MEO模式')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行基准对比测试')

    args = parser.parse_args()

    print("🛰️ MEO-LEO集群路由系统 - 动态MEO版本")
    print("=" * 50)

    # 加载配置
    try:
        config = Config(args.config)
        print(f"✅ 已加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        return
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return

    # 设置随机种子
    random_seed = config.get('simulation.random_seed', 42)
    random.seed(random_seed)
    print(f"🎲 随机种子: {random_seed}")

    # 强制启用动态MEO（如果指定）
    if args.force_dynamic_meo:
        config.update('network.enable_dynamic_meo', True)
        print("🔄 强制启用动态MEO模式")

    # 打印动态MEO配置信息
    print_dynamic_meo_info(config)

    # 验证动态MEO设置
    if not validate_dynamic_meo_setup(config):
        print("⚠️  动态MEO设置验证失败")
        if not args.generate_sample_data:
            response = input("是否生成示例动态MEO数据？(y/N): ")
            if response.lower() == 'y':
                args.generate_sample_data = True

    # 生成示例数据（如果需要）
    if args.generate_sample_data:
        if not generate_sample_dynamic_meo_data(config):
            print("❌ 无法生成示例数据，程序退出")
            return

    # 根据模式执行不同功能
    if args.mode == 'setup':
        print("\n=== 设置向导 ===")
        print("动态MEO设置向导功能")
        # 这里可以实现交互式设置向导
        print("设置向导功能开发中...")

    elif args.mode == 'train':
        print("\n=== 开始训练 (动态MEO) ===")
        trainer = TrainingEnvironment(config)
        trainer.train()

    elif args.mode == 'inference':
        print("\n=== 开始推理 (动态MEO) ===")

        # 确定模型路径
        if args.model:
            model_path = args.model
        else:
            # 使用默认的最终模型路径
            results_path = config.get('output.results_path', 'results/')
            model_path = os.path.join(results_path, 'final_model.json')

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请先进行训练或指定正确的模型路径")
            return

        # 创建推理器
        inferencer = ModelInferencer(config)

        # 加载模型
        if not inferencer.load_trained_model(model_path):
            print("❌ 模型加载失败")
            return

        # 确定使用的数据集
        use_predict_data = not args.use_train_data  # 默认使用预测数据
        data_type = "预测" if use_predict_data else "训练"
        print(f"🔍 使用{data_type}数据集进行推理")

        # 运行推理
        results = inferencer.run_inference(use_predict_data=use_predict_data)

        # 保存结果
        output_dir = args.output_dir if args.output_dir else None
        inferencer.save_results(output_dir)

        # 生成图表
        if args.plot:
            inferencer.plot_results(output_dir)

        # 模型质量评估
        quality_metrics = inferencer.evaluate_model_quality()
        print(f"\n📊 模型综合质量评分: {quality_metrics.get('overall_quality', 0):.3f}")
        print(f"🔄 动态环境性能: {quality_metrics.get('dynamic_performance', 0):.3f}")

    elif args.mode == 'evaluate':
        print("\n=== 开始评估 (动态MEO) ===")

        # 确定模型路径
        if args.model:
            model_path = args.model
        else:
            results_path = config.get('output.results_path', 'results/')
            model_path = os.path.join(results_path, 'final_model.json')

        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            print("请先进行训练或指定正确的模型路径")
            return

        # 创建推理器
        inferencer = ModelInferencer(config)

        # 加载模型
        if not inferencer.load_trained_model(model_path):
            print("❌ 模型加载失败")
            return

        # 在训练数据和预测数据上都进行评估
        print("📈 在训练数据上评估...")
        train_results = inferencer.run_inference(use_predict_data=False)
        train_metrics = inferencer.performance_metrics.copy()

        print("\n📉 在预测数据上评估...")
        pred_results = inferencer.run_inference(use_predict_data=True)
        pred_metrics = inferencer.performance_metrics.copy()

        # 比较结果
        print("\n=== 评估结果比较 ===")
        print(f"训练集成功率: {train_metrics['success_rate']:.2%}")
        print(f"预测集成功率: {pred_metrics['success_rate']:.2%}")
        print(f"泛化差异: {abs(train_metrics['success_rate'] - pred_metrics['success_rate']):.2%}")

        print(f"训练集平均跳数: {train_metrics['average_hops']:.2f}")
        print(f"预测集平均跳数: {pred_metrics['average_hops']:.2f}")

        # 动态MEO特定指标
        if 'inter_cluster_success_rate' in train_metrics:
            print(f"训练集跨集群成功率: {train_metrics['inter_cluster_success_rate']:.2%}")
            print(f"预测集跨集群成功率: {pred_metrics['inter_cluster_success_rate']:.2%}")

        if 'average_meo_movement' in pred_metrics:
            print(f"平均MEO移动距离: {pred_metrics['average_meo_movement']:.2f}")

        # 保存评估结果
        output_dir = args.output_dir if args.output_dir else config.get('output.results_path', 'results/')
        os.makedirs(output_dir, exist_ok=True)

        evaluation_results = {
            'model_path': model_path,
            'train_metrics': train_metrics,
            'predict_metrics': pred_metrics,
            'generalization_gap': abs(train_metrics['success_rate'] - pred_metrics['success_rate']),
            'dynamic_meo_enabled': True,
            'evaluation_timestamp': json.dumps(str(datetime.now()))
        }

        eval_file = os.path.join(output_dir, 'evaluation_results_dynamic_meo.json')
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"\n💾 评估结果已保存到: {eval_file}")

        # 生成比较图表
        if args.plot:
            inferencer.plot_results(output_dir)

    elif args.mode == 'data':
        print("\n=== 数据加载演示 (动态MEO) ===")
        from data.data_loader import load_complete_environment, print_environment_summary, validate_dynamic_meo_data

        data_file = config.get('data.data_file', 'data/data.json')

        # 验证动态MEO数据
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)

            is_dynamic = validate_dynamic_meo_data(data)
            print(f"动态MEO数据: {'✅ 有效' if is_dynamic else '❌ 无效或不存在'}")

        except Exception as e:
            print(f"❌ 数据验证失败: {e}")
            return

        # 演示几个时间槽的数据加载
        demo_slots = [0, 25, 49]
        print(f"\n演示时间槽: {demo_slots}")

        for slot_id in demo_slots:
            try:
                leos, meos, data = load_complete_environment(slot_id, data_file)
                print_environment_summary(leos, meos, slot_id)

                # 显示MEO移动信息（如果可用）
                if slot_id > 0 and is_dynamic:
                    prev_leos, prev_meos, _ = load_complete_environment(slot_id - 1, data_file)
                    print(f"MEO移动信息 (从时间槽 {slot_id-1} 到 {slot_id}):")
                    for meo_id in meos:
                        if meo_id in prev_meos:
                            prev_pos = (prev_meos[meo_id].latitude, prev_meos[meo_id].longitude, prev_meos[meo_id].altitude)
                            curr_pos = (meos[meo_id].latitude, meos[meo_id].longitude, meos[meo_id].altitude)
                            distance = ((curr_pos[0] - prev_pos[0])**2 +
                                      (curr_pos[1] - prev_pos[1])**2 +
                                      (curr_pos[2] - prev_pos[2])**2)**0.5
                            print(f"  MEO {meo_id}: 移动距离 {distance:.2f}")

            except Exception as e:
                print(f"❌ 加载时间槽 {slot_id} 失败: {e}")

    else:
        print(f"❌ 未知的运行模式: {args.mode}")
        parser.print_help()
        return

    # 运行基准对比（如果启用）
    if args.benchmark:
        run_benchmark_comparison(config, args)

    print("\n🎉 程序执行完成!")


if __name__ == "__main__":
    from datetime import datetime
    main()