"""
测试训练好的智能体
支持单次测试、多次评估、对比不同实例
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import time
import numpy as np
from stable_baselines3 import PPO
from jobshoplab import JobShopLabEnv, load_config


# ============================================
# 1. 颜色输出（美化）
# ============================================

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_green(msg): print(f"{Colors.GREEN}{msg}{Colors.RESET}")
def print_yellow(msg): print(f"{Colors.YELLOW}{msg}{Colors.RESET}")
def print_blue(msg): print(f"{Colors.BLUE}{msg}{Colors.RESET}")
def print_red(msg): print(f"{Colors.RED}{msg}{Colors.RESET}")
def print_bold(msg): print(f"{Colors.BOLD}{msg}{Colors.RESET}")


# ============================================
# 2. 核心测试函数
# ============================================

def test_single(model_path, config_name, episodes=1, render=True, verbose=True):
    """
    测试单个配置
    """
    if verbose:
        print_bold("\n" + "=" * 70)
        print_bold(f"🧪 测试智能体")
        print_bold(f"   模型: {model_path}")
        print_bold(f"   配置: {config_name}.yaml")
        print_bold(f"   运行次数: {episodes}")
        print_bold("=" * 70)
    
    # 加载模型
    try:
        model = PPO.load(model_path)
        if verbose:
            print_green("✅ 模型加载成功")
    except Exception as e:
        print_red(f"❌ 模型加载失败: {e}")
        return None
    
    # 加载配置
    config_path = Path(f"./data/config/{config_name}.yaml")
    if not config_path.exists():
        print_red(f"❌ 配置文件不存在: {config_path}")
        print_yellow("💡 可用配置文件:")
        for f in Path("./data/config/").glob("*.yaml"):
            print(f"   - {f.name}")
        return None
    
    try:
        config = load_config(config_path=config_path)
        if verbose:
            print_green("✅ 配置加载成功")
    except Exception as e:
        print_red(f"❌ 配置加载失败: {e}")
        return None
    
    results = []
    
    for episode in range(episodes):
        if verbose:
            print(f"\n📊 Episode {episode + 1}/{episodes}")
            print("-" * 50)
        
        env = JobShopLabEnv(config=config)
        obs, _ = env.reset()
        
        done = False
        step_count = 0
        total_reward = 0
        
        start_time = time.time()
        
        while not done and step_count < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated
            total_reward += reward
            step_count += 1
        
        elapsed = time.time() - start_time
        
        makespan = info.get('makespan', float('inf'))
        results.append({
            'makespan': makespan,
            'steps': step_count,
            'reward': total_reward,
            'time': elapsed
        })
        
        if verbose:
            print(f"   📋 Makespan: {makespan}")
            print(f"   📋 步数: {step_count}")
            print(f"   📋 总奖励: {total_reward:.2f}")
            print(f"   ⏱️  耗时: {elapsed:.2f}秒")
        
        # 最后一次渲染
        if render and episode == episodes - 1:
            if verbose:
                print("\n🖥️  渲染甘特图...")
            env.render()
    
    # 统计
    if episodes > 1:
        makespans = [r['makespan'] for r in results]
        avg = np.mean(makespans)
        std = np.std(makespans)
        
        if verbose:
            print_bold("\n" + "=" * 70)
            print_bold("📊 统计结果")
            print(f"   平均 Makespan: {avg:.2f} ± {std:.2f}")
            print(f"   最优 Makespan: {min(makespans)}")
            print(f"   最差 Makespan: {max(makespans)}")
            print("=" * 70)
    
    return results


# ============================================
# 3. 对比多个配置
# ============================================

def compare_instances(model_path, config_names, episodes=5):
    """
    对比多个配置的效果
    """
    print_bold("\n" + "=" * 70)
    print_bold("📊 对比不同实例")
    print_bold(f"模型: {model_path}")
    print_bold(f"每个实例运行 {episodes} 次")
    print_bold("=" * 70)
    
    results = {}
    
    for config_name in config_names:
        print(f"\n📂 测试: {config_name}.yaml")
        print("-" * 40)
        
        # 不渲染，只收集数据
        r = test_single(model_path, config_name, episodes, render=False, verbose=False)
        if r:
            makespans = [x['makespan'] for x in r]
            results[config_name] = {
                'avg': np.mean(makespans),
                'std': np.std(makespans),
                'min': min(makespans),
                'max': max(makespans),
                'data': makespans
            }
            print(f"   Makespan: {results[config_name]['avg']:.2f} ± {results[config_name]['std']:.2f}")
            print(f"   最优/最差: {results[config_name]['min']} / {results[config_name]['max']}")
    
    # 汇总对比
    print_bold("\n" + "=" * 70)
    print_bold("📊 对比汇总")
    print("-" * 50)
    print(f"{'实例':<15} {'平均 Makespan':<20} {'最优':<10} {'最差':<10}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<15} {data['avg']:<20.2f} {data['min']:<10} {data['max']:<10}")
    print("=" * 70)
    
    return results


# ============================================
# 4. 随机策略对比
# ============================================

def test_random(config_name, episodes=5):
    """
    测试随机策略作为对比基线
    """
    print_bold("\n" + "=" * 70)
    print_bold("🎲 测试随机策略 (基线)")
    print_bold(f"配置: {config_name}.yaml")
    print_bold(f"运行次数: {episodes}")
    print_bold("=" * 70)
    
    config_path = Path(f"./data/config/{config_name}.yaml")
    if not config_path.exists():
        print_red(f"❌ 配置文件不存在: {config_path}")
        return None
    
    config = load_config(config_path=config_path)
    results = []
    
    for episode in range(episodes):
        print(f"\n📊 Episode {episode + 1}/{episodes}")
        print("-" * 40)
        
        env = JobShopLabEnv(config=config)
        obs, _ = env.reset()
        
        done = False
        step_count = 0
        total_reward = 0
        
        while not done and step_count < 2000:
            action = env.action_space.sample()  # 随机动作
            obs, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated
            total_reward += reward
            step_count += 1
        
        makespan = info.get('makespan', float('inf'))
        results.append(makespan)
        print(f"   Makespan: {makespan}, 步数: {step_count}")
    
    avg = np.mean(results)
    std = np.std(results)
    
    print_bold("\n" + "=" * 70)
    print_bold("📊 随机策略统计")
    print(f"   平均 Makespan: {avg:.2f} ± {std:.2f}")
    print(f"   最优 Makespan: {min(results)}")
    print(f"   最差 Makespan: {max(results)}")
    print("=" * 70)
    
    return results


# ============================================
# 5. 主入口
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 JobShopLab 智能体")
    
    # 测试模式
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "compare", "random"],
                        help="测试模式: single=单配置, compare=多配置对比, random=随机策略")
    
    # 模型参数
    parser.add_argument("--model", type=str, 
                        default="./models/ppo_real_world_config_10000000steps_gpu_fixed",
                        help="模型路径")
    
    # 配置参数
    parser.add_argument("--config", type=str, default="real_world_config",
                        help="配置文件名称")
    parser.add_argument("--configs", type=str, nargs="+",
                        help="对比模式下的多个配置名称")
    
    # 运行参数
    parser.add_argument("--episodes", type=int, default=1,
                        help="运行次数")
    parser.add_argument("--no-render", action="store_true",
                        help="不渲染甘特图")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="显示详细信息")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        test_single(
            args.model, 
            args.config, 
            args.episodes, 
            render=not args.no_render,
            verbose=args.verbose
        )
    
    elif args.mode == "compare":
        if args.configs is None:
            print_red("❌ 对比模式需要指定 --configs 参数")
            print("例如: --configs ft06 ft10 la01")
        else:
            compare_instances(args.model, args.configs, args.episodes)
    
    elif args.mode == "random":
        test_random(args.config, args.episodes)