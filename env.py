"""
评估训练好的模型
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
from stable_baselines3 import PPO
from jobshoplab import JobShopLabEnv, load_config


def evaluate(model_path, config_name="ft20", episodes=10):
    print("=" * 60)
    print("评估模型")
    print(f"模型: {model_path}")
    print(f"配置文件: {config_name}.yaml")
    print(f"运行次数: {episodes}")
    print("=" * 60)
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 加载配置
    config_path = Path(f"./data/config/{config_name}.yaml")
    config = load_config(config_path=config_path)
    
    makespans = []
    rewards_list = []
    
    for episode in range(episodes):
        env = JobShopLabEnv(config=config)
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done and step_count < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, truncated, terminated, info = env.step(action)
            done = truncated or terminated
            total_reward += reward
            step_count += 1
        
        makespan = info.get('makespan', float('inf'))
        makespans.append(makespan)
        rewards_list.append(total_reward)
        
        print(f"  Episode {episode+1}: Makespan = {makespan}, 步数 = {step_count}")
        
        # 只对最后一个 episode 显示甘特图
        if episode == episodes - 1:
            env.render()
    
    # 统计
    avg_makespan = np.mean(makespans)
    std_makespan = np.std(makespans)
    
    print("\n" + "=" * 60)
    print("📊 评估结果")
    print(f"  平均 Makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    print(f"  最优 Makespan: {min(makespans)}")
    print(f"  最差 Makespan: {max(makespans)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, 
                        default="./models/ppo_ft20_100000steps",
                        help="模型路径")
    parser.add_argument("--config", type=str, default="ft20",
                        help="配置文件名称")
    parser.add_argument("--episodes", type=int, default=10,
                        help="评估次数")
    
    args = parser.parse_args()
    
    evaluate(args.model, args.config, args.episodes)