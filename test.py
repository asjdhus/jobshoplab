"""
JobShopLab - PPO 训练脚本
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from jobshoplab import JobShopLabEnv, load_config


# ============================================
# 1. 环境工厂函数
# ============================================

def make_env(config_name="ft10"):
    """创建环境的工厂函数"""
    def _init():
        # 使用 config_name 而不是 config_path
        config = load_config(config_path=config_name)
        print(f"✅ 配置加载成功: {config_name}")
        env = JobShopLabEnv(config=config)
        return env
    return _init


# ============================================
# 2. 训练回调
# ============================================

class MakespanCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.best_makespan = float('inf')
        self.episode = 0
    
    def _on_step(self):
        if self.locals.get('done', False):
            self.episode += 1
            info = self.locals.get('info', {})
            makespan = info.get('makespan', float('inf'))
            if makespan < self.best_makespan and makespan > 0:
                self.best_makespan = makespan
                print(f"✅ Episode {self.episode} | 最优 Makespan: {makespan}")
        return True


# ============================================
# 3. 主训练函数
# ============================================

def train(config_name="ft10", steps=100000):
    print("=" * 60)
    print("JobShopLab PPO 训练")
    print(f"配置文件: {config_name}.yaml")
    print(f"训练步数: {steps}")
    print("=" * 60)
    
    print("🏗️ 创建环境...")
    env = make_vec_env(
        make_env(config_name),
        n_envs=1,
        seed=42
    )
    print("✅ 环境创建成功")
    
    print("🤖 创建 PPO 模型...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
    )
    
    print(f"\n🚀 开始训练 ({steps} 步)...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=steps,
        callback=MakespanCallback(verbose=1),
        progress_bar=True,
    )
    
    import os
    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/ppo_{config_name}_{steps}steps"
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"💾 模型已保存: {model_path}")
    print("=" * 60)
    
    return model


# ============================================
# 4. 主入口
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JobShopLab PPO 训练")
    parser.add_argument("--config", type=str, default="ft10",
                        help="配置文件名称 (不含 .yaml)")
    parser.add_argument("--steps", type=int, default=100000,
                        help="训练总步数")
    
    args = parser.parse_args()
    
    train(args.config, args.steps)