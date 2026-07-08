"""
JobShopLab - PPO 训练脚本 
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv  # 修复导入
from stable_baselines3.common.callbacks import BaseCallback

from jobshoplab import JobShopLabEnv, load_config


def make_env(config_name="ft20"):
    def _init():
        config = load_config(config_path=config_name)
        env = JobShopLabEnv(config=config)
        return env
    return _init


class MakespanCallback(BaseCallback):
    def __init__(self, verbose=0):
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
                if self.verbose > 0:
                    print(f"✅ Episode {self.episode} | 最优 Makespan: {makespan}")
        return True


def train(config_name="ft20", steps=500000, n_envs=8):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=" * 60)
    print("JobShopLab PPO 训练 (修复版)")
    print(f"配置文件: {config_name}.yaml")
    print(f"训练步数: {steps}")
    print(f"并行环境数: {n_envs}")
    print(f"使用设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 60)
    
    # 创建并行环境
    print("🏗️ 创建并行环境...")
    env = make_vec_env(
        make_env(config_name),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,  # 使用SubprocVecEnv
        seed=42
    )
    print("✅ 环境创建成功")
    
    # 创建PPO模型
    print("🤖 创建 PPO 模型...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=16384,
        batch_size=1024,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    )
    
    print(f"📱 模型设备: {model.device}")
    print(f"📱 参数位置: {next(model.policy.parameters()).device}")
    
    print(f"\n🚀 开始训练 ({steps} 步)...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=steps,
        callback=MakespanCallback(verbose=1),
        progress_bar=True,
        log_interval=1
    )
    
    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/ppo_{config_name}_{steps}steps_gpu_fixed"
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"💾 模型已保存: {model_path}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="JobShopLab PPO 训练 ")
    parser.add_argument("--config", type=str, default="real_world_config",
                        help="配置文件名称")
    parser.add_argument("--steps", type=int, default=1000000,
                        help="训练总步数")
    parser.add_argument("--n_envs", type=int, default=8,
                        help="并行环境数")
    
    args = parser.parse_args()
    
    train(args.config, args.steps, args.n_envs)