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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from jobshoplab import JobShopLabEnv, load_config


def make_env(config_name="ft20"):
    def _init():
        config = load_config(config_path=config_name)
        env = JobShopLabEnv(config=config)
        return env
    return _init


class EntropyDecayCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.5, final_ent_coef=0.02, total_steps=5000000, verbose=0):
        super().__init__(verbose)
        self.initial = initial_ent_coef
        self.final = final_ent_coef
        self.total = total_steps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total, 1.0)
        self.model.ent_coef = self.initial + (self.final - self.initial) * progress
        return True


class MakespanCallback(BaseCallback):
    def __init__(self, model=None, save_path="./models/best_model",
                 min_episodes=200, patience=10000, target_makespan=76, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.save_path = save_path
        self.min_episodes = min_episodes
        self.patience = patience
        self.target_makespan = target_makespan
        self.best_makespan = float('inf')
        self.episode_count = 0
        self._step_counter = 0
        self._done_count = 0
        self._first_call = True
        self._no_improve_episodes = 0
        self._last_log_timesteps = 0

    def _on_step(self):
        self._step_counter += 1

        if self._first_call:
            self._first_call = False
            print(f"[MakespanCallback] 收敛条件: target={self.target_makespan}, patience={self.patience}ep, min={self.min_episodes}ep")

        dones = self.locals.get('dones')
        infos = self.locals.get('infos')

        if dones is None:
            dones = self.locals.get('done')
            infos = self.locals.get('info')

        if dones is None:
            return True

        if not hasattr(dones, '__iter__'):
            dones = [dones]
            infos = [infos] if infos is not None else [{}]

        for env_idx, done in enumerate(dones):
            if done:
                self._done_count += 1
                self.episode_count += 1
                info = infos[env_idx] if env_idx < len(infos) else {}
                if isinstance(info, dict):
                    makespan = info.get('makespan')
                    terminated = info.get('terminated', False)
                    truncated = info.get('truncated', False)
                    if self.episode_count <= 5:
                        print(f"[DEBUG] Ep {self.episode_count} | info keys={list(info.keys())} | term={terminated} | trunc={truncated} | ms={makespan}")
                    if makespan is not None:
                        print(f"Episode {self.episode_count} | Makespan: {makespan}")
                        if makespan < self.best_makespan:
                            self.best_makespan = makespan
                            self._no_improve_episodes = 0
                            print(f"    New best makespan: {makespan}")
                            if (self.model is not None
                                    and self.episode_count >= self.min_episodes):
                                self.model.save(self.save_path)
                                print(f"    Model saved to: {self.save_path}")
                            if makespan <= self.target_makespan:
                                print(f"    🎯 达到目标 makespan={self.target_makespan}，停止训练!")
                                return False
                        else:
                            if self.episode_count >= self.min_episodes:
                                self._no_improve_episodes += 1
                                if self._no_improve_episodes >= self.patience:
                                    print(f"    ⏹ 连续 {self.patience} 个 episode 无改进 (best={self.best_makespan})，停止训练!")
                                    return False
                    elif truncated:
                        print(f"Episode {self.episode_count} | TRUNCATED (no makespan)")
                elif self.episode_count <= 5:
                    print(f"[DEBUG] Ep {self.episode_count} | info type={type(info).__name__} | val={info}")

        if self.num_timesteps - self._last_log_timesteps >= 100000:
            self._last_log_timesteps = self.num_timesteps
            print(f"[Callback] total_steps={self.num_timesteps} | episodes={self.episode_count} | best={self.best_makespan} | no_impr={self._no_improve_episodes}")

        return True


def train(config_name="ft20", steps=5000000, n_envs=8):
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
        vec_env_cls=DummyVecEnv,
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
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=1.0,
        target_kl=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/",
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    )
    
    print(f"📱 模型设备: {model.device}")
    print(f"📱 参数位置: {next(model.policy.parameters()).device}")
    
    print(f"\n🚀 开始训练 ({steps} 步)...")
    print("-" * 60)
    
    makespan_callback = MakespanCallback(
        model=model,
        save_path=f"./models/best_{config_name}",
        verbose=1
    )

    entropy_callback = EntropyDecayCallback(
        initial_ent_coef=1.0,
        final_ent_coef=0.02,
        total_steps=steps,
        verbose=0
    )

    model.learn(
        total_timesteps=steps,
        callback=[makespan_callback, entropy_callback],
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
    parser.add_argument("--steps", type=int, default=5000000,
                        help="训练总步数")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="并行环境数")
    
    args = parser.parse_args()
    
    train(args.config, args.steps, args.n_envs)