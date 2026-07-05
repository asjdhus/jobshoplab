from pathlib import Path

from jobshoplab import JobShopLabEnv, load_config

# Load a pre-defined configuration
config = load_config(config_path=Path("./data/config/getting_started_config.yaml"))

# Create the environment
env = JobShopLabEnv(config=config)

# Run with random actions until done
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, truncated, terminated, info = env.step(action)
    done = truncated or terminated

# Visualize the final schedule
env.render()