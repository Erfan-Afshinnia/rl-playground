import time
import gymnasium as gym
from stable_baselines3 import SAC

def main():

    env = gym.make("Pendulum-v1" , render_mode = "human")

    model = SAC.load("sac_pendulum" , env = env)

    num_episodes = 5
    for ep in range(num_episodes):
        obs,info = env.reset()
        done = False
        total_reward = 0.0

        while not done:

            action, _ = model.predict(obs , deterministic = True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.02)
        print(f"Episode {ep+1}: total reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()