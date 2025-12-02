import gymnasium as gym
from stable_baselines3 import DQN

def main():
    
    env = gym.make("CartPole-v1")

    
    model = DQN(
        "MlpPolicy",
        env,
        verbose = 1,
        learning_rate = 1e-3,
        buffer_size = 50_000,
        batch_size = 64,
        tensorboard_log = "./cartpole_tensorboard/"
    )

    
    model.learn(total_timesteps = 50_000)

    
    model.save("dqn_cartpole")

    env.close()
    print("Training finished and model saved as dqn_cartpole.zip")

if __name__ == "__main__":
    main()