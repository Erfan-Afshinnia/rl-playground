import gymnasium as gym
from stable_baselines3 import SAC

def main():

    #Create the environment
    env = gym.make("Pendulum-v1")

    #Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        verbose = 1,
        learning_rate = 3e-4,
        buffer_size = 100_000,
        batch_size = 256,
        tau = 0.02,
        tensorboard_log = "./pendulum_tensorboard/"
    )

    #Train
    model.learn(total_timesteps = 100_000)

    #Save
    model.save("sac_pendulum")
    env.close()
    print("Training finished and model saved as sac_pendulum.zip")

if __name__ == "__main__":
    main()
