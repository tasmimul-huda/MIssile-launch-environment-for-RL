
from missile_env import MissileEnv

if __name__ == "__main__":

    # Create the environment
    env = MissileEnv()

    # Reset Observation Space
    observation = env.reset(seed=1997)

    # While the episode is not finished
    done = False
    while not done:
        # Select an action (here, a random one)
        action = env.action_space.sample()

        # One step forward
        observation, reward, done, _ = env.step(action)

        # Render (or not) the environment
        env.render()
