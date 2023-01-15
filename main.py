from missile_env import MissileEnv
import warnings
import time
from dqn_agent import DQNAgent
from tqdm import tqdm
from config import Config
import numpy as np

warnings.filterwarnings('ignore')

ep_rewards = [-200]
# Iterate over episodes

def update():
    for episode in tqdm(range(1, Config.DQN_HYPERPARAMETER.EPISODES + 1), ascii=True, unit='episodes'):

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > Config.DQN_HYPERPARAMETER.epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # if Config.DQN_HYPERPARAMETER.SHOW_PREVIEW and not episode % Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY:
            env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)

            current_state = new_state
            step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        
        if not episode % Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY:])/len(ep_rewards[-Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY:])
        
            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= Config.DQN_HYPERPARAMETER.MIN_REWARD:
                agent.model.save(f'models/{Config.DQN_HYPERPARAMETER.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if Config.DQN_HYPERPARAMETER.epsilon > Config.DQN_HYPERPARAMETER.MIN_EPSILON:
            Config.DQN_HYPERPARAMETER.epsilon *= Config.DQN_HYPERPARAMETER.EPSILON_DECAY
            Config.DQN_HYPERPARAMETER.epsilon = max(Config.DQN_HYPERPARAMETER.MIN_EPSILON, Config.DQN_HYPERPARAMETER.epsilon)
        
        
        
if __name__ == '__main__':
    env = MissileEnv()
    agent = DQNAgent(env)
    update()
    