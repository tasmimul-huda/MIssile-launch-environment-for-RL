from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import random
from tqdm import tqdm
import time

import numpy as np 
import pandas as pd 
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
import random
from config import Config

from missile_env import MissileEnv

env = MissileEnv()
# # Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=Config.DQN_HYPERPARAMETER.REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=env.observation_space.shape))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(128))

        model.add(Dense(env.action_space.n, activation='linear')) 
        model.compile(loss="mse", optimizer=Adam(lr=0.003), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        if len(self.replay_memory) < Config.DQN_HYPERPARAMETER.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, Config.DQN_HYPERPARAMETER.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + Config.DQN_HYPERPARAMETER.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        # self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        self.model.fit(np.array(X)/255, np.array(y), batch_size=Config.DQN_HYPERPARAMETER.MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > Config.DQN_HYPERPARAMETER.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]



agent = DQNAgent()

ep_rewards = [-200]
# Iterate over episodes
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

        if Config.DQN_HYPERPARAMETER.SHOW_PREVIEW and not episode % Config.DQN_HYPERPARAMETER.AGGREGATE_STATS_EVERY:
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
    if epsilon > Config.DQN_HYPERPARAMETER.MIN_EPSILON:
        epsilon *= Config.DQN_HYPERPARAMETER.EPSILON_DECAY
        epsilon = max(Config.DQN_HYPERPARAMETER.MIN_EPSILON, epsilon)