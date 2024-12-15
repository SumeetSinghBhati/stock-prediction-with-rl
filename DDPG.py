import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import Portfolio

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

HIDDEN1_UNITS = 24
HIDDEN2_UNITS = 48
HIDDEN3_UNITS = 24

class ActorNetwork:
    def __init__(self, state_size, action_dim, buffer_size, tau, learning_rate, is_eval=False, model_name=""):
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim

        if is_eval:
            self.model = self.create_actor_network(state_size, action_dim)
            self.model.load_weights('saved_models/{}_actor.h5'.format(model_name))
            self.model_target = None
        else:
            self.model = self.create_actor_network(state_size, action_dim)
            self.model_target = self.create_actor_network(state_size, action_dim)
            self.model_target.set_weights(self.model.get_weights())

    def train_target(self):
        if self.model_target is not None:
            actor_weights = self.model.get_weights()
            actor_target_weights = self.model_target.get_weights()
            for i in range(len(actor_weights)):
                actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
            self.model_target.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(states)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN3_UNITS, activation='relu')(h1)
        actions = Dense(action_dim, activation='softmax')(h2)
        model = Model(inputs=states, outputs=actions)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model


class CriticNetwork:
    def __init__(self, state_size, action_dim, tau, learning_rate, is_eval=False, model_name=""):
        self.tau = tau
        self.learning_rate = learning_rate
        self.action_dim = action_dim

        if is_eval:
            self.model = self.create_critic_network(state_size, action_dim)
            self.model.load_weights('saved_models/{}_critic.h5'.format(model_name))
            self.model_target = None
        else:
            self.model = self.create_critic_network(state_size, action_dim)
            self.model_target = self.create_critic_network(state_size, action_dim)
            self.model_target.set_weights(self.model.get_weights())

    def gradients(self, states_batch, actions_batch):
        # In TF2, use GradientTape for computing gradients
        with tf.GradientTape() as tape:
            Q_values = self.model([states_batch, actions_batch])
        return tape.gradient(Q_values, actions_batch)

    def train_target(self):
        if self.model_target is not None:
            critic_weights = self.model.get_weights()
            critic_target_weights = self.model_target.get_weights()
            for i in range(len(critic_weights)):
                critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
            self.model_target.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        states = Input(shape=[state_size])
        actions = Input(shape=[action_dim])
        h0 = Concatenate()([states, actions])
        h1 = Dense(HIDDEN1_UNITS, activation='relu')(h0)
        h2 = Dense(HIDDEN2_UNITS, activation='relu')(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu')(h2)
        Q = Dense(action_dim, activation='relu')(h3)
        model = Model(inputs=[states, actions], outputs=Q)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.states = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.states
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.states = x + dx
        return self.states

    def get_actions(self, actions, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(actions + ou_state, 0, 1)


class Agent(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False, model_name=""):
        super().__init__(balance=balance)
        self.model_type = 'DDPG'
        self.state_dim = state_dim
        self.action_dim = 3  # hold, buy, sell
        self.memory = deque(maxlen=100)
        self.buffer_size = 90

        self.gamma = 0.95 # discount factor
        self.is_eval = is_eval
        self.noise = OUNoise(self.action_dim)
        tau = 0.001  # Target network hyperparameter
        learning_rate_actor = 0.1  # learning rate for Actor network
        learning_rate_critic = 0.1  # learning rate for Critic network

        self.actor = ActorNetwork(state_dim, self.action_dim, self.buffer_size, tau, learning_rate_actor, is_eval, model_name)
        self.critic = CriticNetwork(state_dim, self.action_dim, tau, learning_rate_critic)

        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/DDPG_tensorboard', update_freq=90)
        self.tensorboard.set_model(self.critic.model)

    def reset(self):
        self.reset_portfolio()
        self.noise.reset()

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state, t):
        actions = self.actor.model.predict(state, verbose=0)[0]
        if not self.is_eval:
            return self.noise.get_actions(actions, t)
        return actions

    def experience_replay(self):
        # sample random buffer_size long memory
        mini_batch = random.sample(self.memory, self.buffer_size)

        y_batch = []
        for state, actions, reward, next_state, done in mini_batch:
            if not done:
                Q_target_value = self.critic.model_target.predict([next_state, self.actor.model_target.predict(next_state, verbose=0)], verbose=0)
                y = reward + self.gamma * Q_target_value
            else:
                y = reward * np.ones((1, self.action_dim))
            y_batch.append(y)

        y_batch = np.vstack(y_batch)
        states_batch = np.vstack([tup[0] for tup in mini_batch]) # batch_size * state_dim
        actions_batch = np.vstack([tup[1] for tup in mini_batch]) # batch_size * action_dim
        
        # update critic by minimizing the loss
        loss = self.critic.model.train_on_batch([states_batch, actions_batch], y_batch)

        # update actor policy
        with tf.GradientTape() as tape:
            actions = self.actor.model(states_batch)
            critic_values = self.critic.model([states_batch, actions])
            actor_loss = -tf.reduce_mean(critic_values)
        
        actor_grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.model.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        
        # update target networks
        self.actor.train_target()
        self.critic.train_target()
        return loss