import random
from collections import deque
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class ExperimentParams:
    """Class for holding experiment parameters.

    :param max_steps_per_episode : Maximum number of steps per episode.
    :param ema_ratio : EMA ratio that controls the update of the reward between episodes.
    :param training_ratio : Ratio between generating experiences and sampling for training.
    :param batch_size : Batch size for sampling tranining experiences.
    :param mem_size : Size of the buffer that stores the experiences.
    :param gamma : Discount factor for future rewards.
    :param epsilon : Exploration rate.
    :param epsilon_min : Minimum exploration rate.
    :param approx_iterations : Number of iterations to reduce exploration rate.
    :param tau : Factor of the EMA that controls the update of the target network.
    :param learning_rate : Controls the update of weights in the NN.
    """

    max_steps_per_episode: int
    ema_ratio: float
    training_ratio: int
    batch_size: int
    mem_size: int
    gamma: float
    epsilon: float
    epsilon_min: float
    approx_iterations: float
    tau: float
    learning_rate: float

    def epsilon_decay(self) -> float:
        """Decay rate after each iteration to reduce exploration rate."""
        return (self.epsilon_min / self.epsilon) ** (1 / self.approx_iterations)


class SimpleNet(tf.keras.Model):
    def __init__(self, out_dim):
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.output_layer = tf.keras.layers.Dense(out_dim)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.output_layer(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
