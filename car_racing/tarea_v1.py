# %%
import os
import random
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import tensorflow as tf
from utils import ExperimentParams, SimpleNet


# %%
# Helper functions
class Agent:
    def __init__(self, experiment_params: ExperimentParams, q_model: SimpleNet, t_model: SimpleNet):
        self.experiment_params = experiment_params
        self.q_model = q_model
        self.t_model = t_model

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=experiment_params.learning_rate)
        self.loss_function = tf.keras.losses.Huber()

    def select_action(self, state, model: SimpleNet):
        """Select an action using an epsilon-greedy policy.

        This policy works as follows:

        - With probability epsilon, pick a random action
        - With probability 1 - epsilon, pick the best action

        Args:
            state (tf.Tensor): Tensor representing the current state, shape (batch_size, height,
                width, channels).
            model (SimpleNet): Model used to select an action.

        Returns:
            tuple: The selected action and epsilon.
        """
        self.experiment_params.epsilon *= self.experiment_params.epsilon_decay()
        epsilon = max(self.experiment_params.epsilon_min, self.experiment_params.epsilon)

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action_values = model(state, training=False)
            action = tf.argmax(action_values, axis=1).numpy().item()
        return action, self.experiment_params.epsilon

    def train(self, samples: list):
        """Train the agent on the given batch of experiences.

        The training loop implements the following steps:
        1. Calculate updated Q values for target and main Q models (`t_model` and `q_model`).
        2. Select the best action using the main Q model.
        3. Update the Q values for the selected action using the target model.
        4. Estimate the loss based on updated Q values from target model and actions from
            main Q model.
        5. Update the weights of the main Q model using the loss.

        Args:
            samples (list): Batch of tuples containing the state, action, reward, next state, and
                not_done values, extracted from memory.

        Returns:
            float: The loss value.
        """
        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = process_samples(
            samples
        )
        masks = create_action_masks(action_batch, _NUM_ACTIONS)

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_target_actions = self.t_model(next_state_batch)
        future_q_actions = self.q_model(next_state_batch)

        # Select the best action using q_model
        best_future_action = tf.argmax(future_q_actions, axis=-1)
        next_action_mask = create_action_masks(best_future_action, _NUM_ACTIONS)
        future_q_action = tf.reduce_sum(
            tf.multiply(future_target_actions, next_action_mask), axis=1
        )

        # Update Q value = reward + discount factor * expected future reward
        updated_q_values = reward_batch + experiment_params.gamma * tf.multiply(
            future_q_action, not_done_batch
        )

        with tf.GradientTape() as tape:
            q_values = self.q_model(state_batch)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(q_action, updated_q_values)

            # Backpropagation
            grads = tape.gradient(loss, self.q_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))

        return loss.numpy()

    def learning_transfer(self) -> None:
        """
        Transfer the learning from the main model to the target model using soft update.

        Returns:
            None
        """
        weights = q_model.get_weights()
        target_weights = t_model.get_weights()
        for j in range(len(target_weights)):
            target_weights[j] = weights[j] * self.experiment_params.tau + target_weights[j] * (
                1 - self.experiment_params.tau
            )
        t_model.set_weights(target_weights)

    def evaluate(self, episodes: int = 5):
        """
        Evaluate the agent's performance by running multiple episodes of the environment.

        Args:
            episodes (int, optional): The number of episodes to run. Defaults to 5.

        Returns:
            tuple: A tuple containing the mean episode reward and the evaluation memory.
                - mean_episode_reward (float): The mean reward obtained from running the episodes.
                - eval_memory (list): A list of episodes, where each episode is a list of tuples
                    containing the state, action, reward, next state, and done values.
        """
        agent_rewards = []
        eval_memory = []
        for _ in range(episodes):
            state, _ = env.reset()
            state = preprocess_state(state)

            episode_reward = 0
            for time_step in range(1, experiment_params.max_steps_per_episode):
                action_values = self.q_model(state, training=False)
                action = tf.argmax(action_values, axis=1).numpy().item()

                # Apply the sampled action in our environment
                next_state, reward, done, _, _ = env.step(action)
                next_state = preprocess_state(next_state)

                eval_memory.append([state, action, reward, next_state, done])
                episode_reward += reward
                state = next_state

                if done:
                    break

            agent_rewards.append(episode_reward)
        mean_episode_reward = np.mean(agent_rewards)

        return mean_episode_reward, eval_memory


def process_samples(samples):
    """Process a batch of samples and concatenate the state, action, reward, next state,
    and not done tensors. State batch is preprocessed using the preprocess_state function.

    Args:
        samples (list): A list of tuples containing the state, action, reward, next state, and not
        done values.

    Returns:
        tuple: A tuple containing the concatenated state, action, reward, next state, and not done tensors.
    """
    state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = zip(*samples)
    state_batch = tf.concat(state_batch, axis=0)
    action_batch = tf.stack(action_batch, axis=0)
    reward_batch = tf.stack(reward_batch, axis=0)
    next_state_batch = tf.concat(next_state_batch, axis=0)
    not_done_batch = tf.stack([float(not done) for done in not_done_batch], axis=0)
    return state_batch, action_batch, reward_batch, next_state_batch, not_done_batch


def create_action_masks(action_batch, num_actions):
    """
    Create a mask for the actions in the batch.

    This function creates a mask tensor that indicates which actions were taken in the batch.
    Each element in the mask corresponds to an action in the batch and has a value of 1 if the
    action was taken and 0 otherwise. This mask is used to calculate the loss on the updated
    Q-values. In the case of the discrete Car Racing environment there are 5 possible actions, so
    the mask has a shape of (batch_size, 5).


    Args:
        action_batch (tf.Tensor): Tensor representing the actions taken in the batch, shape (batch_size,).
        num_actions (int): The number of actions in the action space.

    Returns:
        tf.Tensor: The mask tensor, shape (batch_size, num_actions).
    """
    return tf.one_hot(action_batch, num_actions)


def log_step_info(epoch: int, episode_train_reward: float, episode_eval_reward: float, loss: float):
    """Log episode information to Tensorboard.

    **Note:** Rewards are estimated at the end of each episode, therefore they are constant for
    each step within an episode.

    Args:
        epoch (int): Step to which the information pertains, each episode is composed of different
            epochs.
        episode_train_reward (float): Training reward of the episode.
        episode_eval_reward (float): Evaluation reward of the episode.
        loss (float): Loss of the step.
    """
    with summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)
        tf.summary.scalar("ema_reward_train", episode_train_reward, step=epoch)
        tf.summary.scalar("ema_reward_eval", episode_eval_reward, step=epoch)
        tf.summary.scalar("epsilon", epsilon, step=epoch)


def update_cumulative_rewards(
    cum_train_reward: float,
    cum_eval_reward: float,
    eval_rewards: float,
    episode_reward: float,
    experiment_params: ExperimentParams,
) -> tuple:
    """
    Update the cumulative training and evaluation rewards based on the current episode reward.

    Parameters:
        cum_train_reward (float): The cumulative training reward.
        cum_eval_reward (float): The cumulative evaluation reward.
        eval_rewards (float): The evaluation rewards obtained from the current episode.
        episode_reward (float): The reward obtained from the current episode.
        experiment_params (ExperimentParams): The experiment parameters.

    Returns:
        tuple: A tuple containing the updated cumulative training reward and the updated cumulative evaluation reward.
    """
    if cum_train_reward is None:
        cum_train_reward = episode_reward

    if cum_eval_reward is None:
        cum_eval_reward = eval_rewards

    cum_train_reward = (
        experiment_params.ema_ratio * episode_reward
        + (1 - experiment_params.ema_ratio) * cum_train_reward
    )
    cum_eval_reward = (
        experiment_params.ema_ratio * eval_rewards
        + (1 - experiment_params.ema_ratio) * cum_eval_reward
    )
    return cum_train_reward, cum_eval_reward


def preprocess_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    state = np.expand_dims(state, axis=-1)
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    state = tf.expand_dims(state, 0)
    return state / 255.0


# %%
# Tensorboard config
implementation = "DQN_v1"
train_log_dir = os.path.join("logs", "RacingCar", implementation)
summary_writer = tf.summary.create_file_writer(train_log_dir)

env = gym.make("CarRacing-v2", continuous=False)

_NUM_ACTIONS = env.action_space.n
_EPISODE_LOG_FREQ = 10
_STOPPING_REWARD_CRITERIA = 200

experiment_params = ExperimentParams(
    max_steps_per_episode=100,
    ema_ratio=0.01,
    training_ratio=4,
    batch_size=32,
    mem_size=4096,
    gamma=0.99,
    epsilon=1,
    epsilon_min=0.056,
    approx_iterations=1000,
    tau=0.125,
    learning_rate=0.05,
)

observation_dim = env.observation_space.shape
input_dim = observation_dim[0] * observation_dim[1] * observation_dim[2]
hidden_units = input_dim // 2
output_dim = _NUM_ACTIONS

q_model = SimpleNet(output_dim)
t_model = SimpleNet(output_dim)

agent = Agent(experiment_params, q_model, t_model)
memory = deque(maxlen=experiment_params.mem_size)
cum_train_reward = None
cum_eval_reward = None
episode_count = 0
epoch = 0
historic_reward = []
while True:  # Run until solved

    state, info = env.reset()
    state = preprocess_state(state)

    episode_reward = 0
    for time_step in range(1, experiment_params.max_steps_per_episode):
        # env.render()

        action, epsilon = agent.select_action(state, q_model)

        # Apply the sampled action in our environment
        next_state, step_reward, done, _, info = env.step(action)
        next_state = preprocess_state(next_state)

        memory.append([state, action, step_reward, next_state, done])
        episode_reward += step_reward

        # Train the agent on the sampled batch of experiences
        if (
            len(memory) >= 2 * experiment_params.batch_size
            and time_step % experiment_params.training_ratio == 0
        ):
            samples = random.sample(memory, experiment_params.batch_size)
            loss = agent.train(samples)
            agent.learning_transfer()

            if cum_train_reward is not None:
                log_step_info(epoch, cum_train_reward, cum_eval_reward, loss)
                epoch += 1
        state = next_state

        if done:
            break

    eval_rewards, eval_memories = agent.evaluate()
    memory.extend(eval_memories)

    cum_train_reward, cum_eval_reward = update_cumulative_rewards(
        cum_train_reward, cum_eval_reward, eval_rewards, episode_reward, experiment_params
    )

    historic_reward.append(cum_train_reward)

    episode_count += 1
    if episode_count % _EPISODE_LOG_FREQ == 0 and "loss" in locals():
        template = "running reward: {:.2f} at episode {} with epsilon {:.2f} and loss {:.2f}"
        print(template.format(cum_eval_reward, episode_count, epsilon, loss))

    # Condition to consider the task solved
    if cum_eval_reward > _STOPPING_REWARD_CRITERIA:
        print("Solved at episode {}!".format(episode_count))
        break


# %%
from gymnasium.wrappers import RecordVideo

env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
# record_env = RecordVideo(env, video_folder="./videos", disable_logger=True)
episodes = 10

agent_rewards = []
for env_i in range(episodes):
    state, info = env.reset()
    episode_reward = 0

    for time_step in range(1, experiment_params.max_steps_per_episode):
        print("Episode: {}, TimeStep: {}".format(env_i, time_step))
        env.render()  # Show the attempts of the agent in a pop up window.

        state = preprocess_state(state)

        action_values = t_model(state, training=False)
        action = tf.argmax(action_values, axis=1).numpy().item()

        # Apply the sampled action in our environment
        state, reward, done, _, info = env.step(action)
        episode_reward += reward

        if done:
            break
    agent_rewards.append(episode_reward)

# %%
env.close()
