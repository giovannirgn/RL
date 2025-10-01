import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import collections
import random

GAMMA = 0.99
LR = 0.001

BATCH_SIZE = 64
BUFFER_SIZE = 150_000

MIN_REPLAY_SIZE = 500
TAU = 0.1

EPS_START = 1.0
EPS_END=0.01
EPS_DECAY_STEPS= 2_000_000

class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def build_q_network(state_shape, n_actions, hidden_units=(128,128)):
    inputs = keras.Input(shape=state_shape)
    x = inputs
    for h in hidden_units:
        x = layers.Dense(h, activation="relu")(x)
    outputs = layers.Dense(n_actions, activation=None)(x)
    q_net = keras.Model(inputs=inputs, outputs=outputs)
    #keras.utils.plot_model(q_net, "q_netwrk.png")
    return q_net

class DoubleDQNAgent:
    def __init__(self, state_shape, n_actions):

        self.n_actions = n_actions
        self.q_net = build_q_network(state_shape, n_actions)
        self.target_q_net = build_q_network(state_shape, n_actions)
        self.target_q_net.set_weights(self.q_net.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=LR)
        self.loss_fn = keras.losses.MeanSquaredError()
        self.replay_buffer = ReplayBuffer()
        self.total_steps = 0
        self.epsilon = 1.0

    def select_action(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_net(np.array([state], dtype=np.float32))
        return int(tf.argmax(q_values[0]).numpy())

    def update_epsilon(self):
        decay = min(1.0, max(0.0, self.total_steps / EPS_DECAY_STEPS))
        self.epsilon = EPS_START + (EPS_END - EPS_START) * decay

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        next_q_main = self.q_net(next_states)
        next_actions = tf.argmax(next_q_main, axis=1)
        next_q_target = self.target_q_net(next_states)
        max_next_q = tf.gather(next_q_target, next_actions[:, None], batch_dims=1)
        target_q = rewards + (1.0 - dones) * GAMMA * tf.squeeze(max_next_q, axis=1)

        with tf.GradientTape() as tape:
            q_vals = self.q_net(states)
            idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            chosen_q = tf.gather_nd(q_vals, idx)
            loss = self.loss_fn(target_q, chosen_q)
        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def train_from_replay(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        loss = self.train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones.astype(np.float32), dtype=tf.float32)
        )
        return float(loss)

    def soft_update_target(self):
        q_weights = self.q_net.get_weights()
        target_weights = self.target_q_net.get_weights()
        new_weights = [TAU * qw + (1 - TAU) * tw for qw, tw in zip(q_weights, target_weights)]
        self.target_q_net.set_weights(new_weights)

    def save(self, path):
        self.q_net.save_weights(path)

    def load(self, path):
        self.q_net.load_weights(path)
        self.target_q_net.set_weights(self.q_net.get_weights())