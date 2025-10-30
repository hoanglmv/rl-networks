# agent.py
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from replay_buffer import ReplayBuffer

class TD3Agent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Hyperparameters (theo bài báo)
        self.gamma = 0.99
        self.tau = 5e-3
        self.batch_size = 128
        self.lr = 1e-3
        self.policy_noise = 0.2
        self.noise_clip = 0.1

        # Actor network μθ(s)
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())
        # Critic networks Qφ(s,a)
        self.critic1 = self.build_critic()
        self.critic2 = self.build_critic()
        self.critic_target1 = self.build_critic()
        self.critic_target2 = self.build_critic()
        self.critic_target1.set_weights(self.critic1.get_weights())
        self.critic_target2.set_weights(self.critic2.get_weights())

        # Dummy calls to build the models and instantiate variables
        dummy_state = tf.zeros((1, self.state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, self.action_dim), dtype=tf.float32)
        self.critic1([dummy_state, dummy_action])
        self.critic2([dummy_state, dummy_action])

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, int(1e6))
        self.actor_optimizer = tf.keras.optimizers.Adam(self.lr)
        self.critic_optimizer1 = tf.keras.optimizers.Adam(self.lr)
        self.critic_optimizer2 = tf.keras.optimizers.Adam(self.lr)

    def build_actor(self):
        state_input = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(state_input)
        x = layers.Dense(256, activation='relu')(x)
        out = layers.Dense(self.action_dim, activation='tanh')(x)
        # Chuyển từ [-1,1] sang [0,1]
        output = layers.Lambda(lambda x: (x + 1) / 2)(out)
        return tf.keras.Model(state_input, output)

    def build_critic(self):
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        q = layers.Dense(1)(x)
        return tf.keras.Model([state_input, action_input], q)

    def get_action(self, state, noise=True):
        state = state.reshape(1, -1)
        action = self.actor(state)[0].numpy()
        if noise:
            action += np.random.normal(0, self.policy_noise, size=self.action_dim)
        return np.clip(action, 0, 1)

    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Thêm noise vào hành động mục tiêu (TD3) và clip vào [0,1]
        next_actions = self.actor_target(next_states)
        noise = tf.clip_by_value(tf.random.normal(shape=next_actions.shape, stddev=self.policy_noise),
                                 -self.noise_clip, self.noise_clip)
        next_actions = tf.clip_by_value(next_actions + noise, 0, 1)
        # Tính target Q giá trị: r + γ * min(Q1', Q2')
        target_q1 = self.critic_target1([next_states, next_actions])
        target_q2 = self.critic_target2([next_states, next_actions])
        target_q = rewards + self.gamma * (1 - dones) * tf.minimum(target_q1, target_q2)
        # Cập nhật critic networks
        with tf.GradientTape() as tape:
            current_q1 = self.critic1([states, actions])
            loss1 = tf.reduce_mean((current_q1 - target_q)**2)
        grads1 = tape.gradient(loss1, self.critic1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(grads1, self.critic1.trainable_variables))
        with tf.GradientTape() as tape:
            current_q2 = self.critic2([states, actions])
            loss2 = tf.reduce_mean((current_q2 - target_q)**2)
        grads2 = tape.gradient(loss2, self.critic2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(grads2, self.critic2.trainable_variables))
        # Cập nhật actor policy (dùng gradient của Q từ critic1)
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic1([states, actions_pred]))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        # Cập nhật target networks
        self.update_target(self.actor.variables,   self.actor_target.variables)
        self.update_target(self.critic1.variables, self.critic_target1.variables)
        self.update_target(self.critic2.variables, self.critic_target2.variables)

    def update_target(self, weights, target_weights):
        for (w, tw) in zip(weights, target_weights):
            tw.assign(self.tau * w + (1 - self.tau) * tw)
