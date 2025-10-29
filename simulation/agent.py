
# simulation/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from replay_buffer import ReplayBuffer

# --- Actor and Critic Network Definitions ---

class Actor(nn.Module):
    """Actor (Policy) Network."""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        # Use tanh to output actions in [-1, 1], then scale by max_action
        # The paper mentions a hyperbolic tangent activation function.
        return self.max_action * torch.tanh(self.layer_4(x))

class Critic(nn.Module):
    """Critic (Value) Network."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Critic 1
        self.layer_1 = nn.Linear(state_dim + action_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 1)

        # Critic 2
        self.layer_5 = nn.Linear(state_dim + action_dim, 128)
        self.layer_6 = nn.Linear(128, 64)
        self.layer_7 = nn.Linear(64, 32)
        self.layer_8 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = F.relu(self.layer_3(q1))
        q1 = self.layer_4(q1)

        q2 = F.relu(self.layer_5(sa))
        q2 = F.relu(self.layer_6(q2))
        q2 = F.relu(self.layer_7(q2))
        q2 = self.layer_8(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = F.relu(self.layer_3(q1))
        q1 = self.layer_4(q1)
        return q1

# --- TD3 Agent Implementation ---

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, state_dim, action_dim)
        self.policy_noise = POLICY_NOISE
        self.noise_clip = NOISE_CLIP
        self.tau = TARGET_UPDATE_TAU
        self.discount = DISCOUNT_FACTOR

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=100):
        # Sample replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        # Update actor policy
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = self.critic
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = self.actor
