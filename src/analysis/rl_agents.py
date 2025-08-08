#!/usr/bin/env python3
"""
Pure RL Agents for Time Series Method Selection
Implements DQN, PPO, and A3C agents as strong baselines for comparison with I-ASNH
For academic publication - comprehensive RL vs Neural Meta-learning comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import logging

# Experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TimeSeriesStateEncoder(nn.Module):
    """Encode time series into state representation for RL agents"""
    
    def __init__(self, input_dim: int = 96, state_dim: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, sequence_length]
        x = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
        conv_out = self.conv_layers(x)  # [batch_size, 64, 16]
        conv_out = conv_out.view(conv_out.size(0), -1)  # [batch_size, 64*16]
        state = self.fc_layers(conv_out)  # [batch_size, state_dim]
        return state

class DQNAgent:
    """Deep Q-Network agent for method selection"""
    
    def __init__(self, state_dim: int = 64, num_methods: int = 8, lr: float = 1e-3):
        self.state_dim = state_dim
        self.num_methods = num_methods
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State encoder
        self.state_encoder = TimeSeriesStateEncoder(state_dim=state_dim).to(self.device)
        
        # Q-networks
        self.q_network = self._build_q_network().to(self.device)
        self.target_network = self._build_q_network().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + list(self.q_network.parameters()), 
            lr=lr
        )
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Training parameters
        self.gamma = 0.95
        self.target_update_freq = 100
        self.steps = 0
        
        logging.info(f"DQN Agent initialized with {self._count_parameters()} parameters")
    
    def _build_q_network(self) -> nn.Module:
        """Build Q-network architecture"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_methods)
        )
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        encoder_params = sum(p.numel() for p in self.state_encoder.parameters())
        q_params = sum(p.numel() for p in self.q_network.parameters())
        return encoder_params + q_params
    
    def select_action(self, time_series: torch.Tensor, training: bool = True) -> Tuple[int, float]:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.num_methods - 1)
            confidence = 0.5  # Random confidence
        else:
            with torch.no_grad():
                # Ensure input is on correct device
                time_series = time_series.to(self.device)
                state = self.state_encoder(time_series.unsqueeze(0))
                q_values = self.q_network(state)
                action = q_values.argmax().item()
                confidence = torch.softmax(q_values, dim=1).max().item()

        return action, confidence
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        actions = torch.tensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch]).to(self.device)
        
        # Encode states
        current_states = self.state_encoder(states)
        next_states_encoded = self.state_encoder(next_states)
        
        # Current Q values
        current_q_values = self.q_network(current_states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states_encoded).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.state_encoder.parameters()) + list(self.q_network.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Update exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

class PPOAgent:
    """Proximal Policy Optimization agent for method selection"""
    
    def __init__(self, state_dim: int = 64, num_methods: int = 8, lr: float = 3e-4):
        self.state_dim = state_dim
        self.num_methods = num_methods
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State encoder
        self.state_encoder = TimeSeriesStateEncoder(state_dim=state_dim).to(self.device)
        
        # Policy and value networks
        self.policy_network = self._build_policy_network().to(self.device)
        self.value_network = self._build_value_network().to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.policy_network.parameters()) + 
            list(self.value_network.parameters()), 
            lr=lr
        )
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Experience storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        logging.info(f"PPO Agent initialized with {self._count_parameters()} parameters")
    
    def _build_policy_network(self) -> nn.Module:
        """Build policy network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_methods),
            nn.Softmax(dim=-1)
        )
    
    def _build_value_network(self) -> nn.Module:
        """Build value network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        encoder_params = sum(p.numel() for p in self.state_encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_network.parameters())
        value_params = sum(p.numel() for p in self.value_network.parameters())
        return encoder_params + policy_params + value_params
    
    def select_action(self, time_series: torch.Tensor, training: bool = True) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        """Select action using policy network"""
        with torch.no_grad():
            # Ensure input is on correct device
            time_series = time_series.to(self.device)
            state = self.state_encoder(time_series.unsqueeze(0))
            action_probs = self.policy_network(state)
            value = self.value_network(state)

            if training:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                confidence = action_probs.max().item()
                return action.item(), confidence, log_prob, value
            else:
                action = action_probs.argmax()
                confidence = action_probs.max().item()
                return action.item(), confidence, None, None
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, 
                        log_prob: torch.Tensor, value: torch.Tensor, done: bool):
        """Store experience for PPO update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        if next_value is None:
            next_value = torch.tensor(0.0).to(self.device)

        # Ensure all values have consistent dimensions (1D tensors)
        values_list = []
        for v in self.values:
            if isinstance(v, torch.Tensor):
                if v.dim() == 0:
                    values_list.append(v.unsqueeze(0))
                elif v.dim() > 1:
                    values_list.append(v.squeeze())
                else:
                    values_list.append(v)
            else:
                # Convert scalar to tensor
                values_list.append(torch.tensor([v], device=self.device))

        # Ensure next_value has correct dimension (1D tensor)
        if isinstance(next_value, torch.Tensor):
            if next_value.dim() == 0:
                next_value = next_value.unsqueeze(0)
            elif next_value.dim() > 1:
                next_value = next_value.squeeze()
        else:
            next_value = torch.tensor([next_value], device=self.device)

        # Only concatenate if we have values
        if len(values_list) > 0:
            values = torch.cat(values_list + [next_value])
        else:
            values = next_value
        rewards = torch.tensor(self.rewards).to(self.device)
        dones = torch.tensor(self.dones).to(self.device)
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + torch.cat(self.values)
        return advantages, returns
    
    def train_step(self, next_value: torch.Tensor = None) -> Dict[str, float]:
        """Perform PPO update"""
        if len(self.states) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.stack(self.states).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # PPO epochs
            # Forward pass
            encoded_states = self.state_encoder(states)
            action_probs = self.policy_network(encoded_states)
            values = self.value_network(encoded_states).squeeze()
            
            # Policy loss
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.state_encoder.parameters()) + 
                list(self.policy_network.parameters()) + 
                list(self.value_network.parameters()), 
                max_norm=0.5
            )
            self.optimizer.step()
        
        # Clear experience
        self.clear_experience()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def clear_experience(self):
        """Clear stored experience"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

class A3CAgent:
    """Advantage Actor-Critic agent for method selection"""
    
    def __init__(self, state_dim: int = 64, num_methods: int = 8, lr: float = 1e-3):
        self.state_dim = state_dim
        self.num_methods = num_methods
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State encoder
        self.state_encoder = TimeSeriesStateEncoder(state_dim=state_dim).to(self.device)
        
        # Actor-Critic networks
        self.actor = self._build_actor_network().to(self.device)
        self.critic = self._build_critic_network().to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 
            lr=lr
        )
        
        # A3C parameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        
        logging.info(f"A3C Agent initialized with {self._count_parameters()} parameters")
    
    def _build_actor_network(self) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_methods),
            nn.Softmax(dim=-1)
        )
    
    def _build_critic_network(self) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        encoder_params = sum(p.numel() for p in self.state_encoder.parameters())
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        return encoder_params + actor_params + critic_params
    
    def select_action(self, time_series: torch.Tensor) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        """Select action using actor network"""
        # Ensure input is on correct device
        time_series = time_series.to(self.device)
        state = self.state_encoder(time_series.unsqueeze(0))
        action_probs = self.actor(state)
        value = self.critic(state)

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        confidence = action_probs.max().item()

        return action.item(), confidence, log_prob, value
    
    def train_step(self, states: List[torch.Tensor], actions: List[int], 
                  rewards: List[float], log_probs: List[torch.Tensor], 
                  values: List[torch.Tensor]) -> Dict[str, float]:
        """Perform A3C update"""
        if len(states) == 0:
            return {}
        
        # Convert to tensors with proper dimension handling
        try:
            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.tensor(actions, device=self.device)
            rewards_tensor = torch.tensor(rewards, device=self.device)

            # Handle log_probs and values with dimension checking
            log_probs_list = []
            values_list = []

            for lp in log_probs:
                if lp.dim() == 0:
                    log_probs_list.append(lp.unsqueeze(0))
                else:
                    log_probs_list.append(lp.squeeze())

            for v in values:
                if v.dim() == 0:
                    values_list.append(v.unsqueeze(0))
                else:
                    values_list.append(v.squeeze())

            log_probs_tensor = torch.cat(log_probs_list).to(self.device)
            values_tensor = torch.cat(values_list).to(self.device)

        except Exception as e:
            # Return empty dict on tensor conversion error
            return {}
        
        # Compute returns
        returns = torch.zeros_like(rewards_tensor)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards_tensor[t] + self.gamma * R
            returns[t] = R
        
        # Compute advantages
        advantages = returns - values_tensor
        
        # Actor loss
        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values_tensor, returns)
        
        # Entropy loss
        encoded_states = self.state_encoder(states_tensor)
        action_probs = self.actor(encoded_states)
        dist = Categorical(action_probs)
        entropy_loss = -dist.entropy().mean()
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.state_encoder.parameters()) + 
            list(self.actor.parameters()) + 
            list(self.critic.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }

def create_rl_agent(agent_type: str, **kwargs):
    """Factory function to create RL agents"""
    if agent_type.lower() == 'dqn':
        return DQNAgent(**kwargs)
    elif agent_type.lower() == 'ppo':
        return PPOAgent(**kwargs)
    elif agent_type.lower() == 'a3c':
        return A3CAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

class RLMethodSelectionEnvironment:
    """Environment for training RL agents on method selection"""

    def __init__(self, datasets: Dict[str, np.ndarray], baseline_methods: List[str]):
        self.datasets = datasets
        self.baseline_methods = baseline_methods
        self.num_methods = len(baseline_methods)
        self.current_dataset = None
        self.current_data = None
        self.method_performances = {}

        # Pre-compute method performances for reward calculation
        self._precompute_performances()

    def _precompute_performances(self):
        """Pre-compute performance of each method on each dataset"""
        from models.baseline_methods import get_all_baseline_methods
        baseline_evaluator = get_all_baseline_methods()

        for dataset_name, data in self.datasets.items():
            self.method_performances[dataset_name] = {}

            # Use first column if multivariate
            if len(data.shape) > 1:
                time_series = data[:, 0]
            else:
                time_series = data

            # Limit data size for efficiency
            time_series = time_series[:min(2000, len(time_series))]

            for method_name in self.baseline_methods:
                try:
                    if method_name in baseline_evaluator:
                        # Simple evaluation
                        split_idx = int(0.8 * len(time_series))
                        train_data = time_series[:split_idx]
                        test_data = time_series[split_idx:]

                        if len(test_data) >= 10:
                            # Simple forecasting (placeholder - would use actual methods)
                            if method_name == 'Linear':
                                predictions = np.linspace(train_data[-1],
                                                        train_data[-1] + (train_data[-1] - train_data[0]) / len(train_data),
                                                        len(test_data))
                            elif method_name == 'Seasonal_Naive':
                                season_length = min(24, len(train_data) // 4)
                                predictions = np.tile(train_data[-season_length:],
                                                    (len(test_data) // season_length) + 1)[:len(test_data)]
                            else:
                                predictions = np.full(len(test_data), np.mean(train_data))

                            # Calculate MASE
                            mae = np.mean(np.abs(predictions - test_data))
                            naive_mae = np.mean(np.abs(test_data[1:] - test_data[:-1]))
                            mase = mae / (naive_mae + 1e-8)

                            self.method_performances[dataset_name][method_name] = mase
                        else:
                            self.method_performances[dataset_name][method_name] = 1.5
                    else:
                        self.method_performances[dataset_name][method_name] = 1.5
                except:
                    self.method_performances[dataset_name][method_name] = 1.5

    def reset(self, dataset_name: str = None) -> torch.Tensor:
        """Reset environment with a dataset"""
        if dataset_name is None:
            dataset_name = np.random.choice(list(self.datasets.keys()))

        self.current_dataset = dataset_name
        data = self.datasets[dataset_name]

        # Use first column if multivariate
        if len(data.shape) > 1:
            self.current_data = data[:, 0]
        else:
            self.current_data = data

        # Return last 96 points as state
        if len(self.current_data) >= 96:
            state = torch.FloatTensor(self.current_data[-96:])
        else:
            # Pad if necessary
            padded = np.pad(self.current_data, (96 - len(self.current_data), 0), mode='edge')
            state = torch.FloatTensor(padded)

        return state

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """Take action and return next state, reward, done"""
        if self.current_dataset is None:
            raise ValueError("Environment not reset")

        # Get method name
        method_name = self.baseline_methods[action % len(self.baseline_methods)]

        # Get performance (MASE)
        mase = self.method_performances[self.current_dataset].get(method_name, 1.5)

        # Convert MASE to reward (lower MASE = higher reward)
        # Reward in range [-1, 1] where 1 is best possible
        reward = max(-1.0, min(1.0, 2.0 - mase))

        # Episode is done after one step (single method selection)
        done = True

        # Next state is same (single-step episodes)
        next_state = torch.FloatTensor(self.current_data[-96:]) if len(self.current_data) >= 96 else torch.FloatTensor(np.pad(self.current_data, (96 - len(self.current_data), 0), mode='edge'))

        return next_state, reward, done

    def get_optimal_action(self, dataset_name: str = None) -> int:
        """Get optimal action for current or specified dataset"""
        if dataset_name is None:
            dataset_name = self.current_dataset

        if dataset_name not in self.method_performances:
            return 0

        # Find method with lowest MASE
        best_method = min(self.method_performances[dataset_name].items(), key=lambda x: x[1])[0]

        try:
            return self.baseline_methods.index(best_method)
        except ValueError:
            return 0
