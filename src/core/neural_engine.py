#!/usr/bin/env python3
"""
Neural Engine Module
Consolidated neural network integration for financial analysis and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import random
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class NeuralEngineConfig:
    """Configuration for neural engine components"""
    # PDF Extraction
    use_neural_extractor: bool = True
    neural_extractor_model: str = "microsoft/layoutlmv3-base"
    extraction_confidence_threshold: float = 0.7
    
    # Mesh Surrogate
    use_neural_mesh_surrogate: bool = True
    mesh_surrogate_type: str = "mlp"  # "mlp" or "gnn"
    mesh_surrogate_path: Optional[str] = None
    
    # Policy Optimization
    use_neural_policy_optimizer: bool = True
    policy_optimizer_type: str = "actor_critic"  # "actor_critic" or "dqn"
    policy_optimizer_path: Optional[str] = None
    
    # Training
    enable_training: bool = True
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100

@dataclass
class Experience:
    """Experience for reinforcement learning"""
    state: Dict[str, float]
    action: Dict[str, float]
    reward: float
    next_state: Dict[str, float]
    done: bool

class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic policy network
    Actor: π(s) -> action distribution
    Critic: V(s) -> state value
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 5, 
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (action, value)
        """
        features = self.feature_net(state)
        action = self.actor(features)
        value = self.critic(features)
        
        return action, value
    
    def get_action(self, state: torch.Tensor, exploration: float = 0.1) -> torch.Tensor:
        """
        Get action with exploration noise
        
        Args:
            state: State tensor
            exploration: Exploration noise level
            
        Returns:
            Action tensor
        """
        action, _ = self.forward(state)
        
        # Add exploration noise
        if exploration > 0:
            noise = torch.randn_like(action) * exploration
            action = action + noise
        
        return action

class DQNPolicy(nn.Module):
    """
    Deep Q-Network for discrete action spaces
    Q(s, a) -> Q-value for each action
    """
    
    def __init__(self, state_dim: int = 10, num_actions: int = 10, 
                 hidden_dim: int = 128):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Q-values for all actions [batch_size, num_actions]
        """
        return self.q_network(state)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: State tensor
            epsilon: Exploration probability
            
        Returns:
            Action index
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.forward(state)
            return q_values.argmax().item()

class NeuralPolicyOptimizer:
    """
    Neural policy optimizer using reinforcement learning
    Can replace derivative-free optimizers with learned policies
    """
    
    def __init__(self, policy_type: str = "actor_critic", 
                 state_dim: int = 10, action_dim: int = 5,
                 learning_rate: float = 0.001, gamma: float = 0.99):
        """
        Initialize neural policy optimizer
        
        Args:
            policy_type: "actor_critic" or "dqn"
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimization
            gamma: Discount factor for future rewards
        """
        self.policy_type = policy_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Initialize policy network
        if policy_type == "actor_critic":
            self.policy = ActorCriticPolicy(state_dim, action_dim)
        elif policy_type == "dqn":
            self.policy = DQNPolicy(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Target network for DQN
        if policy_type == "dqn":
            self.target_policy = DQNPolicy(state_dim, action_dim)
            self.target_policy.load_state_dict(self.policy.state_dict())
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Optimization setup
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        if policy_type == "dqn":
            self.target_policy.to(self.device)
        
        logger.info(f"✅ Initialized {policy_type.upper()} policy optimizer on {self.device}")
    
    def optimize_policy(self, current_state: Dict[str, float], 
                       risk_preference: str = "moderate") -> Dict[str, float]:
        """
        Optimize policy for given state
        
        Args:
            current_state: Current financial state
            risk_preference: Risk preference level
            
        Returns:
            Optimized action dictionary
        """
        # Convert state to tensor
        state_tensor = self._dict_to_tensor(current_state)
        state_tensor = state_tensor.to(self.device)
        
        # Get action from policy
        if self.policy_type == "actor_critic":
            action_tensor = self.policy.get_action(state_tensor, exploration=0.1)
        else:  # DQN
            action_idx = self.policy.get_action(state_tensor, epsilon=0.1)
            action_tensor = self._index_to_action(action_idx)
        
        # Convert to dictionary
        action_dict = self._tensor_to_dict(action_tensor.cpu().numpy())
        
        # Adjust based on risk preference
        action_dict = self._adjust_for_risk_preference(action_dict, risk_preference)
        
        logger.info(f"✅ Policy optimization complete for {risk_preference} risk preference")
        return action_dict
    
    def train_on_experience(self, experiences: List[Experience], 
                           batch_size: int = 32, epochs: int = 10):
        """
        Train policy on collected experiences
        
        Args:
            experiences: List of experiences
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        logger.info(f"🔄 Training {self.policy_type.upper()} policy on {len(experiences)} experiences")
        
        # Add experiences to replay buffer
        for exp in experiences:
            self.replay_buffer.append(exp)
        
        if len(self.replay_buffer) < batch_size:
            logger.warning("Not enough experiences for training")
            return
        
        # Training loop
        for epoch in range(epochs):
            # Sample batch
            batch = random.sample(self.replay_buffer, batch_size)
            
            if self.policy_type == "actor_critic":
                self._train_actor_critic(batch)
            else:
                self._train_dqn(batch)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} completed")
        
        # Update target network for DQN
        if self.policy_type == "dqn":
            self.target_policy.load_state_dict(self.policy.state_dict())
        
        logger.info("✅ Policy training completed")
    
    def _train_actor_critic(self, batch: List[Experience]):
        """Train Actor-Critic policy"""
        
        states = torch.stack([self._dict_to_tensor(exp.state) for exp in batch]).to(self.device)
        actions = torch.stack([self._dict_to_tensor(exp.action) for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([self._dict_to_tensor(exp.next_state) for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).to(self.device)
        
        # Forward pass
        action_pred, value_pred = self.policy(states)
        _, next_value_pred = self.policy(next_states)
        
        # Compute loss
        # Actor loss (policy gradient)
        action_loss = F.mse_loss(action_pred, actions)
        
        # Critic loss (value function)
        target_values = rewards + self.gamma * next_value_pred.squeeze() * (1 - dones)
        value_loss = F.mse_loss(value_pred.squeeze(), target_values)
        
        # Total loss
        total_loss = action_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def _train_dqn(self, batch: List[Experience]):
        """Train DQN policy"""
        
        states = torch.stack([self._dict_to_tensor(exp.state) for exp in batch]).to(self.device)
        actions = torch.stack([self._action_to_index(exp.action) for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([self._dict_to_tensor(exp.next_state) for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_policy(next_states)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def _dict_to_tensor(self, d: Dict[str, float]) -> torch.Tensor:
        """Convert dictionary to tensor"""
        values = list(d.values())[:self.state_dim]
        while len(values) < self.state_dim:
            values.append(0.0)
        return torch.tensor(values[:self.state_dim], dtype=torch.float32)
    
    def _tensor_to_dict(self, tensor: np.ndarray) -> Dict[str, float]:
        """Convert tensor to dictionary"""
        action_keys = ["invest", "save", "spend", "borrow", "repay"]
        d = {}
        for i, key in enumerate(action_keys):
            if i < len(tensor):
                d[key] = float(tensor[i])
            else:
                d[key] = 0.0
        return d
    
    def _action_to_index(self, action: Dict[str, float]) -> torch.Tensor:
        """Convert action dictionary to index for DQN"""
        # Simplified conversion - in practice you'd have a proper mapping
        action_values = list(action.values())
        max_index = action_values.index(max(action_values))
        return torch.tensor([max_index], dtype=torch.long)
    
    def _index_to_action(self, action_idx: int) -> torch.Tensor:
        """Convert action index to tensor for DQN"""
        action = torch.zeros(self.action_dim)
        action[action_idx] = 1.0
        return action
    
    def _adjust_for_risk_preference(self, action: Dict[str, float], risk_preference: str) -> Dict[str, float]:
        """Adjust action based on risk preference"""
        if risk_preference == "conservative":
            # Increase saving and debt reduction
            action["save"] *= 1.2
            action["repay"] *= 1.2
            action["invest"] *= 0.8
        elif risk_preference == "aggressive":
            # Increase investing
            action["invest"] *= 1.3
            action["save"] *= 0.9
        # moderate stays the same
        
        # Normalize to sum to 1
        total = sum(action.values())
        if total > 0:
            action = {k: v / total for k, v in action.items()}
        
        return action
    
    def save_policy(self, path: str):
        """Save policy to file"""
        torch.save(self.policy.state_dict(), path)
        logger.info(f"✅ Policy saved to {path}")
    
    def load_policy(self, path: str):
        """Load policy from file"""
        if Path(path).exists():
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"✅ Policy loaded from {path}")
        else:
            logger.warning(f"⚠️ Policy file not found: {path}")

class NeuralFinancialEngine:
    """Main neural financial engine that integrates all neural components"""
    
    def __init__(self, config: NeuralEngineConfig):
        """Initialize neural financial engine"""
        self.config = config
        self.policy_optimizer = None
        
        # Initialize policy optimizer if enabled
        if config.use_neural_policy_optimizer:
            self.policy_optimizer = NeuralPolicyOptimizer(
                policy_type=config.policy_optimizer_type,
                state_dim=10,
                action_dim=5,
                learning_rate=config.learning_rate
            )
            
            # Load pre-trained model if available
            if config.policy_optimizer_path and Path(config.policy_optimizer_path).exists():
                self.policy_optimizer.load_policy(config.policy_optimizer_path)
        
        logger.info("✅ Neural Financial Engine initialized")
    
    def optimize_policy(self, current_state: Dict[str, float], 
                       risk_preference: str = "moderate") -> Dict[str, float]:
        """
        Optimize policy for current financial state
        
        Args:
            current_state: Current financial state
            risk_preference: Risk preference level
            
        Returns:
            Optimized action dictionary
        """
        if self.policy_optimizer:
            return self.policy_optimizer.optimize_policy(current_state, risk_preference)
        else:
            # Fallback to simple heuristics
            return self._fallback_policy(current_state, risk_preference)
    
    def _fallback_policy(self, current_state: Dict[str, float], 
                         risk_preference: str) -> Dict[str, float]:
        """Fallback policy when neural optimization is not available"""
        
        # Simple heuristic-based policy
        cash = current_state.get("cash", 0)
        investments = current_state.get("investments", 0)
        debt = current_state.get("debt", 0)
        income = current_state.get("income", 0)
        age = current_state.get("age", 35)
        
        # Calculate basic allocations
        total_assets = cash + investments
        
        if risk_preference == "conservative":
            return {
                "invest": 0.2,
                "save": 0.4,
                "spend": 0.1,
                "borrow": 0.0,
                "repay": 0.3
            }
        elif risk_preference == "aggressive":
            return {
                "invest": 0.5,
                "save": 0.2,
                "spend": 0.1,
                "borrow": 0.0,
                "repay": 0.2
            }
        else:  # moderate
            return {
                "invest": 0.3,
                "save": 0.3,
                "spend": 0.1,
                "borrow": 0.0,
                "repay": 0.3
            }
    
    def train_on_experiences(self, experiences: List[Experience], **kwargs):
        """Train neural components on experiences"""
        if self.policy_optimizer:
            self.policy_optimizer.train_on_experience(experiences, **kwargs)
        else:
            logger.warning("⚠️ No policy optimizer available for training")
    
    def save_models(self, base_path: str = "data/outputs/neural_models"):
        """Save all neural models"""
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        if self.policy_optimizer:
            policy_path = Path(base_path) / "policy_optimizer.pth"
            self.policy_optimizer.save_policy(str(policy_path))
        
        # Save configuration
        config_path = Path(base_path) / "neural_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info(f"✅ Neural models saved to {base_path}")
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get metadata about neural models"""
        metadata = {
            "config": self.config.__dict__,
            "policy_optimizer": None
        }
        
        if self.policy_optimizer:
            metadata["policy_optimizer"] = {
                "type": self.policy_optimizer.policy_type,
                "state_dim": self.policy_optimizer.state_dim,
                "action_dim": self.policy_optimizer.action_dim,
                "device": str(self.policy_optimizer.device)
            }
        
        return metadata

def main():
    """Main function for testing"""
    # Create configuration
    config = NeuralEngineConfig(
        use_neural_policy_optimizer=True,
        policy_optimizer_type="actor_critic",
        enable_training=False
    )
    
    # Initialize neural engine
    engine = NeuralFinancialEngine(config)
    
    # Test optimization
    current_state = {
        "cash": 10000,
        "investments": 50000,
        "debt": 20000,
        "income": 80000,
        "age": 35,
        "risk_tolerance": 0.6
    }
    
    action = engine.optimize_policy(current_state, "moderate")
    print(f"Optimized action: {action}")
    
    # Get metadata
    metadata = engine.get_model_metadata()
    print(f"Model metadata: {json.dumps(metadata, indent=2, default=str)}")

if __name__ == "__main__":
    main() 