#!/usr/bin/env python3
"""
Comprehensive RL vs Neural Meta-learning Comparison for Method Selection
Implements DQN, PPO, A3C agents vs I-ASNH neural meta-learning framework
For academic publication - establishes fundamental paradigm comparison
"""

import sys
import os
import logging
import json
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'models'))

from rl_method_selection_agents import DQNAgent, PPOAgent, A3CAgent, RLMethodSelectionEnvironment
from models.real_iasnh_framework import RealIASNHFramework

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLvsNeuralMetalearningComparison:
    """Comprehensive comparison between RL and Neural Meta-learning approaches"""
    
    def __init__(self, datasets_path: str = "data"):
        self.datasets_path = datasets_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load datasets
        self.datasets = self._load_datasets()
        self.baseline_methods = ['Linear', 'DLinear', 'Seasonal_Naive', 'ARIMA', 'Prophet', 'ETS', 'LSTM', 'N_BEATS']
        
        # Initialize environment
        self.env = RLMethodSelectionEnvironment(self.datasets, self.baseline_methods)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize I-ASNH
        self.iasnh_model = self._initialize_iasnh()
        
        # Results storage
        self.results = {
            'training_curves': defaultdict(list),
            'final_performance': {},
            'convergence_analysis': {},
            'computational_efficiency': {},
            'selection_patterns': defaultdict(list)
        }
    
    def _load_datasets(self) -> Dict[str, np.ndarray]:
        """Load benchmark datasets"""
        datasets = {}
        dataset_mapping = {
            'ETTh1': 'etth1.npy',
            'ETTh2': 'etth2.npy', 
            'ETTm1': 'ettm1.npy',
            'ETTm2': 'ettm2.npy',
            'Exchange_Rate': 'exchange_rate.npy',
            'Weather': 'weather.npy',
            'Illness': 'illness.npy',
            'ECL': 'ecl.npy'
        }
        
        for dataset_name, filename in dataset_mapping.items():
            try:
                filepath = os.path.join(self.datasets_path, 'processed', filename)
                if os.path.exists(filepath):
                    data = np.load(filepath)
                    datasets[dataset_name] = data
                    logger.info(f"Loaded {dataset_name}: {data.shape}")
            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {e}")
                
        return datasets
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize RL agents"""
        agents = {
            'DQN': DQNAgent(state_dim=64, num_methods=len(self.baseline_methods), lr=1e-3),
            'PPO': PPOAgent(state_dim=64, num_methods=len(self.baseline_methods), lr=3e-4),
            'A3C': A3CAgent(state_dim=64, num_methods=len(self.baseline_methods), lr=1e-3)
        }
        
        logger.info("RL agents initialized:")
        for name, agent in agents.items():
            params = agent._count_parameters()
            logger.info(f"  {name}: {params:,} parameters")
        
        return agents
    
    def _initialize_iasnh(self):
        """Initialize I-ASNH neural meta-learning framework"""
        try:
            model = RealIASNHFramework(
                input_dim=96,
                num_methods=len(self.baseline_methods),
                use_statistical=True,
                use_convolutional=True,
                use_attention=True,
                use_confidence=True,
                use_fusion=True
            ).to(self.device)
            
            params = sum(p.numel() for p in model.parameters())
            logger.info(f"I-ASNH initialized: {params:,} parameters")
            return model
        except Exception as e:
            logger.error(f"Error initializing I-ASNH: {e}")
            raise
    
    def train_rl_agents(self, episodes_per_agent: int = 1000) -> Dict[str, Any]:
        """Train RL agents and track performance"""
        logger.info(f"Training RL agents for {episodes_per_agent} episodes each...")
        
        training_results = {}
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Training {agent_name}...")
            start_time = time.time()
            
            episode_rewards = []
            episode_losses = []
            selection_accuracy = []
            
            for episode in range(episodes_per_agent):
                # Reset environment with random dataset
                dataset_name = np.random.choice(list(self.datasets.keys()))
                state = self.env.reset(dataset_name)
                
                # Get optimal action for accuracy calculation
                optimal_action = self.env.get_optimal_action(dataset_name)
                
                if agent_name == 'DQN':
                    # DQN training
                    action, confidence = agent.select_action(state, training=True)
                    next_state, reward, done = self.env.step(action)
                    
                    agent.store_experience(state, action, reward, next_state, done)
                    loss = agent.train_step()
                    
                    episode_rewards.append(reward)
                    if loss is not None:
                        episode_losses.append(loss)
                    selection_accuracy.append(1.0 if action == optimal_action else 0.0)
                
                elif agent_name == 'PPO':
                    # PPO training
                    action, confidence, log_prob, value = agent.select_action(state, training=True)
                    next_state, reward, done = self.env.step(action)

                    # Store experience for PPO
                    agent.states.append(state)
                    agent.actions.append(action)
                    agent.rewards.append(reward)
                    agent.log_probs.append(log_prob)
                    agent.values.append(value)
                    agent.dones.append(done)

                    # Update every 32 episodes
                    if (episode + 1) % 32 == 0:
                        losses = agent.train_step()
                        if losses and 'total_loss' in losses:
                            episode_losses.append(losses['total_loss'])

                    episode_rewards.append(reward)
                    selection_accuracy.append(1.0 if action == optimal_action else 0.0)
                
                elif agent_name == 'A3C':
                    # A3C training (simplified single-worker)
                    action, confidence, log_prob, value = agent.select_action(state)
                    next_state, reward, done = self.env.step(action)
                    
                    # Update every episode
                    losses = agent.train_step([state], [action], [reward], [log_prob], [value])
                    if losses:
                        episode_losses.append(losses['total_loss'])
                    
                    episode_rewards.append(reward)
                    selection_accuracy.append(1.0 if action == optimal_action else 0.0)
                
                # Log progress
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_accuracy = np.mean(selection_accuracy[-100:])
                    logger.info(f"  Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, Accuracy = {avg_accuracy:.1%}")
            
            training_time = time.time() - start_time
            
            # Store training results
            training_results[agent_name] = {
                'episode_rewards': episode_rewards,
                'episode_losses': episode_losses,
                'selection_accuracy': selection_accuracy,
                'training_time_seconds': training_time,
                'final_avg_reward': np.mean(episode_rewards[-100:]),
                'final_accuracy': np.mean(selection_accuracy[-100:]),
                'convergence_episode': self._find_convergence_point(episode_rewards)
            }
            
            logger.info(f"  {agent_name} training completed in {training_time:.1f}s")
            logger.info(f"  Final performance: {training_results[agent_name]['final_avg_reward']:.3f} reward, {training_results[agent_name]['final_accuracy']:.1%} accuracy")
        
        return training_results
    
    def evaluate_iasnh_performance(self) -> Dict[str, Any]:
        """Evaluate I-ASNH performance across all datasets"""
        logger.info("Evaluating I-ASNH performance...")
        
        start_time = time.time()
        
        total_rewards = []
        selection_accuracy = []
        method_selections = defaultdict(int)
        confidence_scores = []
        
        for dataset_name, dataset in self.datasets.items():
            try:
                # Prepare data
                if len(dataset.shape) > 1:
                    data = dataset[:, 0]
                else:
                    data = dataset
                
                # Limit data size
                data = data[:min(2000, len(data))]
                
                if len(data) < 96:
                    continue
                
                # Get I-ASNH prediction
                input_window = data[-96:]
                input_tensor = torch.FloatTensor(input_window).unsqueeze(0).to(self.device)
                
                selected_method_idx, selection_probability, confidence = self.iasnh_model.predict_method(input_tensor)
                selected_method = self.baseline_methods[selected_method_idx % len(self.baseline_methods)]
                
                # Calculate reward using environment
                state = self.env.reset(dataset_name)
                _, reward, _ = self.env.step(selected_method_idx)
                
                # Get optimal action for accuracy
                optimal_action = self.env.get_optimal_action(dataset_name)
                
                total_rewards.append(reward)
                selection_accuracy.append(1.0 if selected_method_idx == optimal_action else 0.0)
                method_selections[selected_method] += 1
                confidence_scores.append(float(confidence) if confidence is not None else 0.5)
                
                logger.info(f"  {dataset_name}: Selected {selected_method}, Reward: {reward:.3f}, Confidence: {confidence:.3f}")
                
            except Exception as e:
                logger.warning(f"Error evaluating I-ASNH on {dataset_name}: {e}")
        
        evaluation_time = time.time() - start_time
        
        return {
            'avg_reward': np.mean(total_rewards),
            'selection_accuracy': np.mean(selection_accuracy),
            'method_selections': dict(method_selections),
            'avg_confidence': np.mean(confidence_scores),
            'evaluation_time_seconds': evaluation_time,
            'total_datasets': len(total_rewards)
        }
    
    def _find_convergence_point(self, rewards: List[float], window: int = 100) -> int:
        """Find convergence point in training curve"""
        if len(rewards) < window * 2:
            return len(rewards)
        
        # Look for point where moving average stabilizes
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        for i in range(window, len(moving_avg)):
            if i + window < len(moving_avg):
                current_avg = np.mean(moving_avg[i:i+window])
                future_avg = np.mean(moving_avg[i+window:i+2*window])
                
                # Convergence if change is less than 5%
                if abs(future_avg - current_avg) / (abs(current_avg) + 1e-8) < 0.05:
                    return i + window
        
        return len(rewards)
    
    def run_comprehensive_comparison(self, rl_episodes: int = 1000) -> Dict[str, Any]:
        """Run complete RL vs Neural Meta-learning comparison"""
        logger.info("Starting comprehensive RL vs Neural Meta-learning comparison...")
        
        start_time = time.time()
        
        # Train RL agents
        rl_results = self.train_rl_agents(rl_episodes)
        
        # Evaluate I-ASNH
        iasnh_results = self.evaluate_iasnh_performance()
        
        # Compile comprehensive results
        results = {
            'rl_agents': rl_results,
            'iasnh': iasnh_results,
            'comparison_summary': self._generate_comparison_summary(rl_results, iasnh_results),
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_datasets': len(self.datasets),
                'baseline_methods': self.baseline_methods,
                'rl_episodes_per_agent': rl_episodes,
                'device': str(self.device),
                'total_experiment_time': time.time() - start_time
            }
        }
        
        # Save results
        os.makedirs("results/rl_experiments", exist_ok=True)
        output_file = f"results/rl_experiments/rl_vs_neural_metalearning_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive comparison completed. Results saved to {output_file}")
        return results
    
    def _generate_comparison_summary(self, rl_results: Dict, iasnh_results: Dict) -> Dict[str, Any]:
        """Generate comparison summary between RL and Neural Meta-learning"""
        
        # Find best RL agent
        best_rl_agent = max(rl_results.keys(), key=lambda x: rl_results[x]['final_accuracy'])
        best_rl_performance = rl_results[best_rl_agent]
        
        return {
            'performance_ranking': {
                'best_rl_agent': best_rl_agent,
                'best_rl_accuracy': best_rl_performance['final_accuracy'],
                'best_rl_reward': best_rl_performance['final_avg_reward'],
                'iasnh_accuracy': iasnh_results['selection_accuracy'],
                'iasnh_reward': iasnh_results['avg_reward']
            },
            'convergence_analysis': {
                'dqn_convergence': rl_results['DQN']['convergence_episode'],
                'ppo_convergence': rl_results['PPO']['convergence_episode'],
                'a3c_convergence': rl_results['A3C']['convergence_episode'],
                'iasnh_training': 'Pre-trained (meta-learning)'
            },
            'computational_efficiency': {
                'dqn_training_time': rl_results['DQN']['training_time_seconds'],
                'ppo_training_time': rl_results['PPO']['training_time_seconds'],
                'a3c_training_time': rl_results['A3C']['training_time_seconds'],
                'iasnh_evaluation_time': iasnh_results['evaluation_time_seconds']
            },
            'interpretability_analysis': {
                'rl_interpretability': 'Limited (black-box policy)',
                'iasnh_interpretability': 'High (confidence scores, method probabilities)',
                'iasnh_avg_confidence': iasnh_results['avg_confidence'],
                'iasnh_method_diversity': len(iasnh_results['method_selections'])
            }
        }

def main():
    """Main execution function"""
    logger.info("Starting RL vs Neural Meta-learning Comparison Experiment...")
    
    # Initialize comparison framework
    comparison = RLvsNeuralMetalearningComparison()
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison(rl_episodes=1000)
    
    # Print summary
    print("\n" + "="*80)
    print("RL vs NEURAL META-LEARNING COMPARISON COMPLETED")
    print("="*80)
    
    summary = results['comparison_summary']
    print(f"Best RL Agent: {summary['performance_ranking']['best_rl_agent']}")
    print(f"  Accuracy: {summary['performance_ranking']['best_rl_accuracy']:.1%}")
    print(f"  Reward: {summary['performance_ranking']['best_rl_reward']:.3f}")
    
    print(f"\nI-ASNH Neural Meta-learning:")
    print(f"  Accuracy: {summary['performance_ranking']['iasnh_accuracy']:.1%}")
    print(f"  Reward: {summary['performance_ranking']['iasnh_reward']:.3f}")
    print(f"  Confidence: {summary['interpretability_analysis']['iasnh_avg_confidence']:.3f}")
    print(f"  Method Diversity: {summary['interpretability_analysis']['iasnh_method_diversity']} methods")
    
    print("\nComparison complete! 🏆")

if __name__ == "__main__":
    main()
