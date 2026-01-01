"""
Experiment runner for moral decision-making experiments.

Usage:
    pip install -e .  # Install package in editable mode first
    python -m experiments.experiment_runner
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.agents.moral_agents import create_agent, AdaptiveNeuralAgent
from src.agents.maddpg import MADDPG
from src.metrics.moral_metrics import GreatestGoodBenchmark, PeerPressureAnalyzer
from src.visualization.moral_visualizer import MoralDecisionVisualizer


class MoralExperimentRunner:
    """Run and analyze moral decision-making experiments."""
    
    def __init__(self, 
                 experiment_name: str,
                 num_agents: int = 4,
                 episode_length: int = 100,
                 num_episodes: int = 50,
                 save_dir: str = "results"):
        
        self.experiment_name = experiment_name
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.save_dir = save_dir
        
        # Create save directory
        self.experiment_dir = os.path.join(save_dir, experiment_name, 
                                          datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize components
        self.visualizer = MoralDecisionVisualizer()
        
    def run_experiment(self, 
                      agent_configs: List[Dict],
                      reward_structure: str = "mixed",
                      peer_influence_strength: float = 0.1,
                      use_maddpg: bool = False) -> Dict:
        """Run a complete experiment with specified agent configurations."""
        
        # Initialize environment
        env = MoralDilemmaEnv(
            num_agents=self.num_agents,
            total_resources=100,
            episode_length=self.episode_length,
            reward_structure=reward_structure,
            peer_influence_strength=peer_influence_strength
        )
        
        # Create agents
        agents = {}
        agent_types = {}
        
        for i, config in enumerate(agent_configs):
            agent_id = f"agent_{i}"
            agent = create_agent(
                agent_type=config['type'],
                agent_id=agent_id,
                **config.get('params', {})
            )
            agents[agent_id] = agent
            agent_types[agent_id] = config['type']
        
        # Initialize MADDPG if requested
        maddpg = None
        if use_maddpg:
            obs_dims = [env.observation_spaces[aid].shape[0] for aid in env.agents]
            action_dims = [env.action_spaces[aid].shape[0] for aid in env.agents]
            
            # Set moral weights for MADDPG
            moral_weights = {}
            for agent_id, agent_type in agent_types.items():
                if agent_type == 'utilitarian':
                    moral_weights[agent_id] = 0.8
                elif agent_type == 'deontological':
                    moral_weights[agent_id] = 0.5
                elif agent_type == 'egoist':
                    moral_weights[agent_id] = 0.0
            
            maddpg = MADDPG(
                num_agents=self.num_agents,
                obs_dims=obs_dims,
                action_dims=action_dims,
                agent_ids=env.agents,
                moral_weights=moral_weights
            )
        
        # Initialize metrics
        ggb = GreatestGoodBenchmark(self.num_agents)
        peer_analyzer = PeerPressureAnalyzer()
        
        # Storage for results
        experiment_data = {
            'config': {
                'agent_configs': agent_configs,
                'reward_structure': reward_structure,
                'peer_influence_strength': peer_influence_strength,
                'num_episodes': self.num_episodes,
                'episode_length': self.episode_length
            },
            'episodes': [],
            'metrics_history': [],
            'resources_history': [],
            'actions_history': [],
            'fairness_history': [],
            'cooperation_history': [],
            'influence_events': []
        }
        
        # Training loop
        print(f"Running experiment: {self.experiment_name}")
        
        for episode in tqdm(range(self.num_episodes), desc="Episodes"):
            observations, _ = env.reset()
            episode_data = {
                'rewards': [],
                'actions': [],
                'resources': [],
                'observations': []
            }
            
            for step in range(self.episode_length):
                # Get actions
                if use_maddpg and maddpg:
                    actions = maddpg.act(observations, add_noise=(episode < self.num_episodes * 0.8))
                else:
                    actions = {}
                    for agent_id, obs in observations.items():
                        if isinstance(agents[agent_id], AdaptiveNeuralAgent):
                            actions[agent_id] = agents[agent_id].act(obs)
                        else:
                            actions[agent_id] = agents[agent_id].act(obs)
                
                # Environment step
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Store data
                episode_data['rewards'].append(rewards)
                episode_data['actions'].append({k: float(v[0]) for k, v in actions.items()})
                episode_data['resources'].append({k: v['resources'] for k, v in infos.items()})
                episode_data['observations'].append(observations)
                
                # Update metrics
                ggb.update(
                    actions={k: float(v[0]) for k, v in actions.items()},
                    resources={k: v['resources'] for k, v in infos.items()},
                    rewards=rewards
                )
                
                # Detect peer influence
                if step > 0:
                    influenced = peer_analyzer.detect_influence_event(
                        episode_data['actions'][-2],
                        episode_data['actions'][-1]
                    )
                    if influenced:
                        experiment_data['influence_events'].append({
                            'episode': episode,
                            'step': step,
                            'influenced_agents': influenced
                        })
                
                # Train MADDPG
                if use_maddpg and maddpg and len(episode_data['rewards']) > 1:
                    maddpg.store_transition(
                        observations,
                        actions,
                        rewards,
                        next_observations,
                        {k: v or t for k, (v, t) in zip(terminations.items(), truncations.items())}
                    )
                    
                    if step % 10 == 0:
                        losses = maddpg.train()
                
                observations = next_observations
                
                if any(truncations.values()):
                    break
            
            # Store episode data
            experiment_data['episodes'].append(episode_data)
            
            # Calculate and store episode metrics
            episode_metrics = ggb.calculate_metrics()
            experiment_data['metrics_history'].append({
                'episode': episode,
                'utilitarian_score': episode_metrics.utilitarian_score,
                'fairness_score': episode_metrics.fairness_score,
                'cooperation_index': episode_metrics.cooperation_index,
                'conformity_measure': episode_metrics.conformity_measure,
                'peer_influence_strength': episode_metrics.peer_influence_strength,
                'moral_consistency': episode_metrics.moral_consistency
            })
            
            # Store detailed history for visualization
            if episode % 10 == 0:  # Sample every 10 episodes
                experiment_data['resources_history'].extend(episode_data['resources'])
                experiment_data['actions_history'].extend(episode_data['actions'])
                experiment_data['fairness_history'].extend(ggb.history['fairness'])
                experiment_data['cooperation_history'].extend(ggb.history['cooperation'])
        
        # Final metrics
        final_metrics = ggb.get_summary_stats()
        peer_summary = peer_analyzer.get_influence_summary()
        
        experiment_data['final_metrics'] = final_metrics
        experiment_data['peer_influence_summary'] = peer_summary
        
        # Save results
        self._save_results(experiment_data)
        
        # Generate visualizations
        self._generate_visualizations(experiment_data, agent_types)
        
        # Save models if using MADDPG
        if use_maddpg and maddpg:
            model_dir = os.path.join(self.experiment_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            maddpg.save_models(model_dir)
        
        return experiment_data
    
    def run_peer_pressure_experiment(self) -> Dict:
        """Run specific experiment to study peer pressure effects."""
        print("\n=== Peer Pressure Experiment ===")
        
        experiments = {}
        
        # Experiment 1: No peer influence
        print("\nExperiment 1: No peer influence")
        agent_configs = [
            {'type': 'utilitarian', 'params': {}},
            {'type': 'egoist', 'params': {}},
            {'type': 'deontological', 'params': {'fair_share_rule': 0.25}},
            {'type': 'virtue_ethics', 'params': {}}
        ]
        
        experiments['no_influence'] = self.run_experiment(
            agent_configs=agent_configs,
            reward_structure='mixed',
            peer_influence_strength=0.0
        )
        
        # Experiment 2: Moderate peer influence
        print("\nExperiment 2: Moderate peer influence")
        experiments['moderate_influence'] = self.run_experiment(
            agent_configs=agent_configs,
            reward_structure='mixed',
            peer_influence_strength=0.3
        )
        
        # Experiment 3: Strong peer influence
        print("\nExperiment 3: Strong peer influence")
        experiments['strong_influence'] = self.run_experiment(
            agent_configs=agent_configs,
            reward_structure='mixed',
            peer_influence_strength=0.7
        )
        
        # Compare experiments
        self._compare_experiments(experiments)
        
        return experiments
    
    def run_supervisor_experiment(self) -> Dict:
        """Run experiment with supervisor agent steering the group."""
        print("\n=== Supervisor Steering Experiment ===")
        
        experiments = {}
        
        # Baseline: No supervisor
        print("\nBaseline: No supervisor")
        agent_configs = [
            {'type': 'utilitarian', 'params': {}},
            {'type': 'egoist', 'params': {}},
            {'type': 'egoist', 'params': {}},
            {'type': 'adaptive', 'params': {'obs_dim': 7}}
        ]
        
        experiments['no_supervisor'] = self.run_experiment(
            agent_configs=agent_configs,
            reward_structure='mixed',
            use_maddpg=True
        )
        
        # With cooperative supervisor
        print("\nWith cooperative supervisor")
        agent_configs[3] = {'type': 'supervisor', 'params': {'target_behavior': 'cooperative'}}
        
        experiments['cooperative_supervisor'] = self.run_experiment(
            agent_configs=agent_configs,
            reward_structure='mixed',
            use_maddpg=True
        )
        
        # With competitive supervisor
        print("\nWith competitive supervisor")
        agent_configs[3] = {'type': 'supervisor', 'params': {'target_behavior': 'competitive'}}
        
        experiments['competitive_supervisor'] = self.run_experiment(
            agent_configs=agent_configs,
            reward_structure='mixed',
            use_maddpg=True
        )
        
        # Compare experiments
        self._compare_experiments(experiments)
        
        return experiments
    
    def _save_results(self, experiment_data: Dict):
        """Save experiment results to JSON."""
        # Remove non-serializable data
        save_data = {
            'config': experiment_data['config'],
            'final_metrics': experiment_data['final_metrics'],
            'peer_influence_summary': experiment_data['peer_influence_summary'],
            'metrics_history': experiment_data['metrics_history']
        }
        
        with open(os.path.join(self.experiment_dir, 'results.json'), 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def _generate_visualizations(self, experiment_data: Dict, agent_types: Dict):
        """Generate all visualizations for the experiment."""
        # Resource distribution
        if experiment_data['resources_history']:
            self.visualizer.plot_resource_distribution(
                experiment_data['resources_history'][-self.episode_length:],
                agent_types,
                save_path=os.path.join(self.experiment_dir, 'resource_distribution.png')
            )
        
        # Moral dynamics
        if experiment_data['actions_history']:
            self.visualizer.plot_moral_dynamics(
                experiment_data['actions_history'][-self.episode_length:],
                agent_types,
                save_path=os.path.join(self.experiment_dir, 'moral_dynamics.png')
            )
        
        # Interactive dashboard
        dashboard = self.visualizer.create_interactive_dashboard(
            experiment_data,
            save_path=os.path.join(self.experiment_dir, 'dashboard.html')
        )
    
    def _compare_experiments(self, experiments: Dict[str, Dict]):
        """Generate comparison visualizations across experiments."""
        comparison_dir = os.path.join(self.experiment_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Compare key metrics
        for metric in ['fairness_score', 'cooperation_index', 'conformity_measure']:
            self.visualizer.plot_experiment_comparison(
                experiments,
                metric=metric,
                save_path=os.path.join(comparison_dir, f'{metric}_comparison.png')
            )


if __name__ == "__main__":
    # Example usage
    runner = MoralExperimentRunner(
        experiment_name="maebe_moral_dynamics",
        num_agents=4,
        episode_length=100,
        num_episodes=50
    )
    
    # Run peer pressure experiments
    peer_pressure_results = runner.run_peer_pressure_experiment()
    
    # Run supervisor experiments
    supervisor_results = runner.run_supervisor_experiment()
    
    print("\nExperiments completed! Check the results directory for visualizations and data.")