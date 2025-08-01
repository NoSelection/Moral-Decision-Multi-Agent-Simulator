#!/usr/bin/env python3
"""
Moral-Decision Multi-Agent Simulator
Main entry point for running experiments on moral decision-making in multi-agent systems.
"""

import argparse
import sys
from experiments.experiment_runner import MoralExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description='Run moral decision-making experiments in multi-agent systems'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['peer_pressure', 'supervisor', 'custom', 'all'],
        default='all',
        help='Type of experiment to run'
    )
    
    parser.add_argument(
        '--num_agents',
        type=int,
        default=4,
        help='Number of agents in the simulation'
    )
    
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50,
        help='Number of episodes to run'
    )
    
    parser.add_argument(
        '--episode_length',
        type=int,
        default=100,
        help='Length of each episode'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='moral_dynamics_experiment',
        help='Name for the experiment (used for saving results)'
    )
    
    parser.add_argument(
        '--reward_structure',
        type=str,
        choices=['selfish', 'utilitarian', 'mixed'],
        default='mixed',
        help='Reward structure for the environment'
    )
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = MoralExperimentRunner(
        experiment_name=args.experiment_name,
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        num_episodes=args.num_episodes
    )
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          Moral-Decision Multi-Agent Simulator                ║
    ║                                                              ║
    ║  Exploring emergent moral dynamics in multi-agent systems   ║
    ║  Inspired by MAEBE research on moral preferences            ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Configuration:
    - Experiment: {args.experiment}
    - Agents: {args.num_agents}
    - Episodes: {args.num_episodes}
    - Episode Length: {args.episode_length}
    - Reward Structure: {args.reward_structure}
    """)
    
    # Run experiments based on selection
    if args.experiment == 'peer_pressure' or args.experiment == 'all':
        print("\n" + "="*60)
        print("Running Peer Pressure Experiments")
        print("="*60)
        runner.run_peer_pressure_experiment()
    
    if args.experiment == 'supervisor' or args.experiment == 'all':
        print("\n" + "="*60)
        print("Running Supervisor Steering Experiments")
        print("="*60)
        runner.run_supervisor_experiment()
    
    if args.experiment == 'custom':
        print("\n" + "="*60)
        print("Running Custom Experiment")
        print("="*60)
        
        # Example custom configuration
        agent_configs = [
            {'type': 'utilitarian', 'params': {}},
            {'type': 'deontological', 'params': {'fair_share_rule': 0.25}},
            {'type': 'virtue_ethics', 'params': {}},
            {'type': 'adaptive', 'params': {'obs_dim': 7}}
        ]
        
        runner.run_experiment(
            agent_configs=agent_configs,
            reward_structure=args.reward_structure,
            peer_influence_strength=0.3,
            use_maddpg=True
        )
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  Experiments Completed!                      ║
    ║                                                              ║
    ║  Results saved to: results/{args.experiment_name}/          ║
    ║  Open dashboard.html for interactive visualizations         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()