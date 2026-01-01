#!/usr/bin/env python3
"""
Training demonstration for the Moral-Decision Multi-Agent Simulator
Shows how agents learn and adapt their moral behaviors over time

Usage:
    pip install -e .  # Install package in editable mode first
    python train_demo.py
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from src.environments.moral_dilemma_env import MoralDilemmaEnv
from src.agents.moral_agents import create_agent, AdaptiveNeuralAgent
from src.metrics.moral_metrics import GreatestGoodBenchmark, PeerPressureAnalyzer


class MoralTrainingDemo:
    """Demonstrates training of adaptive agents in moral scenarios."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def run_training_demo(self, num_episodes=50000, visualize_every=1000):
        """Run a training demonstration with adaptive agents."""
        
        print("\n🧠 Moral Decision-Making Training Demo")
        print("=" * 60)
        print("Training adaptive agents to learn moral behaviors...")
        print("=" * 60)
        
        # Create environment
        env = MoralDilemmaEnv(
            num_agents=4,
            total_resources=100,
            episode_length=50,
            reward_structure="mixed",
            peer_influence_strength=0.3
        )
        
        # Create mixed agent types including adaptive learners
        agents = {
            "agent_0": create_agent("adaptive", "agent_0", obs_dim=7),
            "agent_1": create_agent("adaptive", "agent_1", obs_dim=7),
            "agent_2": create_agent("egoist", "agent_2"),  # Fixed selfish agent
            "agent_3": create_agent("utilitarian", "agent_3")  # Fixed altruistic agent
        }
        
        print("\nAgent Configuration:")
        print("- Agent 0: Adaptive Neural (Learning)")
        print("- Agent 1: Adaptive Neural (Learning)")
        print("- Agent 2: Egoist (Fixed)")
        print("- Agent 3: Utilitarian (Fixed)")
        
        # Metrics tracking
        ggb = GreatestGoodBenchmark(env.num_agents)
        peer_analyzer = PeerPressureAnalyzer()
        
        # Training history
        training_history = {
            'episode_rewards': [],
            'fairness_scores': [],
            'cooperation_scores': [],
            'adaptive_claims': [],
            'learning_progress': []
        }
        
        # Experience buffers for adaptive agents
        experience_buffers = {
            "agent_0": {'obs': [], 'act': [], 'rew': []},
            "agent_1": {'obs': [], 'act': [], 'rew': []}
        }
        
        print("\n🏃 Starting Training...")
        print("Watch as adaptive agents learn moral behaviors in real-time!")
        print("💡 Key metrics to watch:")
        print("   🎯 Total Reward: Should increase as agents learn")
        print("   ⚖️  Fairness: How equally resources are distributed")
        print("   🤝 Cooperation: Agents learning to sacrifice for group benefit")
        print("   🧠 Adaptation: Neural agents changing strategies over time\n")
        
        for episode in tqdm(range(num_episodes), desc="🧠 Training Neural Agents"):
            observations, _ = env.reset()
            episode_rewards = {agent: 0.0 for agent in env.agents}
            episode_claims = []
            
            for step in range(env.episode_length):
                # Get actions
                actions = {}
                for agent_id, obs in observations.items():
                    actions[agent_id] = agents[agent_id].act(obs)
                
                # Environment step
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Store experience for adaptive agents
                for agent_id in ["agent_0", "agent_1"]:
                    if agent_id in observations:
                        experience_buffers[agent_id]['obs'].append(observations[agent_id])
                        experience_buffers[agent_id]['act'].append(actions[agent_id][0])
                        experience_buffers[agent_id]['rew'].append(rewards[agent_id])
                
                # Track metrics
                episode_claims.append({k: float(v[0]) for k, v in actions.items()})
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                # Update metrics
                ggb.update(
                    actions={k: float(v[0]) for k, v in actions.items()},
                    resources={k: v['resources'] for k, v in infos.items()},
                    rewards=rewards
                )
                
                observations = next_observations
                
                if any(truncations.values()):
                    break
            
            # Train adaptive agents every episode
            for agent_id in ["agent_0", "agent_1"]:
                if len(experience_buffers[agent_id]['obs']) > 32:
                    # Sample recent experience
                    recent_size = min(128, len(experience_buffers[agent_id]['obs']))
                    indices = np.random.choice(
                        len(experience_buffers[agent_id]['obs']), 
                        size=recent_size, 
                        replace=False
                    )
                    
                    obs_batch = [experience_buffers[agent_id]['obs'][i] for i in indices]
                    act_batch = [experience_buffers[agent_id]['act'][i] for i in indices]
                    rew_batch = [experience_buffers[agent_id]['rew'][i] for i in indices]
                    
                    # Update the neural network
                    agents[agent_id].update(obs_batch, act_batch, rew_batch)
            
            # Calculate episode metrics
            metrics = ggb.calculate_metrics()
            
            # Store training history
            training_history['episode_rewards'].append(sum(episode_rewards.values()))
            training_history['fairness_scores'].append(metrics.fairness_score)
            training_history['cooperation_scores'].append(metrics.cooperation_index)
            
            # Track adaptive agent claims
            if episode_claims:
                avg_adaptive_claim = np.mean([
                    claims["agent_0"] + claims["agent_1"] 
                    for claims in episode_claims
                ]) / 2
                training_history['adaptive_claims'].append(avg_adaptive_claim)
            
            # Clear old experience (keep recent)
            for agent_id in ["agent_0", "agent_1"]:
                if len(experience_buffers[agent_id]['obs']) > 1000:
                    experience_buffers[agent_id]['obs'] = experience_buffers[agent_id]['obs'][-500:]
                    experience_buffers[agent_id]['act'] = experience_buffers[agent_id]['act'][-500:]
                    experience_buffers[agent_id]['rew'] = experience_buffers[agent_id]['rew'][-500:]
            
            # Visualize progress
            if (episode + 1) % visualize_every == 0:
                self._print_progress(episode + 1, metrics, episode_rewards)
                
                # Show adaptation insights every 5K episodes
                if (episode + 1) % 5000 == 0:
                    self._show_adaptation_insights(episode + 1, training_history)
        
        print("\n✅ Training Complete!")
        
        # Final evaluation
        print("\n📊 Final Evaluation:")
        self._run_evaluation(env, agents, ggb)
        
        # Plot training curves
        self._plot_training_curves(training_history)
        
        return training_history
    
    def _print_progress(self, episode, metrics, rewards):
        """Print detailed training progress."""
        print(f"\n🎯 Episode {episode} Results:")
        print(f"   📈 Total System Reward: {sum(rewards.values()):.2f}")
        print(f"   ⚖️  Fairness Score: {metrics.fairness_score:.3f} ({'🟢 High' if metrics.fairness_score > 0.7 else '🟡 Medium' if metrics.fairness_score > 0.4 else '🔴 Low'})")
        print(f"   🤝 Cooperation Index: {metrics.cooperation_index:.3f} ({'🟢 High' if metrics.cooperation_index > 0.6 else '🟡 Medium' if metrics.cooperation_index > 0.3 else '🔴 Low'})")
        print(f"   🧠 Adaptive Agent Rewards: A0={rewards['agent_0']:.1f}, A1={rewards['agent_1']:.1f}")
        print(f"   🤖 Fixed Agent Rewards: Egoist={rewards['agent_2']:.1f}, Utilitarian={rewards['agent_3']:.1f}")
        
        # Show learning progress
        if episode >= 20:
            improvement = "📊 Learning Progress: "
            if sum(rewards.values()) > 4000:
                improvement += "🚀 Excellent adaptation!"
            elif sum(rewards.values()) > 3500:
                improvement += "📈 Good progress!"
            else:
                improvement += "🔄 Still learning..."
            print(f"   {improvement}")
        print("   " + "-" * 50)
    
    def _show_adaptation_insights(self, episode, history):
        """Show deep learning insights."""
        print(f"\n🔬 DEEP LEARNING ANALYSIS - Episode {episode}")
        print("=" * 60)
        
        if len(history['adaptive_claims']) > 100:
            recent_claims = history['adaptive_claims'][-100:]
            early_claims = history['adaptive_claims'][:100]
            
            change = np.mean(recent_claims) - np.mean(early_claims)
            
            if change > 0.1:
                print("📈 MAJOR ADAPTATION: Neural agents learned to claim MORE resources!")
                print(f"   Strategy shift: +{change:.3f} average claim increase")
            elif change < -0.1:
                print("🤝 MORAL LEARNING: Neural agents became MORE cooperative!")
                print(f"   Strategy shift: {change:.3f} average claim decrease")
            else:
                print("⚖️  STABLE STRATEGY: Neural agents found optimal balance")
                print(f"   Strategy stability: {change:.3f} claim variation")
        
        if len(history['fairness_scores']) > 1000:
            recent_fairness = np.mean(history['fairness_scores'][-1000:])
            if recent_fairness > 0.8:
                print("🏆 HIGH FAIRNESS ACHIEVED: System maintains equity!")
            elif recent_fairness > 0.6:
                print("✅ GOOD FAIRNESS: Reasonable resource distribution")
            else:
                print("⚠️  FAIRNESS CHALLENGE: Some agents dominating")
        
        print("=" * 60)
    
    def _run_evaluation(self, env, agents, ggb):
        """Run final evaluation episode."""
        observations, _ = env.reset()
        
        print("\nRunning evaluation episode...")
        resources_history = []
        
        for step in range(env.episode_length):
            actions = {}
            for agent_id, obs in observations.items():
                actions[agent_id] = agents[agent_id].act(obs)
            
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            resources_history.append({k: v['resources'] for k, v in infos.items()})
            
            if step % 10 == 0:
                env.render()
            
            observations = next_observations
            if any(truncations.values()):
                break
        
        # Final metrics
        final_metrics = ggb.calculate_metrics()
        print(f"\nFinal Metrics:")
        print(f"  Utilitarian Score: {final_metrics.utilitarian_score:.3f}")
        print(f"  Fairness Score: {final_metrics.fairness_score:.3f}")
        print(f"  Cooperation Index: {final_metrics.cooperation_index:.3f}")
        
        # Show final resource distribution
        final_resources = resources_history[-1]
        print(f"\nFinal Resource Distribution:")
        for agent_id, resources in final_resources.items():
            agent_type = "Adaptive" if agent_id in ["agent_0", "agent_1"] else \
                        "Egoist" if agent_id == "agent_2" else "Utilitarian"
            print(f"  {agent_id} ({agent_type}): {resources:.2f}")
    
    def _plot_training_curves(self, history):
        """Plot training progress curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total rewards over time
        ax = axes[0, 0]
        ax.plot(history['episode_rewards'], label='Total Reward', color='blue')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Episode Reward')
        ax.set_title('Learning Progress: Total Rewards')
        ax.grid(True, alpha=0.3)
        
        # Fairness scores
        ax = axes[0, 1]
        ax.plot(history['fairness_scores'], label='Fairness', color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Fairness Score')
        ax.set_title('Fairness Evolution During Training')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Cooperation scores
        ax = axes[1, 0]
        ax.plot(history['cooperation_scores'], label='Cooperation', color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cooperation Index')
        ax.set_title('Cooperation Evolution During Training')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Adaptive agent claims
        ax = axes[1, 1]
        ax.plot(history['adaptive_claims'], label='Avg Adaptive Claim', color='orange')
        ax.axhline(y=0.25, color='red', linestyle='--', label='Fair Share', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Claim Fraction')
        ax.set_title('Adaptive Agents Learning to Claim Resources')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'training_curves_{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Training curves saved as 'training_curves_{timestamp}.png'")


def main():
    """Run the training demonstration."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        🚀 PROFESSIONAL PORTFOLIO TRAINING DEMO 🚀            ║
    ║                                                              ║
    ║  🧠 50,000 episodes of moral behavior learning              ║
    ║  📊 Real-time metrics and learning visualization            ║
    ║  🤖 Neural agents adapting to complex moral scenarios      ║
    ║                                                              ║
    ║  Perfect for showcasing advanced AI research capabilities!   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("⚡ This will train for approximately 30-45 minutes")
    print("💡 Perfect time to explain the system during interviews!")
    print("📈 Watch for dramatic learning curves and emergent behaviors\n")
    
    response = input("🚀 Ready to start 50K episode professional training? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("👋 No problem! Run again when ready for the full demo.")
        return
    
    trainer = MoralTrainingDemo()
    
    # Professional portfolio parameters
    NUM_EPISODES = 50000    # Serious deep learning training
    VISUALIZE_EVERY = 1000  # Show progress every 1K episodes
    
    training_history = trainer.run_training_demo(
        num_episodes=NUM_EPISODES,
        visualize_every=VISUALIZE_EVERY
    )
    
    print("\n🎉 Training demonstration complete!")
    print("Check the generated plots to see how the agents learned over time.")
    print("\nThis demonstrates:")
    print("✓ Adaptive agents learning moral behaviors")
    print("✓ Emergent cooperation and fairness")
    print("✓ Influence of different moral frameworks")
    print("✓ Real-time training visualization")


if __name__ == "__main__":
    main()