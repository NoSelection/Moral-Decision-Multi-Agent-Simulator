# Moral-Decision Multi-Agent Simulator

A multi-agent reinforcement learning environment for studying moral decision-making and emergent social dynamics. Inspired by the MAEBE (Multi-Agent Ethically-Aligned Behavior Evaluation) research framework, this system enables researchers to study how different moral frameworks interact in multi-agent settings and how group dynamics influence individual decision-making.

## Overview

This simulator creates an environment where multiple AI agents with different moral frameworks must make decisions about resource allocation. The system enables study of:

- Emergent moral behaviors in multi-agent systems
- Peer pressure and conformity effects on decision-making
- Utilitarian vs deontological reasoning trade-offs
- Social dynamics and group welfare optimization
- Supervisor agents that can influence group behavior

## Key Features

### Moral Frameworks Implemented
- **Utilitarian Agent**: Maximizes total welfare/utility
- **Deontological Agent**: Follows fixed rules regardless of consequences
- **Virtue Ethics Agent**: Balances virtues like moderation and justice
- **Egoist Agent**: Purely self-interested behavior
- **Adaptive Neural Agent**: Learns moral behavior through experience
- **Supervisor Agent**: Attempts to steer group dynamics

### Advanced Capabilities
- **MADDPG Training**: Multi-Agent Deep Deterministic Policy Gradient implementation
- **Greatest Good Benchmark (GGB)**: Comprehensive moral metrics calculation
- **Peer Influence Analysis**: Tracks conformity and social pressure effects
- **Comprehensive Visualizations**: Training curves and behavioral analysis

## Requirements

- **Platform**: Windows, macOS, or Linux
- **Python**: 3.8+ (tested on 3.10)
- **RAM**: 8GB minimum (32GB+ recommended for large-scale experiments)
- **CPU**: Multi-core processor recommended (optimized for 8+ cores)
- **GPU**: Optional (CUDA or MPS support for acceleration)

## Installation

```bash
# Clone the repository
cd Moral-Decision-Multi-Agent-Simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run All Experiments
```bash
python main.py --experiment all
```

### Run Specific Experiments
```bash
# Peer pressure analysis
python main.py --experiment peer_pressure --num_episodes 100

# Supervisor steering experiments
python main.py --experiment supervisor --num_agents 6

# Custom configuration
python main.py --experiment custom --reward_structure utilitarian
```

## Experiments

### 1. Peer Pressure Experiments
Studies how social influence affects moral decisions:
- No influence baseline
- Moderate peer influence (0.3 strength)
- Strong peer influence (0.7 strength)

### 2. Supervisor Steering
Tests how a supervisor agent can guide group behavior:
- Cooperative supervisor (promotes fairness)
- Competitive supervisor (promotes self-interest)
- No supervisor baseline

### 3. Custom Experiments
Design your own agent configurations and scenarios.

## Metrics & Analysis

### Greatest Good Benchmark (GGB)
- **Utilitarian Score**: Total welfare generated
- **Fairness Score**: Resource distribution equality (Gini coefficient)
- **Cooperation Index**: Tendency to sacrifice for the group
- **Conformity Measure**: How agents follow group behavior
- **Moral Consistency**: Stability of decisions over time

### Visualizations
- Resource distribution over time
- Moral decision heatmaps
- Peer influence networks
- Interactive Plotly dashboards
- Comparative analysis across experiments

## Architecture

```
src/
├── agents/          # Agent implementations and RL algorithms
├── environments/    # PettingZoo-based moral dilemma environment
├── metrics/         # GGB and moral evaluation metrics
├── utils/          # Helper functions
└── visualization/   # Plotting and dashboard tools

experiments/
└── experiment_runner.py  # Main experiment orchestration
```

## Research Applications

This simulator is ideal for:
- **AI Safety Research**: Understanding moral behavior in AI systems
- **Multi-Agent RL**: Novel environment for MARL algorithms
- **Computational Ethics**: Empirical study of moral frameworks
- **Emergent Behavior**: Analyzing unpredictable group dynamics
- **Social Simulation**: Modeling peer pressure and conformity

## 📈 Performance Notes

Optimized for Apple
- Leverages unified memory architecture
- Efficient tensor operations with PyTorch
- Parallel environment execution with PettingZoo

## Contributing

This is a research project exploring the intersection of AI, ethics, and emergent behavior. Contributions are welcome!

## References

- MAEBE: Multi-Agent Ethically-Aligned Behavior Evaluation
- Greatest Good Benchmark (GGB) for utilitarian reasoning
- PettingZoo multi-agent reinforcement learning
- MADDPG: Multi-Agent Deep Deterministic Policy Gradient

## Future Directions

- Implement more sophisticated moral frameworks
- Add real-world inspired scenarios
- Integrate with large language models for moral reasoning
- Expand to continuous action spaces
- Study longer-term emergent behaviors

---

Built for exploring the intersection of artificial intelligence and moral philosophy.
