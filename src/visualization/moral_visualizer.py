from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class MoralDecisionVisualizer:
    """Visualization tools for moral decision-making dynamics."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        sns.set_style("whitegrid")
        self.color_map = {
            "utilitarian": "#2E86AB",
            "deontological": "#A23B72",
            "virtue_ethics": "#F18F01",
            "egoist": "#C73E1D",
            "adaptive": "#6A994E",
            "supervisor": "#BC4B51",
        }

    def plot_resource_distribution(
        self,
        resources_history: List[Dict[str, float]],
        agent_types: Dict[str, str],
        save_path: Optional[str] = None,
    ):
        """Plot resource distribution over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # Convert to DataFrame
        df = pd.DataFrame(resources_history)

        # Plot individual agent resources over time
        for agent in df.columns:
            agent_type = agent_types.get(agent, "unknown")
            color = self.color_map.get(agent_type, "gray")
            ax1.plot(df.index, df[agent], label=f"{agent} ({agent_type})", color=color, linewidth=2)

        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Resources")
        ax1.set_title("Individual Agent Resources Over Time")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot fairness (Gini coefficient) over time
        gini_scores = []
        for _, row in df.iterrows():
            values = row.values
            if sum(values) > 0:
                sorted_values = sorted(values)
                n = len(sorted_values)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (
                    n + 1
                ) / n
                gini_scores.append(1 - gini)  # Fairness score
            else:
                gini_scores.append(1.0)

        ax2.plot(gini_scores, color="darkblue", linewidth=2)
        ax2.fill_between(range(len(gini_scores)), gini_scores, alpha=0.3)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Fairness Score")
        ax2.set_title("Fairness (Resource Equality) Over Time")
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_moral_dynamics(
        self,
        actions_history: List[Dict[str, float]],
        agent_types: Dict[str, str],
        save_path: Optional[str] = None,
    ):
        """Plot moral decision dynamics and peer influence."""
        fig = plt.figure(figsize=(14, 10))

        # Convert to DataFrame
        df = pd.DataFrame(actions_history)

        # 1. Action heatmap
        ax1 = plt.subplot(3, 2, 1)
        sns.heatmap(df.T, cmap="YlOrRd", cbar_kws={"label": "Claim Fraction"}, ax=ax1)
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Agent")
        ax1.set_title("Agent Claims Heatmap")

        # 2. Average claims by agent type
        ax2 = plt.subplot(3, 2, 2)
        type_claims = {agent_type: [] for agent_type in set(agent_types.values())}

        for agent, agent_type in agent_types.items():
            if agent in df.columns:
                type_claims[agent_type].append(df[agent].mean())

        avg_by_type = {k: np.mean(v) if v else 0 for k, v in type_claims.items()}
        bars = ax2.bar(avg_by_type.keys(), avg_by_type.values())

        for bar, agent_type in zip(bars, avg_by_type.keys()):
            bar.set_color(self.color_map.get(agent_type, "gray"))

        ax2.set_ylabel("Average Claim Fraction")
        ax2.set_title("Average Claims by Agent Type")
        ax2.set_xticklabels(avg_by_type.keys(), rotation=45)

        # 3. Conformity over time
        ax3 = plt.subplot(3, 2, 3)
        conformity_scores = []

        for _, row in df.iterrows():
            variance = row.var()
            conformity = 1 - (variance / 0.25)  # Max variance for [0,1] is 0.25
            conformity_scores.append(max(0, conformity))

        ax3.plot(conformity_scores, color="purple", linewidth=2)
        ax3.fill_between(range(len(conformity_scores)), conformity_scores, alpha=0.3)
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Conformity Score")
        ax3.set_title("Group Conformity Over Time")

        # 4. Peer influence network
        ax4 = plt.subplot(3, 2, 4)
        self._plot_influence_network(df, agent_types, ax4)

        # 5. Moral consistency
        ax5 = plt.subplot(3, 2, 5)
        window = min(5, len(df))

        for agent in df.columns:
            rolling_std = df[agent].rolling(window).std()
            consistency = 1 - (rolling_std * 2).clip(0, 1)
            agent_type = agent_types.get(agent, "unknown")
            ax5.plot(
                consistency,
                label=f"{agent} ({agent_type})",
                color=self.color_map.get(agent_type, "gray"),
                alpha=0.7,
            )

        ax5.set_xlabel("Timestep")
        ax5.set_ylabel("Consistency Score")
        ax5.set_title("Moral Consistency Over Time")
        ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # 6. Group vs Individual trade-off
        ax6 = plt.subplot(3, 2, 6)
        group_welfare = df.mean(axis=1)
        individual_variance = df.var(axis=1)

        ax6_twin = ax6.twinx()
        ax6.plot(group_welfare, color="green", linewidth=2, label="Group Welfare")
        ax6_twin.plot(individual_variance, color="red", linewidth=2, label="Individual Variance")

        ax6.set_xlabel("Timestep")
        ax6.set_ylabel("Group Welfare", color="green")
        ax6_twin.set_ylabel("Individual Variance", color="red")
        ax6.set_title("Group Welfare vs Individual Differences")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def _plot_influence_network(self, df: pd.DataFrame, agent_types: Dict[str, str], ax):
        """Plot peer influence network."""
        # Calculate correlation matrix
        corr_matrix = df.corr()

        # Create network graph
        G = nx.Graph()

        for agent in df.columns:
            G.add_node(agent, agent_type=agent_types.get(agent, "unknown"))

        # Add edges for strong correlations
        threshold = 0.5
        for i, agent1 in enumerate(df.columns):
            for j, agent2 in enumerate(df.columns):
                if i < j and abs(corr_matrix.iloc[i, j]) > threshold:
                    G.add_edge(agent1, agent2, weight=abs(corr_matrix.iloc[i, j]))

        # Draw network
        pos = nx.spring_layout(G, seed=42)

        # Draw nodes
        for agent in G.nodes():
            agent_type = G.nodes[agent]["agent_type"]
            color = self.color_map.get(agent_type, "gray")
            nx.draw_networkx_nodes(G, pos, nodelist=[agent], node_color=color, node_size=500, ax=ax)

        # Draw edges
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title("Peer Influence Network")
        ax.axis("off")

    def create_interactive_dashboard(self, experiment_data: Dict, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Resource Distribution",
                "Moral Metrics Over Time",
                "Agent Claims",
                "Fairness vs Cooperation",
                "Peer Influence Events",
                "Final Moral Scores",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # 1. Resource distribution
        resources_df = pd.DataFrame(experiment_data["resources_history"])
        for agent in resources_df.columns:
            fig.add_trace(
                go.Scatter(x=resources_df.index, y=resources_df[agent], name=agent, mode="lines"),
                row=1,
                col=1,
            )

        # 2. Moral metrics
        metrics_df = pd.DataFrame(experiment_data["metrics_history"])
        for metric in ["fairness_score", "cooperation_index", "conformity_measure"]:
            if metric in metrics_df.columns:
                fig.add_trace(
                    go.Scatter(x=metrics_df.index, y=metrics_df[metric], name=metric, mode="lines"),
                    row=1,
                    col=2,
                )

        # 3. Agent claims over time
        actions_df = pd.DataFrame(experiment_data["actions_history"])
        for agent in actions_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=actions_df.index,
                    y=actions_df[agent],
                    name=f"{agent}_claims",
                    mode="lines",
                    visible="legendonly",
                ),
                row=2,
                col=1,
            )

        # 4. Fairness vs Cooperation scatter
        if "fairness_history" in experiment_data and "cooperation_history" in experiment_data:
            fig.add_trace(
                go.Scatter(
                    x=experiment_data["fairness_history"],
                    y=experiment_data["cooperation_history"],
                    mode="markers",
                    name="Fairness vs Cooperation",
                    marker=dict(
                        size=8,
                        color=list(range(len(experiment_data["fairness_history"]))),
                        colorscale="Viridis",
                        showscale=True,
                    ),
                ),
                row=2,
                col=2,
            )

        # 5. Peer influence events
        if "influence_events" in experiment_data:
            influence_counts = {}
            for event in experiment_data["influence_events"]:
                for agent in event["influenced_agents"]:
                    influence_counts[agent] = influence_counts.get(agent, 0) + 1

            if influence_counts:
                fig.add_trace(
                    go.Bar(
                        x=list(influence_counts.keys()),
                        y=list(influence_counts.values()),
                        name="Influence Count",
                    ),
                    row=3,
                    col=1,
                )

        # 6. Final moral scores
        if "final_metrics" in experiment_data:
            metrics = experiment_data["final_metrics"]
            fig.add_trace(
                go.Bar(x=list(metrics.keys()), y=list(metrics.values()), name="Moral Scores"),
                row=3,
                col=2,
            )

        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Moral Decision-Making Multi-Agent Simulation Dashboard",
            hovermode="x unified",
        )

        # Update axes
        fig.update_xaxes(title_text="Timestep", row=1, col=1)
        fig.update_xaxes(title_text="Timestep", row=1, col=2)
        fig.update_xaxes(title_text="Timestep", row=2, col=1)
        fig.update_xaxes(title_text="Fairness Score", row=2, col=2)
        fig.update_xaxes(title_text="Agent", row=3, col=1)
        fig.update_xaxes(title_text="Metric", row=3, col=2)

        fig.update_yaxes(title_text="Resources", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_yaxes(title_text="Claim Fraction", row=2, col=1)
        fig.update_yaxes(title_text="Cooperation Index", row=2, col=2)
        fig.update_yaxes(title_text="Influence Count", row=3, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=2)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_experiment_comparison(
        self,
        experiments: Dict[str, Dict],
        metric: str = "fairness_score",
        save_path: Optional[str] = None,
    ):
        """Compare multiple experiments."""
        fig, ax = plt.subplots(figsize=self.figsize)

        for exp_name, exp_data in experiments.items():
            if "metrics_history" in exp_data:
                metrics_df = pd.DataFrame(exp_data["metrics_history"])
                if metric in metrics_df.columns:
                    ax.plot(metrics_df.index, metrics_df[metric], label=exp_name, linewidth=2)

        ax.set_xlabel("Timestep")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f'{metric.replace("_", " ").title()} Across Experiments')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
