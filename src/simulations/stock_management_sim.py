"""
Stock Management Simulation - Comparing MPC Planner vs RL Agent.

This module provides simulation functions to compare the performance of:
1. Model Predictive Control (MPC) - a model-based planning approach
2. Proximal Policy Optimization (PPO) - a model-free RL approach

Both agents learn to optimize inventory management for perishable goods.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..envs.stock_management import StockManagementEnv, StockManagementConfig, ItemConfig
from ..controllers.mpc import MPCController
from ..agents.ppo import PPOAgent
from ..utils.seeding import set_global_seeds


@dataclass
class SimulationConfig:
    """Configuration for comparison simulation.

    Attributes:
        n_episodes: Number of episodes to run for each agent.
        seed: Random seed for reproducibility.
        env_config: Stock management environment configuration.
        verbose: Print progress during simulation.
    """

    n_episodes: int = 50
    seed: int = 42
    env_config: Optional[StockManagementConfig] = None
    verbose: bool = True


class StockManagementMPC:
    """MPC-based controller for stock management.

    Wraps the generic MPC controller with stock-management-specific dynamics
    and cost functions. Uses the environment's model for planning.
    """

    def __init__(
        self,
        env: StockManagementEnv,
        horizon: int = 10,
        max_iter: int = 50,
    ):
        """Initialize the MPC controller.

        Args:
            env: Stock management environment instance.
            horizon: Planning horizon (number of steps to look ahead).
            max_iter: Maximum optimizer iterations.
        """
        self.env = env
        self.horizon = horizon
        self.n_items = len(env.config.items)

        # Get dynamics and cost functions from environment
        dynamics_fn = env.get_dynamics_function()
        cost_fn = env.get_cost_function()

        # State and control dimensions
        state_dim = 3 * self.n_items + 1  # inv, age, demand for each item + storage_util
        control_dim = self.n_items  # order quantity for each item

        # Control bounds
        control_low = np.zeros(control_dim)
        control_high = np.full(control_dim, env.config.max_order)

        # State bounds (normalized [0, 1])
        state_low = np.zeros(state_dim)
        state_high = np.ones(state_dim)

        # Reference state: target inventory levels
        # Aim for ~50% storage utilization and moderate inventory
        x_ref = np.zeros(state_dim)
        for i in range(self.n_items):
            x_ref[3 * i] = 0.4  # 40% of max inventory
            x_ref[3 * i + 1] = 0.3  # 30% of decay time
            x_ref[3 * i + 2] = 0.5  # demand at mean level
        x_ref[-1] = 0.5  # 50% storage utilization

        # Cost matrices (penalize deviation from reference and control effort)
        Q = np.eye(state_dim) * 10.0  # State cost
        R = np.eye(control_dim) * 0.1  # Control cost (purchase is part of profit)

        self.mpc = MPCController(
            dynamics_fn=dynamics_fn,
            horizon=horizon,
            state_dim=state_dim,
            control_dim=control_dim,
            dt=1.0,
            Q=Q,
            R=R,
            Q_f=Q * 2,  # Terminal cost
            x_ref=x_ref,
            state_bounds=(state_low, state_high),
            control_bounds=(control_low, control_high),
            cost_fn=cost_fn,
            max_iter=max_iter,
            warm_start=True,
            name="StockMPC",
        )

    def select_action(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        """Select order quantities using MPC.

        Args:
            state: Current observation.
            explore: Not used for MPC (deterministic planning).

        Returns:
            Order quantities for each item.
        """
        return self.mpc.compute_control(state)

    def reset(self) -> None:
        """Reset the MPC controller."""
        self.mpc.reset()


def run_mpc_simulation(
    env: StockManagementEnv,
    mpc: StockManagementMPC,
    n_episodes: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run simulation with MPC controller.

    Args:
        env: Stock management environment.
        mpc: MPC controller.
        n_episodes: Number of episodes to simulate.
        verbose: Print progress.

    Returns:
        Dictionary with simulation results.
    """
    episode_rewards = []
    episode_infos = []

    for ep in range(n_episodes):
        state = env.reset()
        mpc.reset()

        total_reward = 0.0
        ep_info = {
            "revenues": [],
            "costs": [],
            "spoilage": [],
            "stockouts": [],
        }

        done = False
        while not done:
            action = mpc.select_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            ep_info["revenues"].append(info["revenue"])
            ep_info["costs"].append(
                info["purchase_cost"] + info["holding_cost"] +
                info["spoilage_cost"] + info["stockout_cost"]
            )
            ep_info["spoilage"].append(sum(info["spoiled"]))
            ep_info["stockouts"].append(sum(info["stockouts"]))

            state = next_state

        episode_rewards.append(total_reward)
        episode_infos.append(ep_info)

        if verbose and (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  MPC Episode {ep + 1:3d} | Avg Reward: {avg_reward:8.2f}")

    return {
        "episode_rewards": episode_rewards,
        "episode_infos": episode_infos,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }


def run_ppo_simulation(
    env: StockManagementEnv,
    n_episodes_train: int = 100,
    n_episodes_eval: int = 10,
    verbose: bool = True,
) -> Tuple[PPOAgent, Dict[str, Any]]:
    """Train and evaluate PPO agent on stock management.

    Args:
        env: Stock management environment.
        n_episodes_train: Number of training episodes.
        n_episodes_eval: Number of evaluation episodes.
        verbose: Print progress.

    Returns:
        Tuple of (trained agent, evaluation results).
    """
    n_items = len(env.config.items)
    state_dim = 3 * n_items + 1
    action_dim = n_items

    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=0.0,
        action_high=float(env.config.max_order),
        hidden_dims=(128, 128),
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        trajectory_length=env.config.episode_length or 100,
        ppo_epochs=10,
        batch_size=64,
        name="StockPPO",
    )

    # Training phase
    if verbose:
        print("\n  Training PPO agent...")

    training_rewards = []

    for ep in range(n_episodes_train):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)

            agent.store(state, action, reward, next_state, done)

            if agent.ready_to_train():
                metrics = agent.update()

            state = next_state
            total_reward += reward

        training_rewards.append(total_reward)

        if verbose and (ep + 1) % 20 == 0:
            avg_reward = np.mean(training_rewards[-20:])
            print(f"    Train Episode {ep + 1:3d} | Avg Reward: {avg_reward:8.2f}")

    # Evaluation phase
    if verbose:
        print("\n  Evaluating PPO agent...")

    agent.eval()
    episode_rewards = []
    episode_infos = []

    for ep in range(n_episodes_eval):
        state = env.reset()
        total_reward = 0.0
        ep_info = {
            "revenues": [],
            "costs": [],
            "spoilage": [],
            "stockouts": [],
        }

        done = False
        while not done:
            action = agent.select_action(state, explore=False)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            ep_info["revenues"].append(info["revenue"])
            ep_info["costs"].append(
                info["purchase_cost"] + info["holding_cost"] +
                info["spoilage_cost"] + info["stockout_cost"]
            )
            ep_info["spoilage"].append(sum(info["spoiled"]))
            ep_info["stockouts"].append(sum(info["stockouts"]))

            state = next_state

        episode_rewards.append(total_reward)
        episode_infos.append(ep_info)

        if verbose:
            print(f"    Eval Episode {ep + 1:3d} | Reward: {total_reward:8.2f}")

    results = {
        "episode_rewards": episode_rewards,
        "episode_infos": episode_infos,
        "training_rewards": training_rewards,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }

    return agent, results


def run_comparison(
    config: Optional[SimulationConfig] = None,
) -> Dict[str, Any]:
    """Run comparison between MPC planner and PPO RL agent.

    Args:
        config: Simulation configuration.

    Returns:
        Dictionary with comparison results for both methods.
    """
    config = config or SimulationConfig()

    # Set seeds for reproducibility
    set_global_seeds(config.seed)

    # Create environment
    env_config = config.env_config or StockManagementConfig(
        items=[
            ItemConfig(name="fresh_produce", decay_time=3, size=2.0,
                       purchase_cost=2.0, sell_price=5.0, demand_mean=8.0),
            ItemConfig(name="dairy", decay_time=5, size=1.0,
                       purchase_cost=1.5, sell_price=3.0, demand_mean=15.0),
            ItemConfig(name="frozen", decay_time=10, size=1.5,
                       purchase_cost=3.0, sell_price=6.0, demand_mean=5.0),
        ],
        storage_capacity=100.0,
        max_inventory=50,
        max_order=20,
        holding_cost_rate=0.05,
        spoilage_penalty=1.0,
        stockout_penalty=0.5,
        episode_length=100,
    )

    if config.verbose:
        print("=" * 60)
        print("Stock Management: MPC vs PPO Comparison")
        print("=" * 60)
        print(f"\nEnvironment Configuration:")
        print(f"  Items: {len(env_config.items)}")
        for item in env_config.items:
            print(f"    - {item.name}: decay={item.decay_time}, "
                  f"price=${item.sell_price:.2f}, demand={item.demand_mean:.1f}")
        print(f"  Storage capacity: {env_config.storage_capacity}")
        print(f"  Episode length: {env_config.episode_length}")
        print()

    # Create environments (separate for MPC and PPO)
    env_mpc = StockManagementEnv(config=env_config, seed=config.seed)
    env_ppo = StockManagementEnv(config=env_config, seed=config.seed)

    # Run MPC simulation
    if config.verbose:
        print("Running MPC Planner...")

    mpc = StockManagementMPC(env_mpc, horizon=10, max_iter=50)
    mpc_results = run_mpc_simulation(
        env_mpc, mpc,
        n_episodes=config.n_episodes,
        verbose=config.verbose,
    )

    # Run PPO simulation (with training)
    if config.verbose:
        print("\nRunning PPO Agent...")

    ppo_agent, ppo_results = run_ppo_simulation(
        env_ppo,
        n_episodes_train=config.n_episodes * 2,  # More training episodes
        n_episodes_eval=config.n_episodes,
        verbose=config.verbose,
    )

    # Summary
    if config.verbose:
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"\nMPC Planner:")
        print(f"  Mean Reward: {mpc_results['mean_reward']:8.2f} +/- {mpc_results['std_reward']:.2f}")

        print(f"\nPPO Agent (after training):")
        print(f"  Mean Reward: {ppo_results['mean_reward']:8.2f} +/- {ppo_results['std_reward']:.2f}")

        # Detailed breakdown
        mpc_avg_spoilage = np.mean([
            np.sum(info["spoilage"]) for info in mpc_results["episode_infos"]
        ])
        ppo_avg_spoilage = np.mean([
            np.sum(info["spoilage"]) for info in ppo_results["episode_infos"]
        ])
        mpc_avg_stockouts = np.mean([
            np.sum(info["stockouts"]) for info in mpc_results["episode_infos"]
        ])
        ppo_avg_stockouts = np.mean([
            np.sum(info["stockouts"]) for info in ppo_results["episode_infos"]
        ])

        print(f"\nDetailed Metrics (per episode):")
        print(f"  {'Metric':<20} {'MPC':>12} {'PPO':>12}")
        print(f"  {'-' * 44}")
        print(f"  {'Avg Spoilage':.<20} {mpc_avg_spoilage:>12.1f} {ppo_avg_spoilage:>12.1f}")
        print(f"  {'Avg Stockouts':.<20} {mpc_avg_stockouts:>12.1f} {ppo_avg_stockouts:>12.1f}")

        # Determine winner
        if mpc_results["mean_reward"] > ppo_results["mean_reward"]:
            diff = mpc_results["mean_reward"] - ppo_results["mean_reward"]
            print(f"\n  MPC outperforms PPO by {diff:.2f} reward per episode")
        else:
            diff = ppo_results["mean_reward"] - mpc_results["mean_reward"]
            print(f"\n  PPO outperforms MPC by {diff:.2f} reward per episode")

    return {
        "mpc": mpc_results,
        "ppo": ppo_results,
        "config": {
            "n_episodes": config.n_episodes,
            "seed": config.seed,
            "n_items": len(env_config.items),
        },
    }


def main():
    """Run the stock management comparison simulation."""
    config = SimulationConfig(
        n_episodes=20,
        seed=42,
        verbose=True,
    )

    results = run_comparison(config)

    print("\nSimulation complete!")
    return results


if __name__ == "__main__":
    main()
