"""
Training Loop Orchestrator for the Ab Initio RL Simulation System.

Provides:
- Factory functions for environments and agents
- Standardized training loop
- Command-line interface
"""

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import (
    AgentConfig,
    EnvConfig,
    PRESETS,
    TrainingConfig,
)
from .envs import (
    HomeostasisConfig,
    HomeostasisEnv,
    ServerLoadConfig,
    ServerLoadEnv,
    SmartGridConfig,
    SmartGridEnv,
)
from .agents import (
    BaseAgent,
    DQNAgent,
    LinUCBAgent,
    MCTSAgent,
    PPOAgent,
)
from .utils.logger import Logger
from .utils.seeding import set_global_seeds


def make_env(config: EnvConfig, seed: Optional[int] = None) -> Any:
    """Factory function to create environments.

    Args:
        config: Environment configuration.
        seed: Random seed.

    Returns:
        SimulationEnvironment instance.
    """
    env_map = {
        "server_load": (ServerLoadEnv, ServerLoadConfig),
        "smart_grid": (SmartGridEnv, SmartGridConfig),
        "homeostasis": (HomeostasisEnv, HomeostasisConfig),
    }

    if config.name not in env_map:
        raise ValueError(f"Unknown environment: {config.name}. "
                        f"Available: {list(env_map.keys())}")

    EnvClass, ConfigClass = env_map[config.name]
    env_config = ConfigClass(**config.kwargs)
    return EnvClass(config=env_config, seed=seed)


def make_agent(config: AgentConfig, env: Any) -> BaseAgent:
    """Factory function to create agents.

    Args:
        config: Agent configuration.
        env: Environment instance (for inferring dimensions).

    Returns:
        BaseAgent instance.
    """
    # Infer dimensions from environment
    if hasattr(env.observation_space, 'shape'):
        state_dim = env.observation_space.shape[0]
    else:
        # Assuming flat observation
        state_dim = env.observation_space.n if hasattr(env.observation_space, 'n') else 10

    if hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n
        is_discrete = True
    else:
        action_dim = env.action_space.shape[0]
        is_discrete = False

    agent_map = {
        "linucb": lambda: LinUCBAgent(
            n_arms=config.kwargs.get("n_arms", action_dim),
            context_dim=state_dim,
            **{k: v for k, v in config.kwargs.items() if k != "n_arms"},
        ),
        "dqn": lambda: DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **config.kwargs,
        ),
        "mcts": lambda: MCTSAgent(
            n_actions=action_dim,
            **config.kwargs,
        ),
        "ppo": lambda: PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=float(env.action_space.low[0]) if not is_discrete else -1.0,
            action_high=float(env.action_space.high[0]) if not is_discrete else 1.0,
            **config.kwargs,
        ),
    }

    if config.name not in agent_map:
        raise ValueError(f"Unknown agent: {config.name}. "
                        f"Available: {list(agent_map.keys())}")

    return agent_map[config.name]()


def train(
    env_config: EnvConfig,
    agent_config: AgentConfig,
    training_config: TrainingConfig,
) -> Dict[str, Any]:
    """Run training loop.

    Args:
        env_config: Environment configuration.
        agent_config: Agent configuration.
        training_config: Training parameters.

    Returns:
        Training results dictionary.
    """
    # Set seeds
    set_global_seeds(training_config.seed)

    # Create environment and agent
    env = make_env(env_config, seed=training_config.seed)
    agent = make_agent(agent_config, env)

    # Special handling for MCTS
    if agent_config.name == "mcts":
        agent.set_environment(env)

    # Logger
    experiment_name = f"{env_config.name}_{agent_config.name}"
    logger = Logger(
        log_dir=training_config.log_dir,
        experiment_name=experiment_name,
    )

    # Training metrics
    episode_rewards = []
    episode_lengths = []

    print(f"Starting training: {experiment_name}")
    print(f"  Environment: {env_config.name}")
    print(f"  Agent: {agent_config.name}")
    print(f"  Max episodes: {training_config.max_episodes}")
    print()

    for episode in range(training_config.max_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store(state, action, reward, next_state, done)

            # Train if ready
            if agent.ready_to_train():
                metrics = agent.update()

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Check step limit
            if (training_config.max_steps_per_episode > 0 and
                episode_length >= training_config.max_steps_per_episode):
                break

        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Log
        if (episode + 1) % training_config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-training_config.log_interval:])
            avg_length = np.mean(episode_lengths[-training_config.log_interval:])
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Length: {avg_length:6.1f}")

            logger.log_episode(
                episode=episode + 1,
                reward=episode_reward,
                length=episode_length,
                extra={"avg_reward": avg_reward, "avg_length": avg_length},
            )

        # Save checkpoint
        if (episode + 1) % training_config.save_interval == 0:
            checkpoint_path = os.path.join(
                logger.log_dir, f"checkpoint_{episode + 1}.pt"
            )
            try:
                agent.save(checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
            except NotImplementedError:
                pass

    # Final summary
    print("\nTraining complete!")
    print(f"  Total episodes: {training_config.max_episodes}")
    print(f"  Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")

    logger.close()

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "final_avg_reward": float(np.mean(episode_rewards[-100:])),
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ab Initio RL Simulation System Training"
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="server_load",
        choices=["server_load", "smart_grid", "homeostasis"],
        help="Environment name",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="dqn",
        choices=["linucb", "dqn", "mcts", "ppo"],
        help="Agent name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Logging directory",
    )

    args = parser.parse_args()

    # Build configuration
    if args.preset:
        env_config, agent_config = PRESETS[args.preset]()
    else:
        env_config = EnvConfig(name=args.env)
        agent_config = AgentConfig(name=args.agent)

    training_config = TrainingConfig(
        seed=args.seed,
        max_episodes=args.episodes,
        log_dir=args.log_dir,
    )

    # Run training
    results = train(env_config, agent_config, training_config)

    return results


if __name__ == "__main__":
    main()
