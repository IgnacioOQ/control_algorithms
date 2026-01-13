"""
Configuration system for the Ab Initio RL Simulation System.

Provides dataclass-based configurations for environments and agents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainingConfig:
    """General training configuration.

    Attributes:
        seed: Random seed for reproducibility.
        max_episodes: Maximum number of training episodes.
        max_steps_per_episode: Maximum steps per episode (0 = no limit).
        log_interval: Episodes between logging.
        save_interval: Episodes between checkpoints.
        eval_episodes: Number of episodes for evaluation.
        log_dir: Directory for logs and checkpoints.
    """

    seed: int = 42
    max_episodes: int = 1000
    max_steps_per_episode: int = 0
    log_interval: int = 10
    save_interval: int = 100
    eval_episodes: int = 10
    log_dir: str = "logs"


@dataclass
class EnvConfig:
    """Environment configuration wrapper.

    Attributes:
        name: Environment name ('server_load', 'smart_grid', 'homeostasis').
        kwargs: Environment-specific configuration arguments.
    """

    name: str = "server_load"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent configuration wrapper.

    Attributes:
        name: Agent name ('linucb', 'dqn', 'mcts', 'ppo').
        kwargs: Agent-specific configuration arguments.
    """

    name: str = "dqn"
    kwargs: Dict[str, Any] = field(default_factory=dict)


# Preset configurations for common scenarios


def get_server_load_dqn_config() -> Tuple[EnvConfig, AgentConfig]:
    """Get preset config for Server Load with DQN."""
    env_config = EnvConfig(
        name="server_load",
        kwargs={
            "num_servers": 4,
            "arrival_rate": 10.0,
            "service_rate": 3.0,
        },
    )
    agent_config = AgentConfig(
        name="dqn",
        kwargs={
            "hidden_dims": (128, 128),
            "gamma": 0.99,
            "lr": 1e-3,
            "buffer_size": 50000,
            "batch_size": 64,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "double_dqn": True,
        },
    )
    return env_config, agent_config


def get_smart_grid_linucb_config() -> Tuple[EnvConfig, AgentConfig]:
    """Get preset config for Smart Grid with LinUCB."""
    env_config = EnvConfig(
        name="smart_grid",
        kwargs={
            "battery_capacity": 100.0,
            "max_power": 25.0,
            "episode_length": 96 * 7,  # 1 week
        },
    )
    # Discretize actions for LinUCB: 5 power levels
    agent_config = AgentConfig(
        name="linucb",
        kwargs={
            "n_arms": 5,  # -100%, -50%, 0%, +50%, +100%
            "alpha": 1.0,
            "regularization": 1.0,
        },
    )
    return env_config, agent_config


def get_homeostasis_ppo_config() -> Tuple[EnvConfig, AgentConfig]:
    """Get preset config for Homeostasis with PPO."""
    env_config = EnvConfig(
        name="homeostasis",
        kwargs={
            "dt_control": 5.0,  # 5-minute control interval
            "episode_length": 288,  # 24 hours
            "type1_diabetes": True,
        },
    )
    agent_config = AgentConfig(
        name="ppo",
        kwargs={
            "hidden_dims": (64, 64),
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "lr": 3e-4,
            "n_epochs": 10,
            "batch_size": 64,
            "buffer_size": 2048,
        },
    )
    return env_config, agent_config


def get_server_load_mcts_config() -> Tuple[EnvConfig, AgentConfig]:
    """Get preset config for Server Load with MCTS."""
    env_config = EnvConfig(
        name="server_load",
        kwargs={
            "num_servers": 4,
            "arrival_rate": 8.0,
            "service_rate": 3.0,
        },
    )
    agent_config = AgentConfig(
        name="mcts",
        kwargs={
            "n_simulations": 50,
            "c_puct": 1.414,
            "gamma": 0.99,
            "max_depth": 20,
        },
    )
    return env_config, agent_config


# Configuration presets mapping
PRESETS = {
    "server_load_dqn": get_server_load_dqn_config,
    "smart_grid_linucb": get_smart_grid_linucb_config,
    "homeostasis_ppo": get_homeostasis_ppo_config,
    "server_load_mcts": get_server_load_mcts_config,
}
