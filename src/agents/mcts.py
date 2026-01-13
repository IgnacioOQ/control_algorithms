"""
Monte Carlo Tree Search (MCTS) Agent.

Implements MCTS with:
- PUCT selection formula for exploration/exploitation balance
- Rollout policies for value estimation
- State copying for simulation
- Backpropagation of accumulated rewards

Suitable for:
- Server Load Balancing (route planning)
- Smart Grid (discharge scheduling with lookahead)
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .base import BaseAgent


@dataclass
class MCTSNode:
    """Node in the MCTS tree.

    Attributes:
        state: Environment state at this node.
        parent: Parent node (None for root).
        action: Action that led to this node from parent.
        children: Dictionary mapping actions to child nodes.
        N: Visit count.
        W: Total accumulated value.
        Q: Mean value (W/N).
        P: Prior probability (from policy network or uniform).
    """

    state: Any  # Environment state snapshot
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    N: int = 0  # Visit count
    W: float = 0.0  # Total value
    Q: float = 0.0  # Mean value = W/N
    P: float = 1.0  # Prior probability
    untried_actions: List[int] = field(default_factory=list)
    is_terminal: bool = False

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0


class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search Agent.

    Performs planning by building a search tree through simulation.
    Does not have a separate training phase - learning happens during search.

    Algorithm:
    1. Selection: Traverse tree using PUCT formula
    2. Expansion: Add new child for unvisited action
    3. Simulation: Rollout to estimate value
    4. Backpropagation: Update statistics up the tree

    Attributes:
        n_simulations: Number of MCTS simulations per action selection.
        c_puct: Exploration constant for PUCT formula.
        max_depth: Maximum rollout depth.
    """

    def __init__(
        self,
        n_actions: int,
        n_simulations: int = 100,
        c_puct: float = 1.414,
        gamma: float = 0.99,
        max_depth: int = 50,
        rollout_policy: Optional[Callable] = None,
        name: str = "MCTS",
    ):
        """Initialize the MCTS agent.

        Args:
            n_actions: Number of discrete actions.
            n_simulations: Simulations per select_action call.
            c_puct: PUCT exploration constant.
            gamma: Discount factor for rollout returns.
            max_depth: Maximum rollout depth.
            rollout_policy: Optional policy for rollouts. If None, uses random.
            name: Agent identifier.
        """
        super().__init__(name=name)

        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.max_depth = max_depth
        self.rollout_policy = rollout_policy

        # Current environment (set externally before select_action)
        self._env = None

        # Statistics
        self.total_simulations = 0

    def set_environment(self, env: Any) -> None:
        """Set the environment for planning.

        The environment must support copy() for simulation.

        Args:
            env: SimulationEnvironment instance.
        """
        self._env = env

    def select_action(
        self, state: np.ndarray, explore: bool = True
    ) -> int:
        """Select action by running MCTS simulations.

        Args:
            state: Current state (used for consistency, actual state from env).
            explore: If True, select proportionally to visit counts.
                    If False, select action with highest visit count.

        Returns:
            Selected action.
        """
        if self._env is None:
            raise RuntimeError("Environment not set. Call set_environment() first.")

        # Create root node
        root = MCTSNode(
            state=self._env.copy(),
            untried_actions=self._env.get_legal_actions(),
        )

        # Run simulations
        for _ in range(self.n_simulations):
            self._simulate(root)

        self.total_simulations += self.n_simulations

        # Select best action
        if explore and self.training:
            # Select proportionally to visit counts (for exploration)
            visits = np.array([
                root.children[a].N if a in root.children else 0
                for a in range(self.n_actions)
            ])
            if visits.sum() == 0:
                return np.random.randint(0, self.n_actions)
            probs = visits / visits.sum()
            return int(np.random.choice(self.n_actions, p=probs))
        else:
            # Select action with highest visit count (greedy)
            best_action = max(
                root.children.keys(),
                key=lambda a: root.children[a].N,
                default=0,
            )
            return best_action

    def _simulate(self, root: MCTSNode) -> float:
        """Run one MCTS simulation from root.

        Args:
            root: Root node of the search tree.

        Returns:
            Value estimate from this simulation.
        """
        node = root
        path = [node]

        # Selection: traverse to leaf
        while not node.is_leaf() and not node.is_terminal:
            node = self._select_child(node)
            path.append(node)

        # Expansion: add new child if not terminal
        if not node.is_terminal and not node.is_fully_expanded():
            node = self._expand(node)
            path.append(node)

        # Simulation: rollout to get value estimate
        value = self._rollout(node)

        # Backpropagation: update all nodes in path
        self._backpropagate(path, value)

        return value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using PUCT formula.

        PUCT: U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

        Args:
            node: Parent node.

        Returns:
            Selected child node.
        """
        total_visits = sum(child.N for child in node.children.values())
        sqrt_total = np.sqrt(total_visits) if total_visits > 0 else 1.0

        best_score = -float("inf")
        best_child = None

        for action, child in node.children.items():
            # PUCT formula
            exploitation = child.Q
            exploration = self.c_puct * child.P * sqrt_total / (1 + child.N)
            puct_score = exploitation + exploration

            if puct_score > best_score:
                best_score = puct_score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child.

        Args:
            node: Node to expand.

        Returns:
            Newly created child node.
        """
        if not node.untried_actions:
            return node

        # Select random untried action
        action = node.untried_actions.pop(
            np.random.randint(len(node.untried_actions))
        )

        # Simulate action in copied environment
        env_copy = node.state.copy()
        next_state, reward, done, _ = env_copy.step(action)

        # Create child node
        child = MCTSNode(
            state=env_copy,
            parent=node,
            action=action,
            untried_actions=[] if done else env_copy.get_legal_actions(),
            is_terminal=done,
            P=1.0 / self.n_actions,  # Uniform prior
        )

        node.children[action] = child
        return child

    def _rollout(self, node: MCTSNode) -> float:
        """Perform rollout from node to estimate value.

        Args:
            node: Leaf node to start rollout from.

        Returns:
            Discounted cumulative reward.
        """
        if node.is_terminal:
            return 0.0

        env = node.state.copy()
        total_reward = 0.0
        discount = 1.0

        for _ in range(self.max_depth):
            # Select action
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            if self.rollout_policy is not None:
                # Custom rollout policy
                action = self.rollout_policy(env)
            else:
                # Random rollout
                action = legal_actions[np.random.randint(len(legal_actions))]

            # Take step
            _, reward, done, _ = env.step(action)
            total_reward += discount * reward
            discount *= self.gamma

            if done:
                break

        return total_reward

    def _backpropagate(self, path: List[MCTSNode], value: float) -> None:
        """Backpropagate value up the tree.

        Args:
            path: List of nodes from root to leaf.
            value: Value estimate to propagate.
        """
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            # Discount value for parent
            value *= self.gamma

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """MCTS doesn't store transitions (planning-based)."""
        pass

    def update(self) -> Dict[str, float]:
        """MCTS doesn't have a separate training phase."""
        return {}

    def ready_to_train(self) -> bool:
        """MCTS is always 'ready' but never trains separately."""
        return False

    def get_action_values(self, root: MCTSNode) -> Dict[int, float]:
        """Get value estimates for each action from root.

        Args:
            root: Root node with children.

        Returns:
            Dictionary mapping actions to Q-values.
        """
        return {a: child.Q for a, child in root.children.items()}

    def get_action_visits(self, root: MCTSNode) -> Dict[int, int]:
        """Get visit counts for each action from root.

        Args:
            root: Root node with children.

        Returns:
            Dictionary mapping actions to visit counts.
        """
        return {a: child.N for a, child in root.children.items()}

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            **super().get_config(),
            "n_actions": self.n_actions,
            "n_simulations": self.n_simulations,
            "c_puct": self.c_puct,
            "gamma": self.gamma,
            "max_depth": self.max_depth,
        }
