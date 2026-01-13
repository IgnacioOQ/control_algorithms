"""
Server Load Balancing Environment - M/M/k Queueing with Discrete Event Simulation.

This environment simulates a multi-server queueing system where:
- Job arrivals follow a Poisson process
- Service times follow an exponential distribution
- The agent routes incoming jobs or controls server scaling

Based on Kendall's notation M/M/k queue theory.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ContinuousSpace, DiscreteSpace, SimulationEnvironment, State, StepResult


@dataclass
class ServerLoadConfig:
    """Configuration for the Server Load environment.

    Attributes:
        num_servers: Number of servers (k in M/M/k).
        arrival_rate: Mean arrival rate (λ) - jobs per second.
        service_rate: Mean service rate (μ) per server - jobs per second.
        max_queue_size: Maximum queue length per server before drops.
        step_duration: Duration of each agent step in seconds (Δt_step).
        alpha: Latency penalty weight.
        beta: Drop penalty weight.
        gamma: Energy/server cost weight.
        latency_window: Number of recent jobs for latency averaging.
    """

    num_servers: int = 4
    arrival_rate: float = 10.0  # λ: 10 jobs/second
    service_rate: float = 3.0  # μ: 3 jobs/second per server
    max_queue_size: int = 50
    step_duration: float = 1.0  # 1 second per agent step
    alpha: float = 1.0  # Latency penalty
    beta: float = 10.0  # Drop penalty
    gamma: float = 0.1  # Server cost
    latency_window: int = 100


@dataclass
class Job:
    """A job in the queueing system."""

    arrival_time: float
    service_time: float
    start_service_time: Optional[float] = None
    completion_time: Optional[float] = None


@dataclass
class Server:
    """A server in the queueing system."""

    id: int
    queue: List[Job] = field(default_factory=list)
    current_job: Optional[Job] = None
    busy: bool = False
    active: bool = True  # Can be deactivated for energy saving
    completion_time: float = float("inf")  # Time when current job completes

    def reset(self):
        self.queue = []
        self.current_job = None
        self.busy = False
        self.completion_time = float("inf")


class ServerLoadEnv(SimulationEnvironment):
    """M/M/k Queueing Environment with Discrete Event Simulation.

    The agent takes discrete routing decisions: which server to send incoming jobs to.

    State space (continuous, normalized):
        - Queue lengths for each server (k values)
        - Server busy status (k binary values)
        - Observed arrival rate (moving average)
        - Recent average latency

    Action space (discrete):
        - 0 to k-1: Route to specific server
        
    Reward:
        R = -(α·avg_latency + β·num_drops + γ·num_active_servers)
    """

    def __init__(self, config: Optional[ServerLoadConfig] = None, seed: Optional[int] = None):
        """Initialize the environment.

        Args:
            config: Environment configuration.
            seed: Random seed.
        """
        self.config = config or ServerLoadConfig()
        super().__init__(seed=seed)

    def _setup(self) -> None:
        """Set up observation and action spaces."""
        k = self.config.num_servers
        
        # State: [Q1...Qk, σ1...σk, λ_obs, L_recent] = 2k + 2 dimensions
        state_dim = 2 * k + 2
        self.observation_space = ContinuousSpace(
            low=np.zeros(state_dim, dtype=np.float32),
            high=np.array(
                [self.config.max_queue_size] * k  # Queue lengths
                + [1.0] * k  # Busy status
                + [self.config.arrival_rate * 2]  # Observed arrival rate (up to 2x)
                + [10.0],  # Recent latency (seconds, capped)
                dtype=np.float32,
            ),
        )
        
        # Action: route to server 0..k-1
        self.action_space = DiscreteSpace(n=k)
        
        # Initialize internal state
        self._init_internal_state()

    def _init_internal_state(self) -> None:
        """Initialize internal simulation state."""
        self.servers = [Server(id=i) for i in range(self.config.num_servers)]
        self.current_time = 0.0
        self.next_arrival_time = 0.0
        self._schedule_next_arrival()
        
        # Statistics tracking
        self.completed_jobs: List[Job] = []
        self.recent_latencies: List[float] = []
        self.arrivals_this_step = 0
        self.drops_this_step = 0
        self.total_drops = 0
        
        # Arrival rate estimation (moving average)
        self._arrival_times: List[float] = []
        self._arrival_window = 5.0  # 5 second window for rate estimation

    def reset(self, seed: Optional[int] = None) -> State:
        """Reset the environment."""
        if seed is not None:
            self.seed(seed)
        
        self._init_internal_state()
        return self._get_observation()

    def _schedule_next_arrival(self) -> None:
        """Schedule the next job arrival using exponential inter-arrival time."""
        # Inter-arrival time from exponential distribution: -ln(U)/λ
        u = self.rng.random()
        if u < 1e-10:
            u = 1e-10  # Numerical stability
        inter_arrival = -np.log(u) / self.config.arrival_rate
        self.next_arrival_time = self.current_time + inter_arrival

    def _generate_service_time(self) -> float:
        """Generate service time from exponential distribution."""
        u = self.rng.random()
        if u < 1e-10:
            u = 1e-10
        return -np.log(u) / self.config.service_rate

    def step(self, action: int) -> StepResult:
        """Execute one timestep with hybrid Step-Event architecture.

        The agent's action specifies where to route jobs during this step.
        Internally, we run a DES loop until step_duration is consumed.
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.config.num_servers - 1}]")
        
        target_server = action
        step_end_time = self.current_time + self.config.step_duration
        
        # Reset per-step statistics
        self.arrivals_this_step = 0
        self.drops_this_step = 0
        step_latencies: List[float] = []
        
        # Hybrid DES loop: process micro-events until step ends
        while self.current_time < step_end_time:
            # Find next event time
            next_event_time, event_type, event_data = self._get_next_event()
            
            if next_event_time >= step_end_time:
                # No more events before step ends
                self.current_time = step_end_time
                break
            
            # Advance time and process event
            self.current_time = next_event_time
            
            if event_type == "arrival":
                self._process_arrival(target_server)
            elif event_type == "departure":
                latency = self._process_departure(event_data)
                if latency is not None:
                    step_latencies.append(latency)
        
        # Calculate reward
        avg_latency = np.mean(step_latencies) if step_latencies else 0.0
        num_active = sum(1 for s in self.servers if s.active)
        
        reward = -(
            self.config.alpha * avg_latency
            + self.config.beta * self.drops_this_step
            + self.config.gamma * num_active
        )
        
        # Update recent latencies for observation
        self.recent_latencies.extend(step_latencies)
        self.recent_latencies = self.recent_latencies[-self.config.latency_window:]
        
        # Episode never ends (continuous operation), but could add max_steps
        done = False
        
        info = {
            "arrivals": self.arrivals_this_step,
            "drops": self.drops_this_step,
            "total_drops": self.total_drops,
            "avg_latency": avg_latency,
            "queue_lengths": [len(s.queue) + (1 if s.busy else 0) for s in self.servers],
        }
        
        return self._get_observation(), reward, done, info

    def _get_next_event(self) -> Tuple[float, str, Any]:
        """Determine the next event in the DES."""
        # Candidate 1: Next arrival
        next_arrival = self.next_arrival_time
        
        # Candidate 2: Next departure (earliest server completion)
        earliest_departure = float("inf")
        departing_server = None
        for server in self.servers:
            if server.busy and server.completion_time < earliest_departure:
                earliest_departure = server.completion_time
                departing_server = server
        
        if next_arrival <= earliest_departure:
            return next_arrival, "arrival", None
        else:
            return earliest_departure, "departure", departing_server

    def _process_arrival(self, target_server_id: int) -> None:
        """Process a job arrival event."""
        self.arrivals_this_step += 1
        self._arrival_times.append(self.current_time)
        
        # Create new job
        job = Job(
            arrival_time=self.current_time,
            service_time=self._generate_service_time(),
        )
        
        # Schedule next arrival
        self._schedule_next_arrival()
        
        # Route to target server
        server = self.servers[target_server_id]
        
        if len(server.queue) >= self.config.max_queue_size:
            # Queue full - job is dropped
            self.drops_this_step += 1
            self.total_drops += 1
            return
        
        # Add job to queue or start service immediately
        if not server.busy:
            self._start_service(server, job)
        else:
            server.queue.append(job)

    def _start_service(self, server: Server, job: Job) -> None:
        """Start servicing a job on a server."""
        server.current_job = job
        server.busy = True
        job.start_service_time = self.current_time
        job.completion_time = self.current_time + job.service_time
        server.completion_time = job.completion_time

    def _process_departure(self, server: Server) -> Optional[float]:
        """Process a job departure event. Returns job latency."""
        if not server.busy or server.current_job is None:
            return None
        
        job = server.current_job
        latency = job.completion_time - job.arrival_time
        self.completed_jobs.append(job)
        
        # Check if there's a queued job
        if server.queue:
            next_job = server.queue.pop(0)
            self._start_service(server, next_job)
        else:
            server.current_job = None
            server.busy = False
            server.completion_time = float("inf")
        
        return latency

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        k = self.config.num_servers
        
        # Queue lengths (normalized by max)
        queue_lengths = np.array(
            [len(s.queue) + (1 if s.busy else 0) for s in self.servers],
            dtype=np.float32,
        ) / self.config.max_queue_size
        
        # Server busy status
        busy_status = np.array([float(s.busy) for s in self.servers], dtype=np.float32)
        
        # Estimated arrival rate (from recent arrivals)
        self._arrival_times = [
            t for t in self._arrival_times if t > self.current_time - self._arrival_window
        ]
        if len(self._arrival_times) > 1:
            observed_rate = len(self._arrival_times) / self._arrival_window
        else:
            observed_rate = self.config.arrival_rate  # Default to configured rate
        observed_rate_norm = observed_rate / (self.config.arrival_rate * 2)
        
        # Recent average latency
        if self.recent_latencies:
            avg_latency = np.mean(self.recent_latencies)
        else:
            avg_latency = 0.0
        avg_latency_norm = min(avg_latency / 10.0, 1.0)  # Normalize, cap at 10s
        
        return np.concatenate([
            queue_lengths,
            busy_status,
            np.array([observed_rate_norm, avg_latency_norm], dtype=np.float32),
        ])

    def get_legal_actions(self) -> List[int]:
        """Get legal actions. All servers are valid routing targets."""
        return list(range(self.config.num_servers))

    def render(self) -> None:
        """Render current state."""
        print(f"Time: {self.current_time:.2f}s")
        for s in self.servers:
            queue_len = len(s.queue) + (1 if s.busy else 0)
            status = "BUSY" if s.busy else "IDLE"
            print(f"  Server {s.id}: [{status}] Queue: {queue_len}/{self.config.max_queue_size}")
        if self.recent_latencies:
            print(f"  Avg Latency: {np.mean(self.recent_latencies):.3f}s")
        print(f"  Total Drops: {self.total_drops}")

    def _get_state_repr(self) -> str:
        """Get string representation of state."""
        queues = [len(s.queue) + (1 if s.busy else 0) for s in self.servers]
        return f"Queues: {queues}, Drops: {self.total_drops}"

    def copy(self) -> "ServerLoadEnv":
        """Create a deep copy for MCTS planning."""
        new_env = ServerLoadEnv(config=self.config, seed=None)
        new_env.rng = np.random.RandomState()
        new_env.rng.set_state(self.rng.get_state())
        new_env.servers = copy.deepcopy(self.servers)
        new_env.current_time = self.current_time
        new_env.next_arrival_time = self.next_arrival_time
        new_env.completed_jobs = copy.deepcopy(self.completed_jobs)
        new_env.recent_latencies = self.recent_latencies.copy()
        new_env.arrivals_this_step = self.arrivals_this_step
        new_env.drops_this_step = self.drops_this_step
        new_env.total_drops = self.total_drops
        new_env._arrival_times = self._arrival_times.copy()
        return new_env
