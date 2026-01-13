# Comprehensive Architectural and Implementation Guidelines for Ab Initio Reinforcement Learning Simulation Systems

## 1\. Executive Summary and System Philosophy

The contemporary landscape of reinforcement learning (RL) research and development is dominated by high-level abstraction libraries such as OpenAI Gymnasium, Stable Baselines, and Ray RLLib. While these tools facilitate rapid prototyping, they often obscure the fundamental mathematical and mechanical operations governing agent-environment interactions. This report outlines a rigorous, implementation-ready guideline for coding assistants-specifically autonomous agents like Antigravity or Jules-tasked with generating ab initio reinforcement learning ecosystems. The mandate requires a complete departure from these high-level abstractions in favor of foundational, mathematically transparent implementations using Python, NumPy, and PyTorch/TensorFlow.

This approach serves two critical engineering purposes. First, it eliminates "black box" dependencies, ensuring that every stochastic process, gradient update, and state transition is explicitly defined and modifiable. Second, it allows for the seamless integration of heterogeneous environments-ranging from discrete event queueing systems to continuous biological feedback loops-under a unified control interface. By forcing the explicit definition of state-space orchestration, the resulting system exposes the raw mechanics of Markov Decision Processes (MDPs), providing a superior testbed for algorithmic research and educational transparency.

The report is structured to provide exhaustive detail on three distinct classes of simulation environments: Server Load Balancing (Stochastic Queueing), Electricity Management (Smart Grid/Demand Response), and Homeostasis (Biological Control). It then provides parallel depth on the implementation of four distinct agent architectures: Contextual Bandits (LinUCB), Deep Q-Networks (DQN), Monte Carlo Tree Search (MCTS), and Proximal Policy Optimization (PPO).

### 1.1 The Imperative of Explicit State-Space Orchestration

The unifying philosophy for the Coding Assistant must be **Explicit State-Space Orchestration**. Unlike library-based solutions which infer spaces, the generated code must explicitly define the mathematical bounds, transition dynamics, and reward manifolds of the system. This report serves as the "Ground Truth" specification for generating such code.

In standard libraries, a "step" is often treated as a unified concept. However, this abstraction fails when confronting the diversity of physical time scales present in our target domains.

- **Server Load Balancing** operates on microsecond-scale discrete events, requiring a bridge between continuous-time arrival processes and discrete-time agent decisions.<sup>1</sup>
- **Smart Grid Management** operates on hourly or 15-minute intervals, heavily dependent on stochastic forecasting and constraints.<sup>3</sup>
- **Biological Homeostasis** operates as a continuous dynamical system governed by differential equations, requiring numerical integration within the agent's decision loop.<sup>5</sup>

The Coding Assistant must effectively synthesize these disparate temporal dynamics into a coherent SimulationEnvironment protocol. The following sections detail the architectural blueprints, mathematical derivations, and implementation constraints necessary to realize this system.

## 2\. Foundational Architecture: The Agent-Environment Interface

Before addressing specific domains, the Coding Assistant must be instructed to build a shared protocol. Since we are rejecting gymnasium, we must define our own Environment and Space primitives. This ensures compatibility between the discrete nature of MCTS/DQN and the continuous nature of PPO/Homeostasis.

### 2.1 The Abstract Environment Protocol

The Assistant should generate a base class SimulationEnvironment that enforces a rigid contract. This contract must handle the dichotomy between **Discrete Time-Stepping** (DT) and **Discrete Event Simulation** (DES), a critical distinction for the Server Load Balancing use case. The interface must be typed strictly to prevent the runtime errors common in dynamically typed RL scripting.

The interface must expose the following core methods:

- reset() -> State: Re-initializes stochastic generators (Poisson arrival rates, battery SOC, glucose levels) and returns the initial observation vector \$s_0\$. This method must also accept a seed parameter to ensure deterministic reproducibility, a non-negotiable requirement for debugging RL algorithms.<sup>2</sup>
- step(action) -> (State, Reward, Done, Info): The core transition operator \$T(s, a) \\rightarrow s'\$. This function encapsulates the physics engine of the specific domain.
- get_legal_actions() -> List\[Action\]: Crucial for MCTS and Bandits to mask invalid branches (e.g., discharging an empty battery).
- render() -> void: A text-based or simple matplotlib visualization of the current state.

#### 2.1.1 State and Action Space Definitions

Instead of generic "Box" or "Discrete" classes, the Assistant should implement specific data structures for state metadata.

| **Component** | **Discrete Space Implementation** | **Continuous Space Implementation** |
| --- | --- | --- |
| **Data Type** | int or np.int64 | float or np.float32 |
| --- | --- | --- |
| **Bounds** | n_actions (Scalar) | low (Vector), high (Vector) |
| --- | --- | --- |
| **Sampling** | np.random.randint(0, n) | np.random.uniform(low, high) |
| --- | --- | --- |
| **Storage** | One-hot encoding or Integer ID | Normalized Vector \$\[-1, 1\]\$ |
| --- | --- | --- |

For continuous environments like Homeostasis, the Assistant must implement an **Observation Normalizer**. Neural networks struggle with unscaled inputs (e.g., Glucose at 100 mg/dL vs. Insulin at 10 \$\\mu U/mL\$). A running mean and variance filter must be applied to the raw state before it is passed to the agent.<sup>7</sup>

### 2.2 Data Structures for Experience Storage

For agents like DQN and PPO, the storage of experience is fundamental. The implementations must not rely on Python list appends (\`\`), which cause memory fragmentation and costly re-allocations as the buffer grows.

- **Circular Buffers (DQN)**: For DQN, the Assistant must implement a pre-allocated NumPy array buffer with pointers ptr and size. This buffer stores transitions \$(s, a, r, s', d)\$ and enables \$O(1)\$ insertion and sampling. The buffer should be initialized with np.zeros((capacity, state_dim)) to reserve contiguous memory blocks.
- **Trajectory Buffers (PPO)**: For PPO, the buffer need not be circular but must maintain the temporal order of transitions to calculate Generalized Advantage Estimation (GAE). The buffer should be cleared after every training epoch.

### 2.3 Random Number Generation (RNG) and Seeding

Reproducibility is the bedrock of scientific verification in RL. The Assistant must ensure that _every_ stochastic component-from the environment's arrival process to the agent's epsilon-greedy exploration-derives from a specific seed.

- **Global Seeding**: A utility function set_global_seeds(seed) should set torch.manual_seed, np.random.seed, and random.seed.
- **Local RNG**: Each environment instance should maintain its own np.random.RandomState object to prevent crosstalk between parallel environments during vectorized training.<sup>2</sup>

## 3\. Simulation Environment I: Server Load Balancing (Control Environment)

This domain requires the simulation of stochastic arrival processes and resource contention. The Coding Assistant must implement this not merely as a counter, but as a rigorous queueing theoretic model based on Kendall's Notation.

### 3.1 Mathematical Formulation: The M/M/k Model

The foundation of the simulation is the M/M/k queue, where arrivals are Markovian (Poisson process), service times are Markovian (Exponential distribution), and there are \$k\$ servers.<sup>8</sup>

#### 3.1.1 Arrival Process Dynamics

The simulator must generate job arrivals based on a Poisson process with rate \$\\lambda\$ (lambda). The inter-arrival time \$A_t\$ is distributed exponentially:

\$\$P(A_t \\le t) = 1 - e^{-\\lambda t}\$\$

In a naive implementation, one might check for an arrival at every micro-timestep. However, this is computationally inefficient. The Assistant must implement a "next-event" logic or a robust approximation for discrete time steps. For a time step \$\\Delta t\$, the number of arrivals \$N\$ follows a Poisson distribution:

\$\$P(N=k) = \\frac{(\\lambda \\Delta t)^k e^{-\\lambda \\Delta t}}{k!}\$\$

If \$\\Delta t\$ is sufficiently small such that \$\\lambda \\Delta t \\ll 1\$, this approximates a Bernoulli trial where \$P(\\text{arrival}) \\approx \\lambda \\Delta t\$. The environment code must verify this condition or adapt \$\\Delta t\$ dynamically.

#### 3.1.2 Service Process and Latency

Service times \$S\$ also follow an exponential distribution with mean \$1/\\mu\$. The utilization factor \$\\rho\$ is defined as \$\\rho = \\lambda / (k \\mu)\$. The fundamental trade-off in this environment is between **Latency** (\$L\$) and **Energy/Resources** (\$k\$).

- If \$\\rho \\to 1\$, the queue length (and thus latency) grows asymptotically to infinity.
- The RL agent's task is to dynamically adjust \$k\$ (scaling) or route jobs (load balancing) to maintain \$\\rho < 1\$ while minimizing active servers.

The state vector passed to the agent is:

\$\$s_t = \[Q_{1,t}, Q_{2,t}, \\dots, Q_{k,t}, \\sigma_{1,t}, \\dots, \\sigma_{k,t}, \\lambda_{obs}, \\bar{L}\_{recent}\]\$\$

Where:

- \$Q_{i,t}\$: Queue length of server \$i\$.
- \$\\sigma_{i,t}\$: Binary status of server \$i\$ (Busy/Idle).
- \$\\lambda_{obs}\$: Estimated arrival rate (moving average).
- \$\\bar{L}\_{recent}\$: Average latency of the last \$N\$ jobs.

### 3.2 Discrete Event Simulation (DES) vs. Time-Stepping Logic

While RL agents typically operate in fixed time steps (e.g., action every 1 second), queueing events occur in continuous time. The Assistant must implement a **Hybrid Step-Event Architecture**.<sup>9</sup>

- **The Step Call**: The agent calls env.step(action). This represents a fixed duration \$\\Delta T_{step}\$.
- **The Internal DES Loop**: Inside the step method, the environment simulates micro-events until \$\\Delta T_{step}\$ is consumed.
  - _Event 1_: Calculate time to next arrival \$\\tau_{arr} = - \\ln(U) / \\lambda\$.
  - _Event 2_: For each busy server, calculate time to completion \$\\tau_{dep, i}\$.
  - _Execution_: Identify the minimum time \$\\tau_{min} = \\min(\\tau_{arr}, \\tau_{dep})\$.
  - If \$\\tau_{accum} + \\tau_{min} < \\Delta T_{step}\$, advance system time, process the event (increment queue or free server), and repeat.
  - If the next event is beyond the time step, advance time to \$\\Delta T_{step}\$ and return control to the agent.

This hybrid approach ensures that the agent sees the _integrated_ effect of its decisions over the time step, capturing the stochastic burstiness of traffic.<sup>10</sup>

### 3.3 Reward Function Engineering

The reward function must be carefully shaped to avoid sparse feedback. It should consist of three weighted components:

\$\$R_t = - (\\alpha \\cdot \\bar{L}\_t + \\beta \\cdot N_{drops} + \\gamma \\cdot E_{cost})\$\$

- **Latency Penalty (\$\\alpha\$)**: Proportional to the average waiting time of jobs in the system during the step. This is the primary performance metric.
- **Drop Penalty (\$\\beta\$)**: A large scalar penalty applied whenever a job arrives at a full queue (buffer overflow). This acts as a soft constraint on reliability.
- **Energy Cost (\$\\gamma\$)**: A smaller penalty proportional to the number of active servers \$k\$. This encourages the agent to shut down servers during low traffic.

**Insight**: The weighting coefficients \$\\alpha, \\beta, \\gamma\$ define the system's operational profile. A high \$\\gamma\$ produces a "Green" power-saving policy that risks high latency; a high \$\\beta\$ produces a high-reliability policy. The Assistant should expose these as configuration parameters.

## 4\. Simulation Environment II: Electricity Management (Smart Grid)

The Smart Grid environment introduces constraints on storage (batteries), stochastic generation (renewables), and economic incentives (Demand Response). This is a constrained optimization problem wrapped as an MDP, suitable for both Contextual Bandits and RL agents.<sup>3</sup>

### 4.1 Battery Energy Storage System (BESS) Dynamics

The Coding Assistant must implement a BESS model that goes beyond simple integration. Real batteries have efficiency losses and degradation costs. The state equation for the State of Charge (\$SoC\$) is:

\$\$SoC_{t+1} = SoC_t + \\eta_{ch} P_{ch} \\Delta t - \\frac{1}{\\eta_{dis}} P_{dis} \\Delta t - \\delta_{leak}\$\$

- **Efficiency (\$\\eta\$)**: Charging efficiency \$\\eta_{ch} \\approx 0.95\$, discharging efficiency \$\\eta_{dis} \\approx 0.95\$. This asymmetry means energy is lost in every cycle, discouraging the agent from useless churning.<sup>12</sup>
- **Constraints**:
  - \$SoC_{min} \\le SoC_t \\le SoC_{max}\$.
  - \$0 \\le P_{ch} \\le P_{max}\$ (Charging power limit).
  - \$0 \\le P_{dis} \\le P_{max}\$ (Discharging power limit).

The implementation must use **clipping** logic: if the agent requests a discharge \$P_{req}\$ that exceeds the current stored energy, the environment must clip the action to the actual available energy and potentially return a "constraint violation" penalty.

### 4.2 Demand Response and Stochastic Processes

The agent acts as a controller responding to a time-varying price signal \$C_t\$. The goal is **Arbitrage**: buy low, sell (or consume) high.

- Price Signal Generation: To simulate realistic market fluctuations, the Assistant should implement an Ornstein-Uhlenbeck (OU) Process. Unlike white noise, an OU process is mean-reverting and auto-correlated, mimicking the temporal structure of real energy prices.  
    <br/>\$\$dX_t = \\theta (\\mu - X_t) dt + \\sigma dW_t\$\$
- **Load Profiles**: The base load (demand) should be modeled as a composition of a daily sinusoidal pattern (representing human activity cycles) and stochastic noise.

### 4.3 State Space and Observation

The state vector for the Smart Grid agent must provide sufficient context for planning:

\$\$s_t =\$\$

- \$SoC_t\$: Current battery level.
- \$L_t\$: Current building/grid load.
- \$P_{gen, t}\$: Renewable generation (Solar/Wind).
- \$C_t\$: Current electricity price.
- \$C_{t+1 \\dots t+H}\$: Rolling forecast of future prices. This "lookahead" is critical for agents to learn planning behaviors; without it, the problem is partially observable.<sup>3</sup>

### 4.4 Theoretical Optimization Bounds

To evaluate the RL agent, the Assistant should include a baseline solver using **Linear Programming (LP)**. Since the battery dynamics are linear (if we ignore complex degradation), an LP solver (e.g., scipy.optimize.linprog) having perfect foresight of the price trajectory can calculate the theoretical maximum profit.

- **Metric**: \$\\text{Efficiency} = \\frac{R_{agent}}{R_{optimal}}\$.
- This baseline allows the user to verify if the RL agent is learning effectively or stuck in a local optimum.<sup>4</sup>

## 5\. Simulation Environment III: Biological Homeostasis

This environment represents the most complex control challenge: non-linear, continuous dynamics with significant delays. We focus on the **Glucose-Insulin Feedback Loop**, specifically utilizing the **Bergman Minimal Model**. This system requires the agent to act as an "Artificial Pancreas".<sup>5</sup>

### 5.1 The Bergman Minimal Model Dynamics

The Assistant must implement the coupled differential equations governing the three compartments: Plasma Glucose (\$G\$), Remote Insulin (\$X\$), and Plasma Insulin (\$I\$).

- Glucose Dynamics:  
    <br/>\$\$\\frac{dG}{dt} = -(p_1 + X(t))G(t) + p_1 G_b + D(t)\$\$
  - \$p_1\$: Insulin-independent glucose uptake.
  - \$X(t)\$: The effect of active insulin in the remote compartment.
  - \$G_b\$: Basal glucose level (equilibrium).
  - \$D(t)\$: Exogenous glucose disturbance (e.g., a meal).
- Remote Insulin Dynamics:  
    <br/>\$\$\\frac{dX}{dt} = -p_2 X(t) + p_3 (I(t) - I_b)\$\$
  - \$p_2\$: Rate of insulin clearance.
  - \$p_3\$: Sensitivity of the remote compartment to plasma insulin.
- Plasma Insulin Dynamics:  
    <br/>\$\$\\frac{dI}{dt} = -n (I(t) - I_b) + \\gamma \[G(t) - h\]^+ + u(t)\$\$
  - \$n\$: Insulin decay rate.
  - \$\\gamma \[G(t) - h\]^+\$: Endogenous insulin secretion by the pancreas (often 0 in Type 1 Diabetes models).
  - \$u(t)\$: **The Agent's Action** (Insulin infusion rate).

### 5.2 Numerical Integration: RK4 vs. Euler

Since RL operates in discrete steps (e.g., 5 minutes), the environment must integrate these continuous equations over the interval \$\\Delta t\$.

- **Euler's Method**: \$y_{t+1} = y_t + f(y_t) \\Delta t\$. This is simple but numerically unstable for stiff biological equations where rates differ significantly (\$p_3\$ is often \$10^{-6}\$ while \$G\$ is \$10^2\$).
- **Runge-Kutta 4 (RK4)**: The Assistant must implement a custom RK4 solver to ensure stability.<sup>15</sup>  
    Python  
    def rk4_step(state, u, D, dt, derivatives_func):  
    k1 = derivatives_func(state, u, D)  
    k2 = derivatives_func(state + 0.5 \* dt \* k1, u, D)  
    k3 = derivatives_func(state + 0.5 \* dt \* k2, u, D)  
    k4 = derivatives_func(state + dt \* k3, u, D)  
    return state + (dt / 6.0) \* (k1 + 2\*k2 + 2\*k3 + k4)  
    <br/>This manual implementation avoids the need for scipy.integrate, adhering to the "no external libraries" constraint.

### 5.3 Safety-Critical Reward Shaping

Biological control is unique because the negative space is asymmetric. High glucose (Hyperglycemia, \$>180\$) is damaging over years; low glucose (Hypoglycemia, \$<70\$) can be fatal in minutes. The reward function must reflect this immediate risk.

\$\$R_t = -|G_t - G_{target}|^2 - \\lambda_{hypo} \\cdot \\mathbb{I}(G_t < 70)\$\$

- **Target**: Typically \$90-110\$ mg/dL.
- **Hypo Penalty (\$\\lambda_{hypo}\$)**: Must be magnitudes larger than the tracking error to force the agent to avoid low blood sugar at all costs.

### 5.4 Observation Normalization

The state variables in this environment vary wildly in magnitude (\$G \\approx 100\$, \$I \\approx 10\$, \$X \\approx 10^{-2}\$). Feeding these raw values into a neural network (PPO/DQN) will lead to poor convergence due to gradient imbalances. The Assistant must implement an Online Normalizer (Welford's algorithm) that tracks the running mean \$\\mu\$ and variance \$\\sigma^2\$ of the state, transforming inputs to:

\$\$\\hat{s} = \\frac{s - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}\$\$

## 6\. Agent Architecture I: Contextual Bandits (LinUCB)

Contextual bandits are ideal for the Smart Grid (Demand Response) or Server Routing scenarios where the decision depends on the immediate context, and the "state transition" is either non-existent or less relevant than the immediate reward.<sup>16</sup>

### 6.1 The Disjoint LinUCB Algorithm

The Assistant must implement the **Disjoint LinUCB** algorithm. This variant maintains a separate ridge regression model for each arm (action), which is computationally more expensive but often more accurate for arms with distinct behaviors (e.g., Charging vs. Discharging).

Model Assumption:

The expected reward for arm \$a\$ given context \$x_t\$ is linear:

\$\$\\mathbb{E}\[r_{t,a} | x_t\] = x_t^\\top \\theta_a\$\$

### 6.2 Matrix Maintenance and the Sherman-Morrison Formula

For each arm \$a \\in \\{1 \\dots K\\}\$, the agent stores:

- **Covariance Matrix Inverse**: \$A_a^{-1} \\in \\mathbb{R}^{d \\times d}\$ (Initialized as \$\\frac{1}{\\lambda} I_d\$, where \$\\lambda\$ is regularization).
- **Bias Vector**: \$b_a \\in \\mathbb{R}^{d}\$ (Initialized as zeros).

Selection Phase (Inference):

At step \$t\$, for each arm \$a\$, compute the Upper Confidence Bound (UCB):

\$\$\\hat{\\theta}\_a = A_a^{-1} b_a\$\$

\$\$p_{t,a} = \\hat{\\theta}\_a^\\top x_t + \\alpha \\sqrt{x_t^\\top A_a^{-1} x_t}\$\$

Select \$a_t = \\arg\\max_a p_{t,a}\$. Here, \$\\alpha\$ controls the exploration-exploitation trade-off.

Update Phase (Learning):

A naive implementation would invert matrix \$A_a\$ at every step, costing \$O(d^3)\$. The Assistant must implement the Sherman-Morrison update to maintain the inverse directly with \$O(d^2)\$ complexity.17

\$\$A_{new}^{-1} = A_{old}^{-1} - \\frac{A_{old}^{-1} x x^\\top A_{old}^{-1}}{1 + x^\\top A_{old}^{-1} x}\$\$

This optimization is critical for real-time simulation performance.

### 6.3 Implementation Guidelines

- **Context Normalization**: The input features \$x_t\$ (Price, Load) must be normalized. Unlike Deep RL where BatchNorm handles this, linear bandits are highly sensitive to feature scaling.
- **Action Discretization**: Since LinUCB requires discrete arms, the continuous Smart Grid action space (Charge/Discharge Power) must be bucketed (e.g., -100%, -50%, 0%, 50%, 100%).

## 7\. Agent Architecture II: Deep Q-Networks (DQN)

DQN is the primary agent for the Server Load Balancing environment, handling the discrete routing decisions based on complex state features.<sup>19</sup>

### 7.1 Neural Network Architecture

The Assistant should generate a QNetwork class using PyTorch.

- **Input Layer**: Dimension equal to state space size.
- **Hidden Layers**: Two or three layers of 64 to 256 neurons.
- **Activation**: ReLU is standard.
- **Output Layer**: Dimension equal to action space size (Linear activation).

### 7.2 The Replay Buffer

As noted in Section 2.2, a pre-allocated NumPy buffer is required. The implementation must support:

- add(s, a, r, s', d): Writes to the current ptr index and increments ptr % capacity.
- sample(batch_size): Returns a tuple of tensors. Crucially, the sampling indices should be generated using np.random.randint(0, size, batch_size).

### 7.3 Bellman Update and Loss Function

The core learning mechanism uses the Bellman Optimality Equation. The Assistant must explicitly code the target calculation:

\$\$Y_i = r_i + \\gamma \\cdot \\max_{a'} Q_{target}(s'\_i, a'; \\theta^-) \\cdot (1 - d_i)\$\$

- **Target Network**: A copy of the primary network, updated periodically.
- Loss: The Huber Loss (Smooth L1) should be used instead of MSE to prevent exploding gradients from outliers in the Q-values.  
    <br/>\$\$L(\\theta) = \\frac{1}{N} \\sum_i \\text{Huber}(Y_i - Q(s_i, a_i; \\theta))\$\$

### 7.4 Advanced Variants (Optional but Recommended)

To demonstrate "Expert" level capability, the Assistant should ideally implement Double DQN (DDQN) to reduce overestimation bias.

\$\$Y_i^{DDQN} = r_i + \\gamma Q_{target}(s'\_i, \\arg\\max_a Q_{local}(s'\_i, a); \\theta^-)\$\$

This requires only a minor modification to the target calculation logic but significantly improves stability in control tasks.

## 8\. Agent Architecture III: Monte Carlo Tree Search (MCTS)

MCTS serves as a planning agent. While typically associated with games, it is powerful for **Server Load Balancing** (route planning) or **Smart Grid** (discharge scheduling) when a model of the environment is available.<sup>21</sup>

### 8.1 The MCTS Node Structure

The MCTSNode class is the fundamental unit.

- **Attributes**:
  - N (Visit Count): Number of times this node has been traversed.
  - W (Total Value): Sum of rewards collected through this node.
  - Q (Mean Value): \$W/N\$.
  - P (Prior): Probability from a policy network (if using AlphaZero style) or uniform.
  - Children: Dictionary mapping Actions to Child Nodes.
  - State: A snapshot of the environment state at this node.

### 8.2 The Search Loop

The Assistant must implement the four standard phases:

- Selection: Traverse the tree from the root using the PUCT (Predictor + UCB) formula.  
    <br/>\$\$U(s, a) = Q(s, a) + c_{puct} P(s, a) \\frac{\\sqrt{\\sum_b N(s, b)}}{1 + N(s, a)}\$\$  
    <br/>This balances exploitation (\$Q\$) and exploration (driven by low \$N\$).
- **Expansion**: If the selected node is not terminal and has unvisited children, create a new child node.
- **Simulation (Rollout)**: From the new leaf, execute a "Rollout Policy" (usually random) until a terminal state or a depth limit is reached.
  - _Requirement_: The environment must support copy() or save/load to ensure these simulations do not affect the real agent.
- Backpropagation: Propagate the accumulated reward \$v\$ up the tree path.  
    <br/>\$\$N \\leftarrow N + 1\$\$  
    \$\$W \\leftarrow W + v\$\$  
    \$\$Q \\leftarrow W / N\$\$

### 8.3 Handling Continuous Domains

Standard MCTS requires discrete actions. For the Homeostasis environment, the Assistant must implement **Progressive Widening** or simply discretize the action space (e.g., Insulin = {0.0, 0.5, 1.0...}). The "From Scratch" nature suggests discretization is the safer, more robust implementation path for this project.

## 9\. Agent Architecture IV: Proximal Policy Optimization (PPO)

PPO is the designated agent for **Continuous Control** (Homeostasis/Bergman Model). It represents the state-of-the-art in policy gradient methods, balancing sample efficiency with stability.<sup>7</sup>

### 9.1 Actor-Critic Network

The neural network must have two heads (or two separate networks):

- **Actor (\$\\pi_\\theta\$)**: Outputs the parameters of a probability distribution.
  - For continuous actions: Mean \$\\mu\$ and Log-Std \$\\sigma\$. The action is sampled from \$\\mathcal{N}(\\mu, e^\\sigma)\$.
  - _Implementation Detail_: The log_std should be a learnable parameter, not a fixed value, allowing the agent to reduce exploration over time.
- **Critic (\$V_\\phi\$)**: Outputs a scalar value estimate of the state.

### 9.2 Generalized Advantage Estimation (GAE)

Implementing GAE is mandatory for high-performance PPO. It balances bias and variance in the advantage estimate.

The Assistant must implement the backward recursion:

\$\$\\delta_t = r_t + \\gamma V(s_{t+1}) \\cdot (1-d_t) - V(s_t)\$\$

\$\$A_t^{GAE} = \\delta_t + (\\gamma \\lambda) A_{t+1}^{GAE} \\cdot (1-d_t)\$\$

These advantages \$A_t\$ must be normalized (subtract mean, divide by std) before being used in the loss function to stabilize gradient descent.

### 9.3 The Clipped Surrogate Objective

The core PPO innovation is the clipped objective, which prevents the new policy from diverging too far from the old policy.

\$\$r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}\$\$

\$\$L^{CLIP} = \\mathbb{E} \[ \\min(r_t A_t, \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t) \]\$\$

The Assistant must generate code that:

- Collects a trajectory of \$T\$ steps.
- Computes old log-probabilities and advantages.
- Iterates through this batch for \$K\$ epochs.
- Computes the ratio \$r_t\$ and the clipped loss.
- Adds a Value Loss term \$(V(s) - V_{target})^2\$ and an Entropy Bonus \$S\[\\pi\]\$ to encourage exploration.

## 10\. Integration: The Training Loop and Orchestration

The final piece of the guideline is the orchestration of these components. The Assistant must structure the project to allow mixing and matching of agents and environments.

### 10.1 Modular Project Structure

The recommended file structure is:

project_root/

├── envs/

│ ├── base.py # SimulationEnvironment Interface

│ ├── server_load.py # M/M/k DES implementation

│ ├── smart_grid.py # Battery & Price dynamics

│ └── homeostasis.py # Bergman ODE solver

├── agents/

│ ├── base.py # BaseAgent Interface

│ ├── bandit.py # LinUCB

│ ├── dqn.py # Deep Q-Network & Replay Buffer

│ ├── mcts.py # MCTS Node & Search Loop

│ └── ppo.py # Actor-Critic & GAE

├── utils/

│ ├── math_ops.py # RK4, Sherman-Morrison, Normalization

│ └── logger.py # CSV/Tensorboard logging

└── main.py # Training Loop Orchestrator

### 10.2 The Standardized Training Loop

The main.py should implement a loop that works for all RL agents (Bandit, DQN, PPO):

Python

\# Conceptual Structure  
env = make_env(config)  
agent = make_agent(config)  
<br/>for episode in range(MAX_EPISODES):  
state = env.reset()  
done = False  
episode_reward = 0  
<br/>while not done:  
action = agent.select_action(state)  
next_state, reward, done, _= env.step(action)  
<br/>\# Store experience  
agent.store(state, action, reward, next_state, done)  
<br/>\# Train (Frequency depends on algo: per step for DQN, per epoch for PPO)  
if agent.ready_to_train():  
metrics = agent.update()  
<br/>state = next_state  
episode_reward += reward  
<br/>logger.log(episode, episode_reward)  

**Exception for MCTS**: Since MCTS is an online planner, it does not have a "training" phase in the same way. Its "select_action" method performs the search and update simultaneously. The guidelines must explicitly note this architectural difference.

### 10.3 Summary Comparisons

The following table summarizes the compatibility and complexity of the proposed components.

| **Environment** | **Dynamics Type** | **Recommended Agent** | **Key Implementation Challenge** |
| --- | --- | --- | --- |
| **Server Load** | Discrete Event (Stochastic) | DQN / MCTS | Hybrid Step-Event timing, Latency reward scaling |
| --- | --- | --- | --- |
| **Smart Grid** | Constrained Stochastic | LinUCB / DQN | Battery clipping logic, Price forecasting |
| --- | --- | --- | --- |
| **Homeostasis** | Continuous ODE (Stiff) | PPO | RK4 integration stability, Input normalization |
| --- | --- | --- | --- |

| **Agent** | **Action Space** | **Learning Paradigm** | **Key Math Component** |
| --- | --- | --- | --- |
| **LinUCB** | Discrete | Online (One-step) | Sherman-Morrison Matrix Inversion |
| --- | --- | --- | --- |
| **DQN** | Discrete | Off-policy (TD) | Replay Buffer, Bellman Target |
| --- | --- | --- | --- |
| **MCTS** | Discrete | Planning (Search) | UCT Formula, State Copying |
| --- | --- | --- | --- |
| **PPO** | Continuous | On-policy (Gradient) | GAE, Ratio Clipping |
| --- | --- | --- | --- |

## 11\. Conclusion

This report provides the Coding Assistant with a definitive blueprint for constructing a high-fidelity, "from scratch" RL simulation library. By strictly adhering to the mathematical models defined for the environments (Queueing, Battery Physics, Metabolic ODEs) and the architectural constraints for the agents (Explicit Memory, Matrix Operations, Tree Structures), the generated code will serve as a robust, transparent, and educational platform for advanced RL research. The removal of external library dependencies forces a deeper engagement with the underlying mechanics of decision-making under uncertainty, fulfilling the project's core pedagogical and engineering objectives.

#### Works cited

- Integrating Reinforcement Learning into M/M/1/K Retry Queueing Models for 6G Applications - MDPI, accessed January 13, 2026, <https://www.mdpi.com/1424-8220/25/12/3621>
- 4.1. Event-Driven Simulation of M/M/1 Queues - qmodels, accessed January 13, 2026, <https://qmodels.readthedocs.io/en/latest/mm1.html>
- Demand-Response Control in Smart Grids - MDPI, accessed January 13, 2026, <https://www.mdpi.com/2076-3417/13/4/2355>
- \[Basic\] Build Optimization Model to Schedule Battery's Operation in Power Grid Systems, accessed January 13, 2026, <https://medium.com/@yeap0022/basic-build-optimization-model-to-schedule-batterys-operation-in-power-grid-systems-51a8c04b3a0e>
- Homotopy perturbation approximate solutions for Bergman's minimal blood glucose-insulin model - OAText, accessed January 13, 2026, <https://www.oatext.com/homotopy-perturbation-approximate-solutions-for-bergmans-minimal-blood-glucose-insulin-model.php>
- Minimally-Invasive and Efficient Method to Accurately Fit the Bergman Minimal Model to Diabetes Type 2 - NIH, accessed January 13, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9124285/>
- Proximal Policy Optimization with PyTorch and Gymnasium - DataCamp, accessed January 13, 2026, <https://www.datacamp.com/tutorial/proximal-policy-optimization>
- Simulation-Driven Reinforcement Learning in Queuing Network Routing Optimization - arXiv, accessed January 13, 2026, <https://arxiv.org/html/2507.18795v1>
- Discrete Event Queuing Simulation - python - Stack Overflow, accessed January 13, 2026, <https://stackoverflow.com/questions/742776/discrete-event-queuing-simulation>
- Differentiable Discrete Event Simulation for Queuing Network Control - Columbia University, accessed January 13, 2026, <http://www.columbia.edu/~jd2736/publication/Differentiable_discrete_event.pdf>
- Simulation-based Strategies for Smart Demand Response - CORE, accessed January 13, 2026, <https://files.core.ac.uk/download/pdf/212459814.pdf>
- Lithium-ion Battery Mathematical Modelling Using Python and IoT - ResearchGate, accessed January 13, 2026, <https://www.researchgate.net/publication/371091509_Lithium-ion_Battery_Mathematical_Modelling_Using_Python_and_IoT>
- Energy Supply and Demand Optimisation: Mathematical Modelling Using Gurobi Python, accessed January 13, 2026, <https://towardsdatascience.com/energy-supply-and-demand-optimisation-mathematical-modelling-using-gurobi-python-8a8b1cb9559a/>
- An Updated Organ-Based Multi-Level Model for Glucose Homeostasis: Organ Distributions, Timing, and Impact of Blood Flow - Frontiers, accessed January 13, 2026, <https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.619254/full>
- Glucose and Insulin - Modeling and Simulation in Python, accessed January 13, 2026, <https://allendowney.github.io/ModSimPy/chap18.html>
- Linear Upper Confidence Bound Algorithm for Contextual Bandit Problem with Piled Rewards, accessed January 13, 2026, <https://www.csie.ntu.edu.tw/~htlin/paper/doc/pakdd16piled.pdf>
- \[2510.19349\] Scalable LinUCB: Low-Rank Design Matrix Updates for Recommenders with Large Action Spaces - arXiv, accessed January 13, 2026, <https://arxiv.org/abs/2510.19349>
- LinUCB (Linear Contextual UCB) - Paper | PDF | Applied Mathematics - Scribd, accessed January 13, 2026, <https://www.scribd.com/document/952220605/LinUCB-Linear-Contextual-UCB-Paper>
- Reinforcement Learning (DQN) Tutorial - PyTorch documentation, accessed January 13, 2026, <https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html>
- Deep Q-Network (DQN) Agent - MATLAB & Simulink - MathWorks, accessed January 13, 2026, <https://www.mathworks.com/help/reinforcement-learning/ug/dqn-agents.html>
- Tensor Implementation of Monte-Carlo Tree Search for Model-Based Reinforcement Learning - MDPI, accessed January 13, 2026, <https://www.mdpi.com/2076-3417/13/3/1406>
- Monte Carlo Tree Search (MCTS) in Machine Learning - GeeksforGeeks, accessed January 13, 2026, <https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/>
- The Proximal Policy Optimization (PPO) Algorithm, from Scratch - Kaggle, accessed January 13, 2026, <https://www.kaggle.com/code/auxeno/proximal-policy-optimization-from-scratch>
- Generalized Advantage Estimation (GAE) - labml.ai, accessed January 13, 2026, <https://nn.labml.ai/rl/ppo/gae.html>
