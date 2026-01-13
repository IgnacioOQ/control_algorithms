# **Comprehensive Architectural and Implementation Guidelines for Ab Initio Reinforcement Learning Simulation Systems**

## **1\. Executive Summary and System Philosophy**

The contemporary landscape of reinforcement learning (RL) research and development is dominated by high-level abstraction libraries such as OpenAI Gymnasium, Stable Baselines, and Ray RLLib. While these tools facilitate rapid prototyping, they often obscure the fundamental mathematical and mechanical operations governing agent-environment interactions. This report outlines a rigorous, implementation-ready guideline for coding assistants—specifically autonomous agents like Antigravity or Jules—tasked with generating ab initio reinforcement learning ecosystems. The mandate requires a complete departure from these high-level abstractions in favor of foundational, mathematically transparent implementations using Python, NumPy, and PyTorch/TensorFlow.

This approach serves two critical engineering purposes. First, it eliminates "black box" dependencies, ensuring that every stochastic process, gradient update, and state transition is explicitly defined and modifiable. Second, it allows for the seamless integration of heterogeneous environments—ranging from discrete event queueing systems to continuous biological feedback loops—under a unified control interface. By forcing the explicit definition of state-space orchestration, the resulting system exposes the raw mechanics of Markov Decision Processes (MDPs), providing a superior testbed for algorithmic research and educational transparency.

The report is structured to provide exhaustive detail on three distinct classes of simulation environments: Server Load Balancing (Stochastic Queueing), Electricity Management (Smart Grid/Demand Response), and Homeostasis (Biological Control). It then provides parallel depth on the implementation of four distinct agent architectures: Contextual Bandits (LinUCB), Deep Q-Networks (DQN), Monte Carlo Tree Search (MCTS), and Proximal Policy Optimization (PPO).

### **1.1 The Imperative of Explicit State-Space Orchestration**

The unifying philosophy for the Coding Assistant must be **Explicit State-Space Orchestration**. Unlike library-based solutions which infer spaces, the generated code must explicitly define the mathematical bounds, transition dynamics, and reward manifolds of the system. This report serves as the "Ground Truth" specification for generating such code.

In standard libraries, a "step" is often treated as a unified concept. However, this abstraction fails when confronting the diversity of physical time scales present in our target domains.

* **Server Load Balancing** operates on microsecond-scale discrete events, requiring a bridge between continuous-time arrival processes and discrete-time agent decisions.1  
* **Smart Grid Management** operates on hourly or 15-minute intervals, heavily dependent on stochastic forecasting and constraints.3  
* **Biological Homeostasis** operates as a continuous dynamical system governed by differential equations, requiring numerical integration within the agent's decision loop.5

The Coding Assistant must effectively synthesize these disparate temporal dynamics into a coherent SimulationEnvironment protocol. The following sections detail the architectural blueprints, mathematical derivations, and implementation constraints necessary to realize this system.

## ---

**2\. Foundational Architecture: The Agent-Environment Interface**

Before addressing specific domains, the Coding Assistant must be instructed to build a shared protocol. Since we are rejecting gymnasium, we must define our own Environment and Space primitives. This ensures compatibility between the discrete nature of MCTS/DQN and the continuous nature of PPO/Homeostasis.

### **2.1 The Abstract Environment Protocol**

The Assistant should generate a base class SimulationEnvironment that enforces a rigid contract. This contract must handle the dichotomy between **Discrete Time-Stepping** (DT) and **Discrete Event Simulation** (DES), a critical distinction for the Server Load Balancing use case. The interface must be typed strictly to prevent the runtime errors common in dynamically typed RL scripting.

The interface must expose the following core methods:

1. reset() \-\> State: Re-initializes stochastic generators (Poisson arrival rates, battery SOC, glucose levels) and returns the initial observation vector $s\_0$. This method must also accept a seed parameter to ensure deterministic reproducibility, a non-negotiable requirement for debugging RL algorithms.2  
2. step(action) \-\> (State, Reward, Done, Info): The core transition operator $T(s, a) \\rightarrow s'$. This function encapsulates the physics engine of the specific domain.  
3. get\_legal\_actions() \-\> List\[Action\]: Crucial for MCTS and Bandits to mask invalid branches (e.g., discharging an empty battery).  
4. render() \-\> void: A text-based or simple matplotlib visualization of the current state.

#### **2.1.1 State and Action Space Definitions**

Instead of generic "Box" or "Discrete" classes, the Assistant should implement specific data structures for state metadata.

| Component | Discrete Space Implementation | Continuous Space Implementation |
| :---- | :---- | :---- |
| **Data Type** | int or np.int64 | float or np.float32 |
| **Bounds** | n\_actions (Scalar) | low (Vector), high (Vector) |
| **Sampling** | np.random.randint(0, n) | np.random.uniform(low, high) |
| **Storage** | One-hot encoding or Integer ID | Normalized Vector $\[-1, 1\]$ |

For continuous environments like Homeostasis, the Assistant must implement an **Observation Normalizer**. Neural networks struggle with unscaled inputs (e.g., Glucose at 100 mg/dL vs. Insulin at 10 $\\mu U/mL$). A running mean and variance filter must be applied to the raw state before it is passed to the agent.7

### **2.2 Data Structures for Experience Storage**

For agents like DQN and PPO, the storage of experience is fundamental. The implementations must not rely on Python list appends (\`\`), which cause memory fragmentation and costly re-allocations as the buffer grows.

* **Circular Buffers (DQN)**: For DQN, the Assistant must implement a pre-allocated NumPy array buffer with pointers ptr and size. This buffer stores transitions $(s, a, r, s', d)$ and enables $O(1)$ insertion and sampling. The buffer should be initialized with np.zeros((capacity, state\_dim)) to reserve contiguous memory blocks.  
* **Trajectory Buffers (PPO)**: For PPO, the buffer need not be circular but must maintain the temporal order of transitions to calculate Generalized Advantage Estimation (GAE). The buffer should be cleared after every training epoch.

### **2.3 Random Number Generation (RNG) and Seeding**

Reproducibility is the bedrock of scientific verification in RL. The Assistant must ensure that *every* stochastic component—from the environment's arrival process to the agent's epsilon-greedy exploration—derives from a specific seed.

* **Global Seeding**: A utility function set\_global\_seeds(seed) should set torch.manual\_seed, np.random.seed, and random.seed.  
* **Local RNG**: Each environment instance should maintain its own np.random.RandomState object to prevent crosstalk between parallel environments during vectorized training.2

## ---

**3\. Simulation Environment I: Server Load Balancing (Control Environment)**

This domain requires the simulation of stochastic arrival processes and resource contention. The Coding Assistant must implement this not merely as a counter, but as a rigorous queueing theoretic model based on Kendall’s Notation.

### **3.1 Mathematical Formulation: The M/M/k Model**

The foundation of the simulation is the M/M/k queue, where arrivals are Markovian (Poisson process), service times are Markovian (Exponential distribution), and there are $k$ servers.8

#### **3.1.1 Arrival Process Dynamics**

The simulator must generate job arrivals based on a Poisson process with rate $\\lambda$ (lambda). The inter-arrival time $A\_t$ is distributed exponentially:

$$P(A\_t \\le t) \= 1 \- e^{-\\lambda t}$$  
In a naive implementation, one might check for an arrival at every micro-timestep. However, this is computationally inefficient. The Assistant must implement a "next-event" logic or a robust approximation for discrete time steps. For a time step $\\Delta t$, the number of arrivals $N$ follows a Poisson distribution:

$$P(N=k) \= \\frac{(\\lambda \\Delta t)^k e^{-\\lambda \\Delta t}}{k\!}$$  
If $\\Delta t$ is sufficiently small such that $\\lambda \\Delta t \\ll 1$, this approximates a Bernoulli trial where $P(\\text{arrival}) \\approx \\lambda \\Delta t$. The environment code must verify this condition or adapt $\\Delta t$ dynamically.

#### **3.1.2 Service Process and Latency**

Service times $S$ also follow an exponential distribution with mean $1/\\mu$. The utilization factor $\\rho$ is defined as $\\rho \= \\lambda / (k \\mu)$. The fundamental trade-off in this environment is between **Latency** ($L$) and **Energy/Resources** ($k$).

* If $\\rho \\to 1$, the queue length (and thus latency) grows asymptotically to infinity.  
* The RL agent's task is to dynamically adjust $k$ (scaling) or route jobs (load balancing) to maintain $\\rho \< 1$ while minimizing active servers.

The state vector passed to the agent is:

$$s\_t \= \[Q\_{1,t}, Q\_{2,t}, \\dots, Q\_{k,t}, \\sigma\_{1,t}, \\dots, \\sigma\_{k,t}, \\lambda\_{obs}, \\bar{L}\_{recent}\]$$

Where:

* $Q\_{i,t}$: Queue length of server $i$.  
* $\\sigma\_{i,t}$: Binary status of server $i$ (Busy/Idle).  
* $\\lambda\_{obs}$: Estimated arrival rate (moving average).  
* $\\bar{L}\_{recent}$: Average latency of the last $N$ jobs.

### **3.2 Discrete Event Simulation (DES) vs. Time-Stepping Logic**

While RL agents typically operate in fixed time steps (e.g., action every 1 second), queueing events occur in continuous time. The Assistant must implement a **Hybrid Step-Event Architecture**.9

1. **The Step Call**: The agent calls env.step(action). This represents a fixed duration $\\Delta T\_{step}$.  
2. **The Internal DES Loop**: Inside the step method, the environment simulates micro-events until $\\Delta T\_{step}$ is consumed.  
   * *Event 1*: Calculate time to next arrival $\\tau\_{arr} \= \- \\ln(U) / \\lambda$.  
   * *Event 2*: For each busy server, calculate time to completion $\\tau\_{dep, i}$.  
   * *Execution*: Identify the minimum time $\\tau\_{min} \= \\min(\\tau\_{arr}, \\tau\_{dep})$.  
   * If $\\tau\_{accum} \+ \\tau\_{min} \< \\Delta T\_{step}$, advance system time, process the event (increment queue or free server), and repeat.  
   * If the next event is beyond the time step, advance time to $\\Delta T\_{step}$ and return control to the agent.

This hybrid approach ensures that the agent sees the *integrated* effect of its decisions over the time step, capturing the stochastic burstiness of traffic.10

### **3.3 Reward Function Engineering**

The reward function must be carefully shaped to avoid sparse feedback. It should consist of three weighted components:

$$R\_t \= \- (\\alpha \\cdot \\bar{L}\_t \+ \\beta \\cdot N\_{drops} \+ \\gamma \\cdot E\_{cost})$$

* **Latency Penalty ($\\alpha$)**: Proportional to the average waiting time of jobs in the system during the step. This is the primary performance metric.  
* **Drop Penalty ($\\beta$)**: A large scalar penalty applied whenever a job arrives at a full queue (buffer overflow). This acts as a soft constraint on reliability.  
* **Energy Cost ($\\gamma$)**: A smaller penalty proportional to the number of active servers $k$. This encourages the agent to shut down servers during low traffic.

**Insight**: The weighting coefficients $\\alpha, \\beta, \\gamma$ define the system's operational profile. A high $\\gamma$ produces a "Green" power-saving policy that risks high latency; a high $\\beta$ produces a high-reliability policy. The Assistant should expose these as configuration parameters.

## ---

**4\. Simulation Environment II: Electricity Management (Smart Grid)**

The Smart Grid environment introduces constraints on storage (batteries), stochastic generation (renewables), and economic incentives (Demand Response). This is a constrained optimization problem wrapped as an MDP, suitable for both Contextual Bandits and RL agents.3

### **4.1 Battery Energy Storage System (BESS) Dynamics**

The Coding Assistant must implement a BESS model that goes beyond simple integration. Real batteries have efficiency losses and degradation costs. The state equation for the State of Charge ($SoC$) is:

$$SoC\_{t+1} \= SoC\_t \+ \\eta\_{ch} P\_{ch} \\Delta t \- \\frac{1}{\\eta\_{dis}} P\_{dis} \\Delta t \- \\delta\_{leak}$$

* **Efficiency ($\\eta$)**: Charging efficiency $\\eta\_{ch} \\approx 0.95$, discharging efficiency $\\eta\_{dis} \\approx 0.95$. This asymmetry means energy is lost in every cycle, discouraging the agent from useless churning.12  
* **Constraints**:  
  * $SoC\_{min} \\le SoC\_t \\le SoC\_{max}$.  
  * $0 \\le P\_{ch} \\le P\_{max}$ (Charging power limit).  
  * $0 \\le P\_{dis} \\le P\_{max}$ (Discharging power limit).

The implementation must use **clipping** logic: if the agent requests a discharge $P\_{req}$ that exceeds the current stored energy, the environment must clip the action to the actual available energy and potentially return a "constraint violation" penalty.

### **4.2 Demand Response and Stochastic Processes**

The agent acts as a controller responding to a time-varying price signal $C\_t$. The goal is **Arbitrage**: buy low, sell (or consume) high.

* Price Signal Generation: To simulate realistic market fluctuations, the Assistant should implement an Ornstein-Uhlenbeck (OU) Process. Unlike white noise, an OU process is mean-reverting and auto-correlated, mimicking the temporal structure of real energy prices.

  $$dX\_t \= \\theta (\\mu \- X\_t) dt \+ \\sigma dW\_t$$  
* **Load Profiles**: The base load (demand) should be modeled as a composition of a daily sinusoidal pattern (representing human activity cycles) and stochastic noise.

### **4.3 State Space and Observation**

The state vector for the Smart Grid agent must provide sufficient context for planning:

$$s\_t \=$$

* $SoC\_t$: Current battery level.  
* $L\_t$: Current building/grid load.  
* $P\_{gen, t}$: Renewable generation (Solar/Wind).  
* $C\_t$: Current electricity price.  
* $C\_{t+1 \\dots t+H}$: Rolling forecast of future prices. This "lookahead" is critical for agents to learn planning behaviors; without it, the problem is partially observable.3

### **4.4 Theoretical Optimization Bounds**

To evaluate the RL agent, the Assistant should include a baseline solver using **Linear Programming (LP)**. Since the battery dynamics are linear (if we ignore complex degradation), an LP solver (e.g., scipy.optimize.linprog) having perfect foresight of the price trajectory can calculate the theoretical maximum profit.

* **Metric**: $\\text{Efficiency} \= \\frac{R\_{agent}}{R\_{optimal}}$.  
* This baseline allows the user to verify if the RL agent is learning effectively or stuck in a local optimum.4

## ---

**5\. Simulation Environment III: Biological Homeostasis**

This environment represents the most complex control challenge: non-linear, continuous dynamics with significant delays. We focus on the **Glucose-Insulin Feedback Loop**, specifically utilizing the **Bergman Minimal Model**. This system requires the agent to act as an "Artificial Pancreas".5

### **5.1 The Bergman Minimal Model Dynamics**

The Assistant must implement the coupled differential equations governing the three compartments: Plasma Glucose ($G$), Remote Insulin ($X$), and Plasma Insulin ($I$).

1. Glucose Dynamics:

   $$\\frac{dG}{dt} \= \-(p\_1 \+ X(t))G(t) \+ p\_1 G\_b \+ D(t)$$  
   * $p\_1$: Insulin-independent glucose uptake.  
   * $X(t)$: The effect of active insulin in the remote compartment.  
   * $G\_b$: Basal glucose level (equilibrium).  
   * $D(t)$: Exogenous glucose disturbance (e.g., a meal).  
2. Remote Insulin Dynamics:

   $$\\frac{dX}{dt} \= \-p\_2 X(t) \+ p\_3 (I(t) \- I\_b)$$  
   * $p\_2$: Rate of insulin clearance.  
   * $p\_3$: Sensitivity of the remote compartment to plasma insulin.  
3. Plasma Insulin Dynamics:

   $$\\frac{dI}{dt} \= \-n (I(t) \- I\_b) \+ \\gamma \[G(t) \- h\]^+ \+ u(t)$$  
   * $n$: Insulin decay rate.  
   * $\\gamma \[G(t) \- h\]^+$: Endogenous insulin secretion by the pancreas (often 0 in Type 1 Diabetes models).  
   * $u(t)$: **The Agent's Action** (Insulin infusion rate).

### **5.2 Numerical Integration: RK4 vs. Euler**

Since RL operates in discrete steps (e.g., 5 minutes), the environment must integrate these continuous equations over the interval $\\Delta t$.

* **Euler's Method**: $y\_{t+1} \= y\_t \+ f(y\_t) \\Delta t$. This is simple but numerically unstable for stiff biological equations where rates differ significantly ($p\_3$ is often $10^{-6}$ while $G$ is $10^2$).  
* **Runge-Kutta 4 (RK4)**: The Assistant must implement a custom RK4 solver to ensure stability.15  
  Python  
  def rk4\_step(state, u, D, dt, derivatives\_func):  
      k1 \= derivatives\_func(state, u, D)  
      k2 \= derivatives\_func(state \+ 0.5 \* dt \* k1, u, D)  
      k3 \= derivatives\_func(state \+ 0.5 \* dt \* k2, u, D)  
      k4 \= derivatives\_func(state \+ dt \* k3, u, D)  
      return state \+ (dt / 6.0) \* (k1 \+ 2\*k2 \+ 2\*k3 \+ k4)

  This manual implementation avoids the need for scipy.integrate, adhering to the "no external libraries" constraint.

### **5.3 Safety-Critical Reward Shaping**

Biological control is unique because the negative space is asymmetric. High glucose (Hyperglycemia, $\>180$) is damaging over years; low glucose (Hypoglycemia, $\<70$) can be fatal in minutes. The reward function must reflect this immediate risk.

$$R\_t \= \-|G\_t \- G\_{target}|^2 \- \\lambda\_{hypo} \\cdot \\mathbb{I}(G\_t \< 70)$$

* **Target**: Typically $90-110$ mg/dL.  
* **Hypo Penalty ($\\lambda\_{hypo}$)**: Must be magnitudes larger than the tracking error to force the agent to avoid low blood sugar at all costs.

### **5.4 Observation Normalization**

The state variables in this environment vary wildly in magnitude ($G \\approx 100$, $I \\approx 10$, $X \\approx 10^{-2}$). Feeding these raw values into a neural network (PPO/DQN) will lead to poor convergence due to gradient imbalances. The Assistant must implement an Online Normalizer (Welford's algorithm) that tracks the running mean $\\mu$ and variance $\\sigma^2$ of the state, transforming inputs to:

$$\\hat{s} \= \\frac{s \- \\mu}{\\sqrt{\\sigma^2 \+ \\epsilon}}$$

## ---

**6\. Agent Architecture I: Contextual Bandits (LinUCB)**

Contextual bandits are ideal for the Smart Grid (Demand Response) or Server Routing scenarios where the decision depends on the immediate context, and the "state transition" is either non-existent or less relevant than the immediate reward.16

### **6.1 The Disjoint LinUCB Algorithm**

The Assistant must implement the **Disjoint LinUCB** algorithm. This variant maintains a separate ridge regression model for each arm (action), which is computationally more expensive but often more accurate for arms with distinct behaviors (e.g., Charging vs. Discharging).

Model Assumption:  
The expected reward for arm $a$ given context $x\_t$ is linear:

$$\\mathbb{E}\[r\_{t,a} | x\_t\] \= x\_t^\\top \\theta\_a$$

### **6.2 Matrix Maintenance and the Sherman-Morrison Formula**

For each arm $a \\in \\{1 \\dots K\\}$, the agent stores:

1. **Covariance Matrix Inverse**: $A\_a^{-1} \\in \\mathbb{R}^{d \\times d}$ (Initialized as $\\frac{1}{\\lambda} I\_d$, where $\\lambda$ is regularization).  
2. **Bias Vector**: $b\_a \\in \\mathbb{R}^{d}$ (Initialized as zeros).

Selection Phase (Inference):  
At step $t$, for each arm $a$, compute the Upper Confidence Bound (UCB):

$$\\hat{\\theta}\_a \= A\_a^{-1} b\_a$$

$$p\_{t,a} \= \\hat{\\theta}\_a^\\top x\_t \+ \\alpha \\sqrt{x\_t^\\top A\_a^{-1} x\_t}$$

Select $a\_t \= \\arg\\max\_a p\_{t,a}$. Here, $\\alpha$ controls the exploration-exploitation trade-off.  
Update Phase (Learning):  
A naive implementation would invert matrix $A\_a$ at every step, costing $O(d^3)$. The Assistant must implement the Sherman-Morrison update to maintain the inverse directly with $O(d^2)$ complexity.17

$$A\_{new}^{-1} \= A\_{old}^{-1} \- \\frac{A\_{old}^{-1} x x^\\top A\_{old}^{-1}}{1 \+ x^\\top A\_{old}^{-1} x}$$

This optimization is critical for real-time simulation performance.

### **6.3 Implementation Guidelines**

* **Context Normalization**: The input features $x\_t$ (Price, Load) must be normalized. Unlike Deep RL where BatchNorm handles this, linear bandits are highly sensitive to feature scaling.  
* **Action Discretization**: Since LinUCB requires discrete arms, the continuous Smart Grid action space (Charge/Discharge Power) must be bucketed (e.g., \-100%, \-50%, 0%, 50%, 100%).

## ---

**7\. Agent Architecture II: Deep Q-Networks (DQN)**

DQN is the primary agent for the Server Load Balancing environment, handling the discrete routing decisions based on complex state features.19

### **7.1 Neural Network Architecture**

The Assistant should generate a QNetwork class using PyTorch.

* **Input Layer**: Dimension equal to state space size.  
* **Hidden Layers**: Two or three layers of 64 to 256 neurons.  
* **Activation**: ReLU is standard.  
* **Output Layer**: Dimension equal to action space size (Linear activation).

### **7.2 The Replay Buffer**

As noted in Section 2.2, a pre-allocated NumPy buffer is required. The implementation must support:

* add(s, a, r, s', d): Writes to the current ptr index and increments ptr % capacity.  
* sample(batch\_size): Returns a tuple of tensors. Crucially, the sampling indices should be generated using np.random.randint(0, size, batch\_size).

### **7.3 Bellman Update and Loss Function**

The core learning mechanism uses the Bellman Optimality Equation. The Assistant must explicitly code the target calculation:

$$Y\_i \= r\_i \+ \\gamma \\cdot \\max\_{a'} Q\_{target}(s'\_i, a'; \\theta^-) \\cdot (1 \- d\_i)$$

* **Target Network**: A copy of the primary network, updated periodically.  
* Loss: The Huber Loss (Smooth L1) should be used instead of MSE to prevent exploding gradients from outliers in the Q-values.

  $$L(\\theta) \= \\frac{1}{N} \\sum\_i \\text{Huber}(Y\_i \- Q(s\_i, a\_i; \\theta))$$

### **7.4 Advanced Variants (Optional but Recommended)**

To demonstrate "Expert" level capability, the Assistant should ideally implement Double DQN (DDQN) to reduce overestimation bias.

$$Y\_i^{DDQN} \= r\_i \+ \\gamma Q\_{target}(s'\_i, \\arg\\max\_a Q\_{local}(s'\_i, a); \\theta^-)$$

This requires only a minor modification to the target calculation logic but significantly improves stability in control tasks.

## ---

**8\. Agent Architecture III: Monte Carlo Tree Search (MCTS)**

MCTS serves as a planning agent. While typically associated with games, it is powerful for **Server Load Balancing** (route planning) or **Smart Grid** (discharge scheduling) when a model of the environment is available.21

### **8.1 The MCTS Node Structure**

The MCTSNode class is the fundamental unit.

* **Attributes**:  
  * N (Visit Count): Number of times this node has been traversed.  
  * W (Total Value): Sum of rewards collected through this node.  
  * Q (Mean Value): $W/N$.  
  * P (Prior): Probability from a policy network (if using AlphaZero style) or uniform.  
  * Children: Dictionary mapping Actions to Child Nodes.  
  * State: A snapshot of the environment state at this node.

### **8.2 The Search Loop**

The Assistant must implement the four standard phases:

1. Selection: Traverse the tree from the root using the PUCT (Predictor \+ UCB) formula.

   $$U(s, a) \= Q(s, a) \+ c\_{puct} P(s, a) \\frac{\\sqrt{\\sum\_b N(s, b)}}{1 \+ N(s, a)}$$

   This balances exploitation ($Q$) and exploration (driven by low $N$).  
2. **Expansion**: If the selected node is not terminal and has unvisited children, create a new child node.  
3. **Simulation (Rollout)**: From the new leaf, execute a "Rollout Policy" (usually random) until a terminal state or a depth limit is reached.  
   * *Requirement*: The environment must support copy() or save/load to ensure these simulations do not affect the real agent.  
4. Backpropagation: Propagate the accumulated reward $v$ up the tree path.

   $$N \\leftarrow N \+ 1$$  
   $$W \\leftarrow W \+ v$$  
   $$Q \\leftarrow W / N$$

### **8.3 Handling Continuous Domains**

Standard MCTS requires discrete actions. For the Homeostasis environment, the Assistant must implement **Progressive Widening** or simply discretize the action space (e.g., Insulin \= {0.0, 0.5, 1.0...}). The "From Scratch" nature suggests discretization is the safer, more robust implementation path for this project.

## ---

**9\. Agent Architecture IV: Proximal Policy Optimization (PPO)**

PPO is the designated agent for **Continuous Control** (Homeostasis/Bergman Model). It represents the state-of-the-art in policy gradient methods, balancing sample efficiency with stability.7

### **9.1 Actor-Critic Network**

The neural network must have two heads (or two separate networks):

1. **Actor ($\\pi\_\\theta$)**: Outputs the parameters of a probability distribution.  
   * For continuous actions: Mean $\\mu$ and Log-Std $\\sigma$. The action is sampled from $\\mathcal{N}(\\mu, e^\\sigma)$.  
   * *Implementation Detail*: The log\_std should be a learnable parameter, not a fixed value, allowing the agent to reduce exploration over time.  
2. **Critic ($V\_\\phi$)**: Outputs a scalar value estimate of the state.

### **9.2 Generalized Advantage Estimation (GAE)**

Implementing GAE is mandatory for high-performance PPO. It balances bias and variance in the advantage estimate.  
The Assistant must implement the backward recursion:

$$\\delta\_t \= r\_t \+ \\gamma V(s\_{t+1}) \\cdot (1-d\_t) \- V(s\_t)$$

$$A\_t^{GAE} \= \\delta\_t \+ (\\gamma \\lambda) A\_{t+1}^{GAE} \\cdot (1-d\_t)$$

These advantages $A\_t$ must be normalized (subtract mean, divide by std) before being used in the loss function to stabilize gradient descent.

### **9.3 The Clipped Surrogate Objective**

The core PPO innovation is the clipped objective, which prevents the new policy from diverging too far from the old policy.

$$r\_t(\\theta) \= \\frac{\\pi\_\\theta(a\_t|s\_t)}{\\pi\_{\\theta\_{old}}(a\_t|s\_t)}$$

$$L^{CLIP} \= \\mathbb{E} \[ \\min(r\_t A\_t, \\text{clip}(r\_t, 1-\\epsilon, 1+\\epsilon) A\_t) \]$$  
The Assistant must generate code that:

1. Collects a trajectory of $T$ steps.  
2. Computes old log-probabilities and advantages.  
3. Iterates through this batch for $K$ epochs.  
4. Computes the ratio $r\_t$ and the clipped loss.  
5. Adds a Value Loss term $(V(s) \- V\_{target})^2$ and an Entropy Bonus $S\[\\pi\]$ to encourage exploration.

## ---

**10\. Integration: The Training Loop and Orchestration**

The final piece of the guideline is the orchestration of these components. The Assistant must structure the project to allow mixing and matching of agents and environments.

### **10.1 Modular Project Structure**

The recommended file structure is:  
project\_root/  
├── envs/  
│ ├── base.py \# SimulationEnvironment Interface  
│ ├── server\_load.py \# M/M/k DES implementation  
│ ├── smart\_grid.py \# Battery & Price dynamics  
│ └── homeostasis.py \# Bergman ODE solver  
├── agents/  
│ ├── base.py \# BaseAgent Interface  
│ ├── bandit.py \# LinUCB  
│ ├── dqn.py \# Deep Q-Network & Replay Buffer  
│ ├── mcts.py \# MCTS Node & Search Loop  
│ └── ppo.py \# Actor-Critic & GAE  
├── utils/  
│ ├── math\_ops.py \# RK4, Sherman-Morrison, Normalization  
│ └── logger.py \# CSV/Tensorboard logging  
└── main.py \# Training Loop Orchestrator

### **10.2 The Standardized Training Loop**

The main.py should implement a loop that works for all RL agents (Bandit, DQN, PPO):

Python

\# Conceptual Structure  
env \= make\_env(config)  
agent \= make\_agent(config)

for episode in range(MAX\_EPISODES):  
    state \= env.reset()  
    done \= False  
    episode\_reward \= 0  
      
    while not done:  
        action \= agent.select\_action(state)  
        next\_state, reward, done, \_ \= env.step(action)  
          
        \# Store experience  
        agent.store(state, action, reward, next\_state, done)  
          
        \# Train (Frequency depends on algo: per step for DQN, per epoch for PPO)  
        if agent.ready\_to\_train():  
            metrics \= agent.update()  
              
        state \= next\_state  
        episode\_reward \+= reward  
          
    logger.log(episode, episode\_reward)

**Exception for MCTS**: Since MCTS is an online planner, it does not have a "training" phase in the same way. Its "select\_action" method performs the search and update simultaneously. The guidelines must explicitly note this architectural difference.

### **10.3 Summary Comparisons**

The following table summarizes the compatibility and complexity of the proposed components.

| Environment | Dynamics Type | Recommended Agent | Key Implementation Challenge |
| :---- | :---- | :---- | :---- |
| **Server Load** | Discrete Event (Stochastic) | DQN / MCTS | Hybrid Step-Event timing, Latency reward scaling |
| **Smart Grid** | Constrained Stochastic | LinUCB / DQN | Battery clipping logic, Price forecasting |
| **Homeostasis** | Continuous ODE (Stiff) | PPO | RK4 integration stability, Input normalization |

| Agent | Action Space | Learning Paradigm | Key Math Component |
| :---- | :---- | :---- | :---- |
| **LinUCB** | Discrete | Online (One-step) | Sherman-Morrison Matrix Inversion |
| **DQN** | Discrete | Off-policy (TD) | Replay Buffer, Bellman Target |
| **MCTS** | Discrete | Planning (Search) | UCT Formula, State Copying |
| **PPO** | Continuous | On-policy (Gradient) | GAE, Ratio Clipping |

## **11\. Conclusion**

This report provides the Coding Assistant with a definitive blueprint for constructing a high-fidelity, "from scratch" RL simulation library. By strictly adhering to the mathematical models defined for the environments (Queueing, Battery Physics, Metabolic ODEs) and the architectural constraints for the agents (Explicit Memory, Matrix Operations, Tree Structures), the generated code will serve as a robust, transparent, and educational platform for advanced RL research. The removal of external library dependencies forces a deeper engagement with the underlying mechanics of decision-making under uncertainty, fulfilling the project's core pedagogical and engineering objectives.

#### **Works cited**

1. Integrating Reinforcement Learning into M/M/1/K Retry Queueing Models for 6G Applications \- MDPI, accessed January 13, 2026, [https://www.mdpi.com/1424-8220/25/12/3621](https://www.mdpi.com/1424-8220/25/12/3621)  
2. 4.1. Event-Driven Simulation of M/M/1 Queues \- qmodels, accessed January 13, 2026, [https://qmodels.readthedocs.io/en/latest/mm1.html](https://qmodels.readthedocs.io/en/latest/mm1.html)  
3. Demand-Response Control in Smart Grids \- MDPI, accessed January 13, 2026, [https://www.mdpi.com/2076-3417/13/4/2355](https://www.mdpi.com/2076-3417/13/4/2355)  
4. \[Basic\] Build Optimization Model to Schedule Battery's Operation in Power Grid Systems, accessed January 13, 2026, [https://medium.com/@yeap0022/basic-build-optimization-model-to-schedule-batterys-operation-in-power-grid-systems-51a8c04b3a0e](https://medium.com/@yeap0022/basic-build-optimization-model-to-schedule-batterys-operation-in-power-grid-systems-51a8c04b3a0e)  
5. Homotopy perturbation approximate solutions for Bergman's minimal blood glucose-insulin model \- OAText, accessed January 13, 2026, [https://www.oatext.com/homotopy-perturbation-approximate-solutions-for-bergmans-minimal-blood-glucose-insulin-model.php](https://www.oatext.com/homotopy-perturbation-approximate-solutions-for-bergmans-minimal-blood-glucose-insulin-model.php)  
6. Minimally-Invasive and Efficient Method to Accurately Fit the Bergman Minimal Model to Diabetes Type 2 \- NIH, accessed January 13, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9124285/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9124285/)  
7. Proximal Policy Optimization with PyTorch and Gymnasium \- DataCamp, accessed January 13, 2026, [https://www.datacamp.com/tutorial/proximal-policy-optimization](https://www.datacamp.com/tutorial/proximal-policy-optimization)  
8. Simulation-Driven Reinforcement Learning in Queuing Network Routing Optimization \- arXiv, accessed January 13, 2026, [https://arxiv.org/html/2507.18795v1](https://arxiv.org/html/2507.18795v1)  
9. Discrete Event Queuing Simulation \- python \- Stack Overflow, accessed January 13, 2026, [https://stackoverflow.com/questions/742776/discrete-event-queuing-simulation](https://stackoverflow.com/questions/742776/discrete-event-queuing-simulation)  
10. Differentiable Discrete Event Simulation for Queuing Network Control \- Columbia University, accessed January 13, 2026, [http://www.columbia.edu/\~jd2736/publication/Differentiable\_discrete\_event.pdf](http://www.columbia.edu/~jd2736/publication/Differentiable_discrete_event.pdf)  
11. Simulation-based Strategies for Smart Demand Response \- CORE, accessed January 13, 2026, [https://files.core.ac.uk/download/pdf/212459814.pdf](https://files.core.ac.uk/download/pdf/212459814.pdf)  
12. Lithium-ion Battery Mathematical Modelling Using Python and IoT \- ResearchGate, accessed January 13, 2026, [https://www.researchgate.net/publication/371091509\_Lithium-ion\_Battery\_Mathematical\_Modelling\_Using\_Python\_and\_IoT](https://www.researchgate.net/publication/371091509_Lithium-ion_Battery_Mathematical_Modelling_Using_Python_and_IoT)  
13. Energy Supply and Demand Optimisation: Mathematical Modelling Using Gurobi Python, accessed January 13, 2026, [https://towardsdatascience.com/energy-supply-and-demand-optimisation-mathematical-modelling-using-gurobi-python-8a8b1cb9559a/](https://towardsdatascience.com/energy-supply-and-demand-optimisation-mathematical-modelling-using-gurobi-python-8a8b1cb9559a/)  
14. An Updated Organ-Based Multi-Level Model for Glucose Homeostasis: Organ Distributions, Timing, and Impact of Blood Flow \- Frontiers, accessed January 13, 2026, [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.619254/full](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.619254/full)  
15. Glucose and Insulin — Modeling and Simulation in Python, accessed January 13, 2026, [https://allendowney.github.io/ModSimPy/chap18.html](https://allendowney.github.io/ModSimPy/chap18.html)  
16. Linear Upper Confidence Bound Algorithm for Contextual Bandit Problem with Piled Rewards, accessed January 13, 2026, [https://www.csie.ntu.edu.tw/\~htlin/paper/doc/pakdd16piled.pdf](https://www.csie.ntu.edu.tw/~htlin/paper/doc/pakdd16piled.pdf)  
17. \[2510.19349\] Scalable LinUCB: Low-Rank Design Matrix Updates for Recommenders with Large Action Spaces \- arXiv, accessed January 13, 2026, [https://arxiv.org/abs/2510.19349](https://arxiv.org/abs/2510.19349)  
18. LinUCB (Linear Contextual UCB) \- Paper | PDF | Applied Mathematics \- Scribd, accessed January 13, 2026, [https://www.scribd.com/document/952220605/LinUCB-Linear-Contextual-UCB-Paper](https://www.scribd.com/document/952220605/LinUCB-Linear-Contextual-UCB-Paper)  
19. Reinforcement Learning (DQN) Tutorial \- PyTorch documentation, accessed January 13, 2026, [https://docs.pytorch.org/tutorials/intermediate/reinforcement\_q\_learning.html](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)  
20. Deep Q-Network (DQN) Agent \- MATLAB & Simulink \- MathWorks, accessed January 13, 2026, [https://www.mathworks.com/help/reinforcement-learning/ug/dqn-agents.html](https://www.mathworks.com/help/reinforcement-learning/ug/dqn-agents.html)  
21. Tensor Implementation of Monte-Carlo Tree Search for Model-Based Reinforcement Learning \- MDPI, accessed January 13, 2026, [https://www.mdpi.com/2076-3417/13/3/1406](https://www.mdpi.com/2076-3417/13/3/1406)  
22. Monte Carlo Tree Search (MCTS) in Machine Learning \- GeeksforGeeks, accessed January 13, 2026, [https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/](https://www.geeksforgeeks.org/machine-learning/monte-carlo-tree-search-mcts-in-machine-learning/)  
23. The Proximal Policy Optimization (PPO) Algorithm, from Scratch \- Kaggle, accessed January 13, 2026, [https://www.kaggle.com/code/auxeno/proximal-policy-optimization-from-scratch](https://www.kaggle.com/code/auxeno/proximal-policy-optimization-from-scratch)  
24. Generalized Advantage Estimation (GAE) \- labml.ai, accessed January 13, 2026, [https://nn.labml.ai/rl/ppo/gae.html](https://nn.labml.ai/rl/ppo/gae.html)