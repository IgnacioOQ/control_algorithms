# RL Simulations Reference
- status: active
- type: reference
- description: Pattern catalog for building simulation environments for optimization problems. Maps problem domains to state/action/dynamics/reward templates, with pointers to implementations in this repo.
- label: [source-material]
- injection: background
- volatility: stable
- last_checked: 2026-05-01
<!-- content -->

Use this document when you have a problem description or dataset and need to decide how to simulate it. Find the pattern that matches your domain, adapt the templates, then follow [RL_SIMULATIONS_SKILL.md](RL_SIMULATIONS_SKILL.md) to implement.

---

## Domain → Pattern Lookup

| Problem domain | Simulation pattern | Dynamics type | Repo example |
|----------------|--------------------|---------------|--------------|
| Server/network routing, call centers, warehouses | Queueing / routing | Discrete Event Simulation (DES) | `src/envs/server_load.py` |
| Inventory, supply chain, perishable goods | Inventory control | Discrete time-step + stochastic demand | `src/envs/stock_management.py` |
| Battery storage, HVAC, energy arbitrage | Storage / dispatch | Discrete time-step + OU process | `src/envs/smart_grid.py` |
| Drug infusion, chemical reactor, industrial process | Continuous process control | ODE system + RK4 | `src/envs/homeostasis.py` |
| Portfolio, pricing, demand forecasting | Financial / market | Discrete time-step + GBM/OU price | — |
| Fleet dispatch, ride-hailing, delivery routing | Spatial scheduling | Graph-based DES | — |

---

## Pattern 1: Queueing / Routing

**When to use:** Jobs, requests, or customers arrive stochastically and must be assigned to one of several servers, lanes, or nodes. The goal is to minimize latency, waiting time, or queue overflow.

**Canonical examples:** Load balancers, call center routing, warehouse pick stations, hospital triage.

**Repo implementation:** [src/envs/server_load.py](../src/envs/server_load.py)

### State template

```
s_t = [q_1, q_2, ..., q_k,          # queue length per server
       busy_1, ..., busy_k,          # binary: server occupied?
       lambda_obs,                   # observed arrival rate (moving avg)
       mean_latency_recent]          # average latency last N jobs
```

### Action space

Discrete: select server index `i ∈ {0, 1, ..., k-1}` to route the next arriving job.

### Dynamics type

**Discrete Event Simulation (DES)** inside each time step. Arrival times are exponential inter-arrivals (Poisson process); service times are exponential with rate μ. The agent's `step()` call spans a fixed duration Δt; internally the environment processes micro-events until Δt is consumed.

```
Arrivals:  inter-arrival time ~ Exp(λ)  →  P(N arrivals in Δt) = Poisson(λΔt)
Service:   service time ~ Exp(μ)
Utilization: ρ = λ / (k·μ)  — stable queue requires ρ < 1
```

### Reward design choices

```
R_t = -(α · mean_latency  +  β · n_dropped  +  γ · n_active_servers)
```

| Weight | Effect |
|--------|--------|
| High α | Prioritize low latency |
| High β | Prioritize reliability (no drops) |
| High γ | Prioritize energy / cost efficiency |

### Parameterization from data

| Data source | Parameter |
|-------------|-----------|
| Request logs (inter-arrival times) | λ (fit exponential) |
| Request logs (processing times) | μ (fit exponential or Erlang) |
| SLA / cost model | α, β, γ weights |
| Capacity plan | k (number of servers) |

---

## Pattern 2: Inventory Control

**When to use:** Items are held in stock, consumed by stochastic demand, and replenished by placing orders. Costs come from holding excess stock, spoilage, and stockouts.

**Canonical examples:** Retail inventory, pharmaceutical supply chains, food distribution, spare parts management.

**Repo implementation:** [src/envs/stock_management.py](../src/envs/stock_management.py)

### State template

```
s_t = [stock_1, ..., stock_n,        # current inventory per SKU
       age_profile_1, ...,           # FIFO age distribution (if perishable)
       demand_obs_1, ...,            # recent observed demand per SKU
       days_until_delivery]          # lead time remaining on open orders
```

### Action space

Continuous (or discretized): order quantity per SKU, `u_i ∈ [0, max_order_i]`.

### Dynamics

```
stock_{t+1,i} = stock_{t,i} + order_{t-L,i} - demand_{t,i} - spoilage_{t,i}
demand_{t,i}  ~ Poisson(λ_i)                   # or Negative Binomial for overdispersion
spoilage_{t,i} = FIFO units older than shelf_life_i
```

Lead time L means orders placed at t arrive at t+L. Inventory is clipped at [0, capacity_i].

### Reward design choices

```
R_t = revenue - purchase_cost - holding_cost - spoilage_cost - stockout_penalty
    = Σ_i [ p_i · min(stock_i, demand_i)
           - c_i · order_i
           - h_i · stock_i
           - s_i · spoilage_i
           - b_i · max(0, demand_i - stock_i) ]
```

### Parameterization from data

| Data source | Parameter |
|-------------|-----------|
| Sales history | λ_i (demand rate per SKU) |
| Expiry / waste records | shelf_life_i |
| Purchase orders | lead time L, supplier capacity |
| P&L / cost model | p_i (price), c_i (cost), h_i (holding), b_i (backlog) |

---

## Pattern 3: Storage / Dispatch (Energy)

**When to use:** A storage asset (battery, water tank, buffer) is charged and discharged over time in response to a price or demand signal. The goal is arbitrage (buy low / sell high) subject to capacity and rate constraints.

**Canonical examples:** Battery energy storage (BESS), HVAC thermal storage, pumped hydro, cold-chain buffer management.

**Repo implementation:** [src/envs/smart_grid.py](../src/envs/smart_grid.py)

### State template

```
s_t = [SoC_t,                        # state of charge ∈ [SoC_min, SoC_max]
       price_t,                      # current price / signal
       price_forecast_{t+1..t+H},   # rolling H-step price forecast
       load_t,                       # current demand
       generation_t]                 # renewable generation (if applicable)
```

### Action space

Continuous: net power `u ∈ [-P_max, P_max]` (positive = charge, negative = discharge).

### Dynamics

```
SoC_{t+1} = SoC_t + η_c · max(u,0) · Δt  -  max(-u,0) / η_d · Δt  -  δ_leak
price follows OU:  dP = θ(μ - P)dt + σ dW
```

Clip: SoC ∈ [SoC_min, SoC_max]; P ∈ [0, P_max_charge] for charge, [0, P_max_discharge] for discharge.

### Reward design choices

```
R_t = price_t · discharge_t - cost_t · charge_t - degradation_penalty · |u_t|
```

For demand response (minimize bill rather than maximize revenue):
```
R_t = -(grid_draw_t · price_t + peak_demand_charge · max_draw_this_month)
```

### Parameterization from data

| Data source | Parameter |
|-------------|-----------|
| Electricity price history | θ, μ, σ (OU fit via MLE) |
| Battery spec sheet | η_c, η_d, SoC_min/max, P_max, δ_leak |
| Load meter data | load profile mean + variance |
| Utility tariff | price structure for reward |

---

## Pattern 4: Continuous Process Control (ODE)

**When to use:** The system evolves as a continuous dynamical system governed by differential equations. The agent injects a control signal that appears as a forcing term in the ODEs. Key feature: the environment must numerically integrate ODEs inside each `step()` call.

**Canonical examples:** Drug infusion (insulin pump, anesthesia), chemical reactor temperature control, fermentation, water treatment pH.

**Repo implementation:** [src/envs/homeostasis.py](../src/envs/homeostasis.py) (Bergman minimal model)

### State template

```
s_t = [x_1(t), x_2(t), ..., x_n(t)]   # n compartment / state variables
```

The state variables are the outputs of the ODE system at discrete observation times.

### Action space

Continuous: infusion / injection rate `u(t) ∈ [0, u_max]`.

### Dynamics

General form (n coupled ODEs):
```
dx/dt = f(x, u, d, t)    # d = exogenous disturbance (e.g. meal, feed)
```

Integrate from t to t+Δt using RK4:
```
k1 = f(x_t,           u, d)
k2 = f(x_t + Δt/2·k1, u, d)
k3 = f(x_t + Δt/2·k2, u, d)
k4 = f(x_t + Δt·k3,   u, d)
x_{t+1} = x_t + (Δt/6)(k1 + 2k2 + 2k3 + k4)
```

Use RK4 (not Euler) for stiff biological/chemical ODEs where state magnitudes differ by orders of magnitude.

**Bergman minimal model (glucose-insulin, this repo):**
```
dG/dt = -p1(G - G_b) - X·G + D(t)
dX/dt = -p2·X + p3·(I - I_b)
dI/dt = -n·(I - I_b) + γ·[G - h]⁺ + u(t)
```

### Reward design choices

For setpoint tracking with asymmetric safety:
```
R_t = -|x_1 - x_target|²  -  λ_safety · I(x_1 < x_safe_min)
```

The safety penalty λ_safety must be much larger than the tracking term — the agent must learn that constraint violation is catastrophically worse than poor tracking.

### Parameterization from data

| Data source | Parameter |
|-------------|-----------|
| Clinical / lab measurements | ODE parameters (p1, p2, p3, n, γ — fit via least squares or MCMC) |
| Safety guidelines / spec limits | x_safe_min, x_safe_max |
| Historical disturbance records | D(t) distribution (meal timing, size) |
| Actuator spec | u_max (pump capacity) |

---

## Pattern 5: Financial / Market

**When to use:** An agent manages a position (portfolio, price, bid) in a market where prices evolve stochastically. Rewards are profit/loss or cost.

**Canonical examples:** Algorithmic trading, dynamic pricing, auction bidding, demand forecasting for pricing.

*No repo implementation — template only.*

### State template

```
s_t = [price_t,                      # current market price
       price_history_{t-L..t},       # recent price window
       position_t,                   # current holding
       cash_t,                       # available capital
       volatility_estimate_t]
```

### Action space

Continuous: position change `u ∈ [-max_trade, max_trade]`; or discrete: {buy, hold, sell}.

### Dynamics

```
price follows GBM:  dS = μS dt + σS dW   (log-normal returns)
  or mean-reverting OU:  dS = θ(μ - S)dt + σ dW   (commodities, spreads)
position_{t+1} = position_t + u_t
cash_{t+1} = cash_t - u_t · price_t - transaction_cost · |u_t|
```

### Reward design choices

```
R_t = (price_t - price_{t-1}) · position_t - transaction_cost · |u_t|
```

---

## Pattern 6: Spatial Scheduling (Fleet / Routing)

**When to use:** Agents or vehicles are positioned in a graph/map and must be dispatched to serve demand that arrives stochastically at nodes. Costs are travel time, idle time, and unmet demand.

**Canonical examples:** Ride-hailing dispatch, last-mile delivery, ambulance positioning, drone delivery.

*No repo implementation — template only.*

### State template

```
s_t = [vehicle_positions,            # location of each vehicle
       demand_queue_per_node,        # pending requests at each node
       travel_time_matrix,           # or distance matrix
       time_of_day]
```

### Action space

Discrete: assign vehicle `v` to node `n`; or continuous: target location for each vehicle.

### Dynamics

```
vehicle moves toward assigned node at speed v_max
demand arrives: requests_{t,n} ~ Poisson(λ_n(t))   # time-varying rate
service: vehicle at node clears one request per Δt
```

### Reward design choices

```
R_t = n_requests_served · value_per_request
     - Σ_v travel_cost · distance_traveled_v
     - penalty · n_requests_abandoned
```

---

## Choosing Between Patterns

If the problem has features from multiple patterns, combine them:

- **Inventory + routing:** multi-echelon supply chain (inventory at each node, routing between nodes)
- **Storage + process control:** HVAC with thermal mass (ODE for room temperature, storage for chiller)
- **Queueing + financial:** market maker (queue of orders, price as state)

When in doubt, start with the simplest pattern that captures the key cost driver, validate it against historical data, then add complexity.
