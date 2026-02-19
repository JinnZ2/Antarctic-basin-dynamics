import numpy as np
import matplotlib.pyplot as plt

# ─── Time setup ─────────────────────────────
T = 1000
time = np.linspace(0, 100, T)
dt = time[1] - time[0]

# ─── Environment parameters ────────────────
t_c = 60
WARM_A = 0.01
WARM_B = 0.0008
Q10 = 2.0

# ─── Population parameters ─────────────────
slow_mass = 10000
fast_mass = 10
alpha = 0.15
K_base = 5.0

# ─── Network factors (abstracted) ─────────
np.random.seed(42)
network_factors = np.random.rand(8, T) * 0.2 + 0.8
network_support = np.mean(network_factors, axis=0)

# ─── Shadow thresholds ─────────────────────
slow_thresh = 0.3
fast_thresh = 0.3

# ─── Leverage strategies ──────────────────
def run_sim(strategy):
    slow_pop = np.zeros(T)
    fast_pop = np.zeros(T)
    slow_pop[0] = 1.0
    fast_pop[0] = 1.0
    slow_energy = np.zeros(T)
    fast_energy = np.zeros(T)

    for i in range(1, T):
        t = time[i]
        temperature = WARM_A * t + WARM_B * t**2
        met_mult = Q10 ** (temperature / 10)

        slow_energy[i] = network_support[i] ** (alpha * np.log10(slow_mass)) / met_mult - 0.5
        fast_energy[i] = network_support[i] ** (alpha * np.log10(fast_mass)) / met_mult - 0.3

        # Determine leverage
        leverage = 0.0
        if strategy == 'constant':
            leverage = 0.05
        elif strategy == 'pulse':
            if slow_pop[i-1] < slow_thresh or fast_pop[i-1] < fast_thresh:
                leverage = 0.1
        elif strategy == 'adaptive':
            leverage = max(0, slow_thresh - slow_pop[i-1]) * 0.3 + max(0, fast_thresh - fast_pop[i-1]) * 0.3

        conn_slow = network_support[i] + leverage
        conn_fast = network_support[i] + leverage

        K_slow = K_base * conn_slow + 0.1
        K_fast = K_base * conn_fast + 0.1

        r_slow = np.tanh(slow_energy[i] + leverage)
        r_fast = np.tanh(fast_energy[i] + leverage)

        slow_pop[i] = min(max(slow_pop[i-1] + r_slow * slow_pop[i-1] * (1 - slow_pop[i-1]/K_slow) * dt, 0), K_slow)
        fast_pop[i] = min(max(fast_pop[i-1] + r_fast * fast_pop[i-1] * (1 - fast_pop[i-1]/K_fast) * dt, 0), K_fast)

    return slow_pop, fast_pop

# ─── Run simulations ──────────────────────
strategies = ['no_leverage', 'constant', 'pulse', 'adaptive']
results = {}

for s in strategies:
    results[s] = run_sim(s)

# ─── Visualization ─────────────────────────
plt.figure(figsize=(16,5))

colors = {'no_leverage':'black','constant':'blue','pulse':'orange','adaptive':'green'}

# Slow population
plt.subplot(1,2,1)
for s in strategies:
    plt.plot(time, results[s][0], color=colors[s], label=s)
plt.axhline(y=slow_thresh, color='red', linestyle='--', alpha=0.7, label='Threshold')
plt.title("Slow Population: Different Leverage Strategies")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid(alpha=0.3)

# Fast population
plt.subplot(1,2,2)
for s in strategies:
    plt.plot(time, results[s][1], color=colors[s], label=s)
plt.axhline(y=fast_thresh, color='red', linestyle='--', alpha=0.7, label='Threshold')
plt.title("Fast Population: Different Leverage Strategies")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
