**note the header: that v2 introduces percolation connectivity, accelerating forcing, and mass-dependent connectivity scaling.**

 import numpy as np
import matplotlib.pyplot as plt

# Higher resolution time axis
T = 1000
time = np.linspace(0, 100, T)
dt = time[1] - time[0]

# Parameters
k = 0.4
t_c = 60
WARM_A = 0.01
WARM_B = 0.0008
Q10 = 2.0

slow_mass = 10000
fast_mass = 10
alpha = 0.15
K_base = 5.0

# Initialize populations
slow_pop = np.zeros(T)
fast_pop = np.zeros(T)
slow_pop[0] = 1.0
fast_pop[0] = 1.0

slow_energy = np.zeros(T)
fast_energy = np.zeros(T)
ac_series = np.zeros(T)

for i in range(1, T):
    t = time[i]
    
    # Accelerating warming
    temperature = WARM_A * t + WARM_B * t**2
    
    # Percolation-style connectivity
    ac = 1 / (1 + np.exp(k * (t - t_c)))
    ac_series[i] = ac
    
    # Q10 metabolic scaling
    met_mult = Q10 ** (temperature / 10)
    
    # Connectivity scaling
    slow_conn = ac ** (alpha * np.log10(slow_mass))
    fast_conn = ac ** (alpha * np.log10(fast_mass))
    
    # Energy balance
    slow_energy[i] = slow_conn / met_mult - 0.5
    fast_energy[i] = fast_conn / met_mult - 0.3
    
    # Connectivity-linked carrying capacity
    K_slow = K_base * slow_conn + 0.1
    K_fast = K_base * fast_conn + 0.1
    
    # Bounded intrinsic rate
    r_slow = np.tanh(slow_energy[i])
    r_fast = np.tanh(fast_energy[i])
    
    # Clamp populations to carrying capacity FIRST (ecological feasibility)
    slow_pop[i] = min(slow_pop[i-1], K_slow)
    fast_pop[i] = min(fast_pop[i-1], K_fast)
    
    # THEN apply logistic growth with timestep scaling
    slow_growth = r_slow * slow_pop[i] * (1 - slow_pop[i] / K_slow)
    fast_growth = r_fast * fast_pop[i] * (1 - fast_pop[i] / K_fast)
    
    # Update with dt scaling, re-clamp to prevent overshoot
    slow_pop[i] = min(max(slow_pop[i] + slow_growth * dt, 0), K_slow)
    fast_pop[i] = min(max(fast_pop[i] + fast_growth * dt, 0), K_fast)

# Plot 1 – Population
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(time, slow_pop, label='Slow (large mass)')
plt.plot(time, fast_pop, label='Fast (small mass)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title("Population Trajectories\n(Feasibility-Enforced)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2 – Energy
plt.subplot(1, 3, 2)
plt.plot(time, slow_energy, label='Slow energy')
plt.plot(time, fast_energy, label='Fast energy')
plt.title("Energy Balance Over Time")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3 – Connectivity
plt.subplot(1, 3, 3)
plt.plot(time, ac_series, label='Autocorrelation')
plt.axvline(x=t_c, color='r', linestyle='--', alpha=0.7, label=f't_c={t_c}')
plt.title("Percolation Connectivity\n(t_c marks threshold)")
plt.xlabel("Time")
plt.ylabel("Connectivity (ac)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
 
