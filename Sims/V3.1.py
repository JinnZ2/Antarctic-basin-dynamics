import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Constants ─────────────────────────────
PHI = 1.6180339887

# ── Shadow basin parameters ──────────────
class ShadowConfig:
    # Primary basin
    k_perc      = 0.4
    t_c         = 60
    
    # Shadow basin (slightly strengthened)
    C_seed      = 0.15
    beta_shadow = 0.05
    
    # Coupling
    gamma_max   = 1.2
    coupling_k  = 8.0
    
    # Nudge engine
    nudge_scale = 0.02
    
    # Metabolic
    WARM_A      = 0.01
    WARM_B      = 0.0008
    Q10         = 2.0
    baseline_s  = 0.5
    baseline_f  = 0.3


# ── Core Functions ────────────────────────
def percolation_decay(t, config):
    return 1 / (1 + np.exp(config.k_perc * (t - config.t_c)))

def shadow_growth(t, config):
    growth = config.C_seed * (1 - np.exp(-config.beta_shadow * t))
    phi_mod = 0.5 + 0.5 * np.sin(PHI * t * 0.1)
    return growth * phi_mod

def coupling_strength(primary, config):
    proximity = np.tanh(config.coupling_k * (0.6 - primary))
    return max(0, proximity) * config.gamma_max

def nudge_leverage(primary, shadow, t, config):
    risk = max(0, 0.7 - primary)
    shadow_potential = shadow / (1 + np.exp(-config.beta_shadow * t))
    return risk * shadow_potential * config.nudge_scale

def metabolic_multiplier(t, config):
    temp = config.WARM_A * t + config.WARM_B * t**2
    return config.Q10 ** (temp / 10)


# ── Globals ───────────────────────────────
T = 1000
time = np.linspace(0, 100, T)
dt = time[1] - time[0]

slow_mass = 10000
fast_mass = 10
alpha = 0.15
K_base = 5.0


# ── Main Simulation ───────────────────────
def run_shadow_basin(config):
    slow_pop = np.zeros_like(time)
    fast_pop = np.zeros_like(time)
    slow_pop[0] = fast_pop[0] = 1.0
    
    slow_energy = np.zeros_like(time)
    fast_energy = np.zeros_like(time)
    C_total = np.zeros_like(time)
    
    for i in range(1, len(time)):
        t = time[i]
        
        primary = percolation_decay(t, config)
        shadow  = shadow_growth(t, config)
        coupling = coupling_strength(primary, config)
        nudge   = nudge_leverage(primary, shadow, t, config)
        
        # Clamp total connectivity to [0,1]
        C_total[i] = min(1.0, primary + coupling * shadow + nudge)
        
        met_mult = metabolic_multiplier(t, config)
        
        slow_energy[i] = C_total[i] ** (alpha * np.log10(slow_mass)) / met_mult - config.baseline_s
        fast_energy[i] = C_total[i] ** (alpha * np.log10(fast_mass)) / met_mult - config.baseline_f
        
        K_slow = K_base * C_total[i] + 0.1
        K_fast = K_base * C_total[i] + 0.1
        
        r_slow = np.tanh(slow_energy[i])
        r_fast = np.tanh(fast_energy[i])
        
        slow_pop[i] = min(max(
            slow_pop[i-1] + r_slow * slow_pop[i-1] *
            (1 - slow_pop[i-1]/K_slow) * dt, 0), K_slow)
        
        fast_pop[i] = min(max(
            fast_pop[i-1] + r_fast * fast_pop[i-1] *
            (1 - fast_pop[i-1]/K_fast) * dt, 0), K_fast)
    
    return slow_pop, fast_pop, C_total, slow_energy, fast_energy


# ── Run Simulation ────────────────────────
config = ShadowConfig()
slow_pop, fast_pop, C_total, slow_e, fast_e = run_shadow_basin(config)


# ── Visualization ─────────────────────────
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Populations
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(time, slow_pop, label='Slow', lw=2)
ax1.plot(time, fast_pop, label='Fast', lw=2)
ax1.axvline(config.t_c, ls='--')
ax1.set_title('Population Trajectories')
ax1.legend()
ax1.grid(alpha=0.3)

# Connectivity
ax2 = fig.add_subplot(gs[0,1])
primary = percolation_decay(time, config)
shadow  = shadow_growth(time, config)
ax2.plot(time, primary, ls='--', label='Primary')
ax2.plot(time, shadow, label='Shadow')
ax2.plot(time, C_total, lw=3, label='Total')
ax2.axvline(config.t_c, ls='--')
ax2.set_title('Connectivity Cascade')
ax2.legend()
ax2.grid(alpha=0.3)

# Survival window
ax3 = fig.add_subplot(gs[0,2])
met_mult = metabolic_multiplier(time, config)
survival_slow = C_total > config.baseline_s * met_mult
survival_fast = C_total > config.baseline_f * met_mult
ax3.fill_between(time, survival_slow.astype(int), alpha=0.3, label='Slow')
ax3.fill_between(time, survival_fast.astype(int), alpha=0.3, label='Fast')
ax3.axvline(config.t_c, ls='--')
ax3.set_ylim(0,1.2)
ax3.set_title('Survival Window')
ax3.legend()

# Energy
ax4 = fig.add_subplot(gs[1,:])
ax4.plot(time, slow_e, label='Slow energy')
ax4.plot(time, fast_e, label='Fast energy')
ax4.axhline(0, alpha=0.3)
ax4.axvline(config.t_c, ls='--')
ax4.set_title('Energy Balance')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Simulation complete.")
print(f"Final slow population: {slow_pop[-1]:.3f}")
print(f"Final fast population: {fast_pop[-1]:.3f}")
