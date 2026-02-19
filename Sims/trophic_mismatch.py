import numpy as np
import matplotlib.pyplot as plt
from core import metabolic_multiplier, trophic_energy_interception
import json

with open('../parameters.json') as f:
    P = json.load(f)

warming_range = np.linspace(0, 6, 200)

# Energy available to apex predator under warming
# as microbial interception accelerates faster than
# primary production increases

# Primary production modest increase with warming
def primary_production(delta_T, base=100.0):
    return base * (1 + 0.05 * delta_T)

# Microbial multiplier accelerates fastest
micro_multipliers = [
    metabolic_multiplier(P['Q10_microbial'], dT) 
    for dT in warming_range
]

# Energy reaching apex after interception at each level
apex_energy = [
    trophic_energy_interception(
        primary_production(dT),
        metabolic_multiplier(P['Q10_microbial'], dT),
        P['trophic_transfer_efficiency'],
        trophic_levels=3
    )
    for dT in warming_range
]

# Apex metabolic demand increasing simultaneously
apex_demand_multipliers = [
    metabolic_multiplier(P['Q10_apex'], dT)
    for dT in warming_range
]

# Normalize to baseline
baseline_energy = apex_energy[0]
baseline_demand = 1.0

apex_energy_normalized = [e / baseline_energy 
                           for e in apex_energy]
apex_demand_normalized = apex_demand_multipliers

# Energy surplus/deficit
energy_balance = [
    e - d for e, d in 
    zip(apex_energy_normalized, apex_demand_normalized)
]

# Find crossover point
crossover_idx = next(
    (i for i, b in enumerate(energy_balance) if b < 0), 
    None
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: supply vs demand divergence
ax1.plot(warming_range, apex_energy_normalized,
         'blue', linewidth=2, label='Energy supply to apex')
ax1.plot(warming_range, apex_demand_normalized,
         'red', linewidth=2, label='Apex metabolic demand')

if crossover_idx:
    crossover_T = warming_range[crossover_idx]
    ax1.axvline(x=crossover_T, color='black',
                linestyle='--', alpha=0.7)
    ax1.annotate(f'Budget closes\n~+{crossover_T:.1f}°C',
                (crossover_T, 1.1), fontsize=8)

ax1.set_xlabel('Temperature increase (°C)')
ax1.set_ylabel('Normalized rate (baseline=1.0)')
ax1.set_title('Supply vs demand:\napex predator energy budget')
ax1.legend()

# Panel 2: differential acceleration across trophic levels
mid_multipliers = [
    metabolic_multiplier(P['Q10_midtrophic'], dT)
    for dT in warming_range
]
apex_multipliers = [
    metabolic_multiplier(P['Q10_apex'], dT)
    for dT in warming_range
]

ax2.plot(warming_range, micro_multipliers,
         color='darkred', linewidth=2, label=f"Microbial (Q10={P['Q10_microbial']})")
ax2.plot(warming_range, mid_multipliers,
         color='orange', linewidth=2, label=f"Mid-trophic (Q10={P['Q10_midtrophic']})")
ax2.plot(warming_range, apex_multipliers,
         color='darkblue', linewidth=2, label=f"Apex (Q10={P['Q10_apex']})")

ax2.fill_between(warming_range, apex_multipliers, micro_multipliers,
                  alpha=0.15, color='red',
                  label='Mismatch gap')

ax2.set_xlabel('Temperature increase (°C)')
ax2.set_ylabel('Metabolic rate multiplier')
ax2.set_title('Differential acceleration\nby trophic level')
ax2.legend()

# Panel 3: energy balance over time with stochastic prey variation
np.random.seed(42)
time_steps = np.linspace(0, 100, 500)
warming_trajectory = np.linspace(0, 4, 500)

balance_deterministic = []
balance_stochastic = []

for i, dT in enumerate(warming_trajectory):
    supply = trophic_energy_interception(
        primary_production(dT),
        metabolic_multiplier(P['Q10_microbial'], dT),
        P['trophic_transfer_efficiency'],
        trophic_levels=3
    ) / baseline_energy

    demand = metabolic_multiplier(P['Q10_apex'], dT)

    noise = np.random.normal(0, 0.05)
    balance_deterministic.append(supply - demand)
    balance_stochastic.append(supply - demand + noise)

ax3.plot(time_steps, balance_deterministic,
         'blue', linewidth=2, label='Deterministic')
ax3.plot(time_steps, balance_stochastic,
         'lightblue', linewidth=1, alpha=0.7, label='With prey variation')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5,
            label='Budget breakeven')
ax3.fill_between(time_steps, balance_stochastic, 0,
                  where=[b < 0 for b in balance_stochastic],
                  alpha=0.3, color='red', label='Deficit periods')

ax3.set_xlabel('Years')
ax3.set_ylabel('Energy balance (normalized)')
ax3.set_title('Energy budget over warming trajectory\nwith stochastic prey variation')
ax3.legend(fontsize=7)

plt.tight_layout()
plt.savefig('trophic_mismatch_output.png', dpi=150)
plt.show()
