import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import json
from core import (
    metabolic_multiplier,
    adjusted_lifespan,
    trophic_energy_interception,
    patch_viability
)

with open('../Model/parameters.json') as f:
    P = json.load(f)

# ── Agent definitions ──────────────────────────────────────
# Two life-history strategies sharing the same landscape
# Slow integrator: high mass, long lifespan, low Q10 sensitivity
# Fast cycler:     low mass, short lifespan, higher Q10 sensitivity

STRATEGIES = {
    'slow_integrator': {
        'body_mass':       10000,
        'base_lifespan':   300,
        'Q10':             2.5,
        'base_population': 100,
        'color':           'darkblue',
        'label':           'Slow integrator (megafaunal)'
    },
    'fast_cycler': {
        'body_mass':       50,
        'base_lifespan':   15,
        'Q10':             3.0,
        'base_population': 1000,
        'color':           'darkorange',
        'label':           'Fast cycler (small bodied)'
    }
}

# ── Simulation parameters ──────────────────────────────────
TIME_STEPS     = 300          # years
WARMING_RATE   = 0.02         # °C per year  (~2°C over 100 yrs)
FRAG_RATE      = P['fragmentation_rate'] * 0.005
BASE_AC        = P['patch_autocorrelation']
BASE_PP        = 100.0        # baseline primary production

np.random.seed(7)

# ── State arrays ───────────────────────────────────────────
time         = np.arange(TIME_STEPS)
temperature  = P['baseline_temp_C'] + WARMING_RATE * time
delta_T      = temperature - P['baseline_temp_C']
autocorr     = BASE_AC * np.exp(-FRAG_RATE * time)

# Primary production: modest increase then plateau
def primary_production(dT):
    return BASE_PP * (1 + 0.04 * dT - 0.002 * dT**2)

# ── Per-timestep dynamics ──────────────────────────────────
results = {k: {
    'population':    np.zeros(TIME_STEPS),
    'lifespan':      np.zeros(TIME_STEPS),
    'energy_balance':np.zeros(TIME_STEPS),
    'viable':        np.zeros(TIME_STEPS, dtype=bool)
} for k in STRATEGIES}

# Initialise populations
for k in STRATEGIES:
    results[k]['population'][0] = STRATEGIES[k]['base_population']

for t in range(TIME_STEPS):
    dT   = delta_T[t]
    ac   = autocorr[t]
    pp   = primary_production(dT)
    micro_mult = metabolic_multiplier(P['Q10_microbial'], dT)

    # Energy reaching apex level after microbial interception
    apex_supply = trophic_energy_interception(
        pp, micro_mult,
        P['trophic_transfer_efficiency'],
        trophic_levels=3
    )
    baseline_supply = trophic_energy_interception(
        BASE_PP, 1.0,
        P['trophic_transfer_efficiency'],
        trophic_levels=3
    )
    supply_ratio = apex_supply / baseline_supply

    for k, S in STRATEGIES.items():
        met_mult = metabolic_multiplier(S['Q10'], dT)
        lifespan = adjusted_lifespan(S['base_lifespan'], met_mult)
        viable   = patch_viability(
            ac, S['body_mass'],
            P['body_mass_scaling_exponent']
        )

        # Energy balance: supply adjusted for connectivity
        # Fragmentation reduces effective prey encounter rate
        connectivity_factor = ac ** 0.5
        effective_supply    = supply_ratio * connectivity_factor
        energy_balance      = effective_supply - met_mult / metabolic_multiplier(
            S['Q10'], 0
        )

        results[k]['lifespan'][t]       = lifespan
        results[k]['viable'][t]         = viable
        results[k]['energy_balance'][t] = energy_balance

        # Population dynamics
        if t > 0:
            prev_pop = results[k]['population'][t-1]

            # Recruitment scales with energy balance and lifespan
            # Shorter lifespan → faster turnover → higher recruitment rate
            turnover_rate = 1.0 / lifespan
            recruitment   = prev_pop * turnover_rate * max(0, effective_supply)

            # Mortality scales with metabolic demand and connectivity
            base_mortality = turnover_rate
            stress_mortality = max(0, -energy_balance) * 0.1
            connectivity_mortality = 0 if viable else 0.05

            mortality = prev_pop * (
                base_mortality +
                stress_mortality +
                connectivity_mortality
            )

            # Stochastic environmental variation
            noise = np.random.normal(1.0, 0.03)

            new_pop = (prev_pop + recruitment - mortality) * noise
            results[k]['population'][t] = max(0, new_pop)

# ── Normalise populations for comparison ──────────────────
for k in STRATEGIES:
    base = results[k]['population'][0]
    if base > 0:
        results[k]['population_norm'] = (
            results[k]['population'] / base
        )
    else:
        results[k]['population_norm'] = results[k]['population']

# ── Plotting ───────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0a0a0a')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.45,
    wspace=0.35
)

ax_pop    = fig.add_subplot(gs[0, :])   # full width: population
ax_life   = fig.add_subplot(gs[1, 0])
ax_energy = fig.add_subplot(gs[1, 1])
ax_phase  = fig.add_subplot(gs[1, 2])
ax_temp   = fig.add_subplot(gs[2, 0])
ax_ac     = fig.add_subplot(gs[2, 1])
ax_basin  = fig.add_subplot(gs[2, 2])

DARK = '#0a0a0a'
GRID = '#2a2a2a'

for ax in [ax_pop, ax_life, ax_energy, ax_phase,
           ax_temp, ax_ac, ax_basin]:
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.title.set_color('#dddddd')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

# Panel 1: normalised population trajectories
for k, S in STRATEGIES.items():
    ax_pop.plot(
        time,
        results[k]['population_norm'],
        color=S['color'], linewidth=2,
        label=S['label']
    )
    # Mark viability threshold crossings
    viable = results[k]['viable']
    crossings = np.where(np.diff(viable.astype(int)) < 0)[0]
    for cx in crossings:
        ax_pop.axvline(
            x=time[cx], color=S['color'],
            linestyle=':', alpha=0.5, linewidth=1
        )
        ax_pop.annotate(
            'connectivity\nlost',
            (time[cx], 0.5),
            color=S['color'], fontsize=6,
            xytext=(3, 0), textcoords='offset points'
        )

ax_pop.axhline(y=1.0, color='#555555',
               linestyle='--', linewidth=1)
ax_pop.set_xlabel('Years')
ax_pop.set_ylabel('Population (normalised)')
ax_pop.set_title(
    'Integrative dynamics: population trajectories under '
    'simultaneous warming + fragmentation'
)
ax_pop.legend(loc='upper right', fontsize=8,
              facecolor='#1a1a1a', labelcolor='white')

# Panel 2: lifespan compression
for k, S in STRATEGIES.items():
    norm_life = (results[k]['lifespan'] /
                 results[k]['lifespan'][0])
    ax_life.plot(time, norm_life,
                 color=S['color'], linewidth=1.5)

ax_life.axhline(y=1.0, color='#555555',
                linestyle='--', linewidth=1)
ax_life.set_xlabel('Years')
ax_life.set_ylabel('Lifespan (normalised)')
ax_life.set_title('Lifespan compression')

# Panel 3: energy balance
for k, S in STRATEGIES.items():
    ax_energy.plot(
        time, results[k]['energy_balance'],
        color=S['color'], linewidth=1.5
    )
    ax_energy.fill_between(
        time, results[k]['energy_balance'], 0,
        where=results[k]['energy_balance'] < 0,
        alpha=0.2, color=S['color']
    )

ax_energy.axhline(y=0, color='red',
                  linestyle='--', linewidth=1)
ax_energy.set_xlabel('Years')
ax_energy.set_ylabel('Energy balance')
ax_energy.set_title('Energy budget\n(below zero = deficit)')

# Panel 4: phase portrait — population vs energy balance
for k, S in STRATEGIES.items():
    sc = ax_phase.scatter(
        results[k]['energy_balance'],
        results[k]['population_norm'],
        c=time, cmap='plasma',
        s=8, alpha=0.6,
        label=S['label']
    )

ax_phase.axvline(x=0, color='red',
                 linestyle='--', linewidth=1, alpha=0.5)
ax_phase.set_xlabel('Energy balance')
ax_phase.set_ylabel('Population (norm)')
ax_phase.set_title('Phase portrait\n(color = time)')

# Panel 5: temperature trajectory
ax_temp.plot(time, temperature,
             color='#ff6b6b', linewidth=2)
ax_temp.fill_between(time, P['baseline_temp_C'],
                     temperature, alpha=0.2, color='#ff6b6b')
ax_temp.set_xlabel('Years')
ax_temp.set_ylabel('Temperature (°C)')
ax_temp.set_title('Warming trajectory')

# Panel 6: spatial autocorrelation decay
ax_ac.plot(time, autocorr,
           color='#6bff9e', linewidth=2)
ax_ac.axhline(y=0.5, color='red',
              linestyle='--', linewidth=1,
              label='Viability threshold')

# Mark threshold crossing
threshold_cross = np.where(autocorr < 0.5)[0]
if len(threshold_cross) > 0:
    tc = threshold_cross[0]
    ax_ac.axvline(x=time[tc], color='red',
                  linestyle=':', alpha=0.7)
    ax_ac.annotate(
        f'~yr {time[tc]}',
        (time[tc], 0.52),
        color='red', fontsize=7
    )

ax_ac.set_xlabel('Years')
ax_ac.set_ylabel('Spatial autocorrelation')
ax_ac.set_title('Habitat connectivity decay')
ax_ac.legend(fontsize=7, facecolor='#1a1a1a',
             labelcolor='white')

# Panel 7: basin depth proxy
# Basin depth = energy balance × connectivity × lifespan ratio
# Composite stability index
slow = results['slow_integrator']
fast = results['fast_cycler']

slow_basin = (
    np.clip(slow['energy_balance'], 0, None) *
    autocorr *
    (slow['lifespan'] / slow['lifespan'][0])
)
fast_basin = (
    np.clip(fast['energy_balance'], 0, None) *
    autocorr *
    (fast['lifespan'] / fast['lifespan'][0])
)

ax_basin.plot(time, slow_basin,
              color='darkblue', linewidth=2,
              label='Slow integrator basin depth')
ax_basin.plot(time, fast_basin,
              color='darkorange', linewidth=2,
              label='Fast cycler basin depth')
ax_basin.fill_between(time, slow_basin,
                      alpha=0.15, color='darkblue')

ax_basin.set_xlabel('Years')
ax_basin.set_ylabel('Composite stability index')
ax_basin.set_title('Basin depth proxy\n(energy × connectivity × lifespan)')
ax_basin.legend(fontsize=7, facecolor='#1a1a1a',
                labelcolor='white')

fig.suptitle(
    'Antarctic Basin Dynamics — Integrative Model\n'
    'Simultaneous warming, trophic mismatch, '
    'fragmentation, lifespan compression',
    color='white', fontsize=11, y=0.98
)

plt.savefig('integrative_output.png', dpi=150,
            facecolor=fig.get_facecolor())
plt.show()

print("\n── Summary statistics ──")
for k, S in STRATEGIES.items():
    final_pop  = results[k]['population_norm'][-1]
    final_life = (results[k]['lifespan'][-1] /
                  results[k]['lifespan'][0])
    deficit_yrs = np.sum(results[k]['energy_balance'] < 0)
    print(f"\n{S['label']}")
    print(f"  Final population (norm):  {final_pop:.3f}")
    print(f"  Lifespan retained:        {final_life:.1%}")
    print(f"  Years in energy deficit:  {deficit_yrs}")
