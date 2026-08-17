"""Forcing isolation experiments. Implements additions.md item 8.

Drivers run one at a time, then together, then as an interaction
surface.

The reason this needed the structural changes first: a static
viability index built from supply, demand and reach is
*multiplicatively separable* in warming and fragmentation. Its
interaction term is exactly zero, by construction, no matter what
numbers go in. Running that experiment would have produced a clean
result that meant nothing.

Age structure breaks the separability, and does so for a reason that
can be stated in one line: warming shortens generation time, and a
given supply shortfall costs more per year when generations are
short. So the outcome variable here is the population growth rate
from the age-structured model, not a static index.

Reported as percentage change per century, because the animal in
question matures at 150 years and annual rates are unreadable at
that timescale.

Run from anywhere:  python Sims/forcing_isolation.py

Structure and caveats: Docs/structure.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import population as pop
import spatial as sp
from core import (
    metabolic_multiplier,
    trophic_energy_interception,
    dynamic_transfer_efficiency,
    oxygen_availability,
    mass_dependent_connectivity,
)

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

rng = np.random.default_rng(7)
TROPHIC_LEVELS = 3
BASE_BOND_P = P['patch_autocorrelation']

# The lattice is sampled once and interpolated. Re-running percolation
# inside every grid cell would cost a great deal and buy only sampling
# noise.
P_GRID = np.linspace(0.20, 0.98, 40)
GIANT = sp.percolation_sweep(P_GRID, rows=8, cols=60, reps=12, rng=rng)

FECUNDITY = pop.calibrate_fecundity()


def reach(bond_p):
    """Fraction of habitat a wide-ranging animal can actually use."""
    return mass_dependent_connectivity(
        np.interp(bond_p, P_GRID, GIANT),
        P['body_mass_scaling_exponent'],
        P['connectivity_mass_sensitivity'])


def supply_multiplier(delta_T, bond_p, efficiency=True, microbial=True,
                      oxygen=True, fragmentation=True):
    """Energy reaching the apex, relative to baseline.

    Each thermal channel can be switched off independently so its
    contribution can be read alone:

    efficiency     transfer efficiency declining with warming
    microbial      the Q10 differential intercepting energy low down
    oxygen         demand rising as supply falls
    fragmentation  habitat continuity
    """
    def available(dT, use_efficiency, use_microbial, use_oxygen):
        micro = metabolic_multiplier(
            P['Q10_microbial'] if use_microbial else P['Q10_apex'],
            dT if use_microbial else 0.0)
        tte = dynamic_transfer_efficiency(
            P['trophic_transfer_efficiency'],
            dT if use_efficiency else 0.0,
            P['tte_warming_sensitivity_per_C'], P['tte_floor'])
        supply = trophic_energy_interception(1.0, micro, tte, TROPHIC_LEVELS)
        o2 = oxygen_availability(
            dT if use_oxygen else 0.0, P['oxygen_baseline_saturation'],
            P['deep_oxygen_committed_loss_fraction'])
        demand = metabolic_multiplier(
            P['Q10_apex'], dT if use_oxygen else 0.0) / o2
        return supply / demand

    numerator = available(delta_T, efficiency, microbial, oxygen)
    denominator = available(0.0, True, True, True)

    spatial_term = (reach(bond_p) if fragmentation else reach(BASE_BOND_P))
    return (numerator / denominator) * (spatial_term / reach(BASE_BOND_P))


def growth(delta_T, bond_p, age_compression=True, **flags):
    """Annual population growth rate under a forcing combination."""
    s = supply_multiplier(delta_T, bond_p, **flags)
    dT_age = delta_T if age_compression else 0.0
    max_age, maturity, bounds = pop.warmed_ages(
        dT_age, P['Q10_apex'],
        decoupling=P['longevity_maintenance_decoupling'])
    lx, mx = pop.life_table(max_age, maturity, bounds,
                            fecundity=FECUNDITY * max(s, 1e-12))
    return pop.growth_rate_from_life_table(lx, mx)


def per_century(lam):
    """Annual rate expressed as percentage change per century."""
    return (np.asarray(lam, dtype=float) ** 100 - 1.0) * 100.0


BASE_LAMBDA = growth(0.0, BASE_BOND_P)


# --- Isolation runs -------------------------------------------------
# One shared forcing axis so the runs are directly comparable.

steps = np.linspace(0.0, 1.0, 50)
warming_axis = steps * 6.0
bond_axis = BASE_BOND_P - steps * 0.45

OFF = dict(efficiency=False, microbial=False, oxygen=False,
           fragmentation=False)

runs = {
    'thermal only': lambda dT, p: growth(
        dT, BASE_BOND_P, fragmentation=False),
    'fragmentation only': lambda dT, p: growth(
        0.0, p, age_compression=False, **{**OFF, 'fragmentation': True}),
    'microbial only': lambda dT, p: growth(
        dT, BASE_BOND_P, age_compression=False,
        **{**OFF, 'microbial': True}),
    'efficiency only': lambda dT, p: growth(
        dT, BASE_BOND_P, age_compression=False,
        **{**OFF, 'efficiency': True}),
    'combined': lambda dT, p: growth(dT, p),
}

curves = {label: np.array([fn(dT, p) for dT, p in zip(warming_axis, bond_axis)])
          for label, fn in runs.items()}

# Additive null in log growth rate, which is the natural scale: two
# independent multiplicative effects add here.
log_base = np.log(BASE_LAMBDA)
additive = np.exp(np.log(curves['thermal only'])
                  + np.log(curves['fragmentation only']) - log_base)


# --- Interaction surface --------------------------------------------

n_grid = 45
dT_grid = np.linspace(0.0, 6.0, n_grid)
p_grid = np.linspace(BASE_BOND_P, 0.30, n_grid)
DT, PP = np.meshgrid(dT_grid, p_grid, indexing='ij')

surface = np.array([[growth(dT, p) for p in p_grid] for dT in dT_grid])
thermal_edge = np.array([growth(dT, BASE_BOND_P, fragmentation=False)
                         for dT in dT_grid])[:, None]
frag_edge = np.array([growth(0.0, p, age_compression=False,
                             **{**OFF, 'fragmentation': True})
                      for p in p_grid])[None, :]

additive_surface = np.exp(np.log(thermal_edge) + np.log(frag_edge) - log_base)
synergy = per_century(surface) - per_century(additive_surface)


# --- Plots ----------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2))

for label, curve in curves.items():
    style = '-' if label == 'combined' else '--'
    width = 2.6 if label == 'combined' else 1.4
    ax1.plot(steps, per_century(curve), style, lw=width, label=label)
ax1.plot(steps, per_century(additive), ':', color='k', lw=2,
         label='additive null')
ax1.axhline(0.0, color='grey', lw=1)
ax1.set_xlabel('Fraction along the joint forcing ramp')
ax1.set_ylabel('Population change per century (%)')
ax1.set_title('Isolated drivers vs the combined run')
ax1.legend(fontsize=8)

contour = ax2.contourf(DT, PP, per_century(surface), levels=18, cmap='viridis')
fig.colorbar(contour, ax=ax2, label='% per century')
ax2.contour(DT, PP, per_century(surface), levels=[0], colors='w',
            linewidths=2)
ax2.set_xlabel('Temperature increase (°C)')
ax2.set_ylabel('Bond occupation probability')
ax2.set_title('Interaction surface (white = replacement)')

limit = float(np.abs(synergy).max()) or 1.0
diverging = ax3.contourf(DT, PP, synergy,
                         levels=np.linspace(-limit, limit, 19), cmap='RdBu_r')
fig.colorbar(diverging, ax=ax3, label='observed − additive (% per century)')
ax3.set_xlabel('Temperature increase (°C)')
ax3.set_ylabel('Bond occupation probability')
ax3.set_title('Departure from additivity')

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'forcing_isolation_output.png', dpi=140)


# --- Printed summary ------------------------------------------------

print(f'Baseline growth rate {BASE_LAMBDA:.6f} per year '
      f'({per_century(BASE_LAMBDA):+.2f}% per century)\n')
print(f'{"driver":<22}{"50% ramp":>12}{"full ramp":>12}   (% per century)')
for label, curve in curves.items():
    print(f'{label:<22}{per_century(curve[len(curve) // 2]):>12.2f}'
          f'{per_century(curve[-1]):>12.2f}')
print(f'{"additive null":<22}{per_century(additive[len(additive) // 2]):>12.2f}'
      f'{per_century(additive[-1]):>12.2f}')

gap = per_century(curves['combined'][-1]) - per_century(additive[-1])
print(f'\nCombined minus additive at full ramp: {gap:+.2f} '
      f'percentage points per century')
if gap < 0:
    print('  Negative: the drivers compound. Warming shortens generation')
    print('  time, so the supply shortfall fragmentation causes is paid off')
    print('  over fewer years and costs more per year. Neither isolated run')
    print('  contains this term.')
else:
    print('  Positive: the drivers overlap rather than compound at this')
    print('  point on the ramp — each is already removing what the other')
    print('  would have removed.')

worst = np.unravel_index(np.argmin(synergy), synergy.shape)
print(f'\nStrongest departure from additivity:')
print(f'  {synergy[worst]:+.2f} points at dT = {dT_grid[worst[0]]:.1f} °C, '
      f'bond p = {p_grid[worst[1]]:.2f}')

print('\nDrivers ranked by damage at full ramp:')
for label, value in sorted(((k, v[-1]) for k, v in curves.items()
                            if k != 'combined'), key=lambda kv: kv[1]):
    print(f'  {label:<22}{per_century(value):>10.2f}')

print('\nThe thermal channel decomposes further: transfer efficiency')
print('dominates the microbial differential by a wide margin, because')
print('efficiency enters once per trophic level and interception enters')
print('once. Before efficiency was made dynamic, the microbial term was')
print('carrying the whole mismatch mechanism on its own.')

plt.show()
