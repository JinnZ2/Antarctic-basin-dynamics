"""The three structural changes, demonstrated against the scalar model.

Spatial representation, age structure, and coupled basins are
each shown next to the quantity the scalar model used in their
place. In every panel the question is the same: what does the
scalar version fail to show?

Run from anywhere:  python Sims/structural_v4.py

Structure and caveats: Docs/structure.md
Literature: Docs/literature.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import basins
import population as pop
import spatial as sp
from core import metabolic_multiplier, percolation_connectivity

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

rng = np.random.default_rng(11)
ROWS, COLS, REPS = 8, 60, 10

fig, axes = plt.subplots(3, 2, figsize=(14, 14))
(ax1, ax2), (ax3, ax4), (ax5, ax6) = axes


# --- 1. The percolation threshold is emergent -----------------------
# core.py gets a threshold by writing a sigmoid, which assumes the
# answer. Here bond occupation is swept uniformly and the transition
# falls out of the lattice.
#
# The square lattice is the control: bond percolation on it has a
# known threshold of 0.5, so recovering that value is what makes the
# emergent number checkable rather than decorative. The circumpolar
# strip is the model's actual geometry, and its higher threshold is a
# result, not an error — a habitat band that is thin in depth loses
# continuity at a higher bond probability than a compact region does.

p_values = np.linspace(0.30, 0.85, 45)
giant_square = sp.percolation_sweep(p_values, 60, 60, REPS, rng)
giant = sp.percolation_sweep(p_values, ROWS, COLS, REPS, rng)
p_c_square = sp.critical_probability(p_values, giant_square)
p_c = sp.critical_probability(p_values, giant)

ax1.plot(p_values, giant_square, 'o-', ms=3, color='grey',
         label=f'square 60x60 (control), $p_c$={p_c_square:.2f}')
ax1.plot(p_values, giant, 'o-', ms=3, color='C1',
         label=f'circumpolar strip {ROWS}x{COLS}, $p_c$={p_c:.2f}')
ax1.axvline(0.5, color='k', ls=':', label='analytic $p_c$ = 0.5')
ax1.set_xlabel('Bond occupation probability')
ax1.set_ylabel('Largest connected component (fraction)')
ax1.set_title('Threshold emerges from geometry, not from a chosen curve')
ax1.legend(fontsize=8)


# --- 2. Heterogeneous fragmentation stages the collapse -------------
# Sectors fragment at different rates, so the circumpolar transition
# is smeared into a staged retreat toward the sectors that hold
# together longest.

years = np.arange(0, 91, 3)
conn = sp.connectivity_trajectory(years, ROWS, COLS, reps=REPS, rng=rng)
imposed = percolation_connectivity(
    years, P['connectivity_threshold_time_years'],
    P['connectivity_transition_steepness'], P['patch_autocorrelation'])
scalar = P['patch_autocorrelation'] * np.exp(-P['fragmentation_rate'] * years)

ax2.plot(years, conn, 'o-', ms=3, label='lattice, per-sector rates')
ax2.plot(years, imposed, label='imposed sigmoid (v3)')
ax2.plot(years, scalar, label='exponential decay (v1)')
ax2.set_xlabel('Years')
ax2.set_ylabel('Effective connectivity')
ax2.set_title('Regional heterogeneity turns one threshold into several')
ax2.legend()


# --- 3. Redistribution is not decline -------------------------------
# The circumpolar mean is the number a scalar model carries. It is
# small. The sectors underneath it are not.

supply = sp.sector_supply(years)
redist = sp.redistribution_index(supply)

for i, name in enumerate(sp.SECTOR_NAMES):
    ax3.plot(years, supply[:, i], label=name)
ax3.plot(years, supply.mean(axis=1), 'k--', lw=2.5,
         label='circumpolar mean (what a scalar sees)')
ax3.axhline(1.0, color='grey', ls=':')
ax3.set_xlabel('Years')
ax3.set_ylabel('Mid-trophic supply (relative to baseline)')
ax3.set_title(f'Sector divergence — gross change is {redist[-1]:.1f}x net')
ax3.legend(fontsize=8, ncol=2)


# --- 4. Ecological memory is not lifespan ---------------------------
# geometry.md operationalises memory as lifespan. The projection
# matrix gives generation time and a damping timescale instead, and
# they neither equal lifespan nor track it under warming.

warming = np.linspace(0, 6, 25)
fecundity = pop.calibrate_fecundity()

lifespans, gen_times, memories = [], [], []
for dT in warming:
    max_age, maturity, bounds = pop.warmed_ages(dT, P['Q10_apex'])
    A = pop.build_leslie_matrix(fecundity, max_age, maturity, bounds)
    lx, mx = pop.life_table(max_age, maturity, bounds, fecundity=fecundity)
    lifespans.append(P['baseline_lifespan_years']
                     / metabolic_multiplier(P['Q10_apex'], dT))
    gen_times.append(pop.generation_time(lx, mx))
    memories.append(pop.memory_years(A))

ax4.plot(warming, memories, label='damping timescale (memory)')
ax4.plot(warming, gen_times, label='generation time')
ax4.plot(warming, lifespans, '--', label='lifespan (v1 proxy for memory)')
ax4.set_xlabel('Temperature increase (°C)')
ax4.set_ylabel('Years')
ax4.set_title('Three different quantities, one of which was standing in\n'
              'for the others')
ax4.legend()


# --- 5. Several basins, several depths ------------------------------
# Sector-specific forcing, scaled so the fastest-warming sectors
# approach their thresholds and the slowest do not. Depth is a real
# potential barrier, so it goes negative when the well is gone rather
# than being clipped at zero.

# Forcing ramps over the first half of the run and then holds. The
# hold matters: a cascade takes time to propagate, and a run that
# stops at the moment the ramp ends mistakes "no time left" for
# "did not tip".

n = len(sp.SECTOR_NAMES)
FINAL_FORCING = np.array([0.34, 0.33, 0.31, 0.26, 0.22, 0.42])
steps, dt = 60000, 0.01
sim_time = np.arange(steps) * dt
ramp = np.clip(sim_time / (0.5 * sim_time[-1]), 0.0, 1.0)
forcing = np.outer(ramp, FINAL_FORCING)

x0 = np.full(n, basins.COLD_STATE)
D = basins.ring_coupling(n, 0.06)
traj = basins.simulate(x0, forcing, D, dt)
attribution = basins.cascade_attribution(x0, forcing, D, dt)

# Depth is read off the forcing each basin ACTUALLY experiences,
# external plus the push from neighbours that have already tipped.
# Using the external term alone would understate the damage during a
# cascade, since the neighbour contribution is what eats the barrier.
effective = basins.effective_forcing(traj, forcing, D)
depths = np.array([[basins.basin_depth(c) for c in row]
                   for row in effective[::500]])

for i, name in enumerate(sp.SECTOR_NAMES):
    ax5.plot(sim_time[::500], depths[:, i], label=name)
ax5.axhline(0.0, color='k', lw=1)
ax5.set_xlabel('Model time')
ax5.set_ylabel('Basin depth (potential barrier)')
ax5.set_title('Negative depth means the attractor is gone,\n'
              'not that the budget is bad')
ax5.legend(fontsize=8, ncol=2)


# --- 6. Cascade -----------------------------------------------------
# Same forcing, run with and without coupling. A basin that tips only
# in the coupled run was tipped by a neighbour, not by its own
# forcing.

for i, name in enumerate(sp.SECTOR_NAMES):
    style = '-' if attribution['cascade_only'][i] else '--'
    width = 2.5 if attribution['cascade_only'][i] else 1.2
    ax6.plot(sim_time[::100], traj[::100, i], style, lw=width, label=name)
ax6.axhline(0.0, color='k', lw=1, ls=':')
ax6.set_xlabel('Model time')
ax6.set_ylabel('Basin state (−1 cold, +1 reorganised)')
ax6.set_title('Solid = tipped only because a neighbour did')
ax6.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'structural_v4_output.png', dpi=140)


# --- Printed diagnostics --------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('Spatial')
print(f'lattice                     {ROWS} x {COLS}, periodic in longitude')
print(f'emergent percolation p_c    {p_c:.3f}  (analytic 0.500 for an '
      f'infinite square lattice)')
print(f'connectivity at year {years[-1]:<3}     {conn[-1]:.3f}')
print(f'circumpolar mean supply     {supply[-1].mean():.3f} of baseline')
print(f'sector range                {supply[-1].min():.2f} to '
      f'{supply[-1].max():.2f}')
print(f'redistribution index        {redist[-1]:.2f}')
print('  A scalar model reports a '
      f'{100 * (1 - supply[-1].mean()):.0f}% decline. Sectors moved '
      f'{100 * np.abs(supply[-1] - 1).max():.0f}% in opposite directions.')

rule('Demography')
A0 = pop.build_leslie_matrix(fecundity)
lx0, mx0 = pop.life_table(fecundity=fecundity)
print(f'raw literature fecundity    {pop.RAW_FECUNDITY:.2f} -> lambda '
      f'{pop.growth_rate(pop.build_leslie_matrix()):.4f}')
print(f'calibrated fecundity        {fecundity:.2f} -> lambda '
      f'{pop.growth_rate(A0):.4f}')
print(f'generation time             {pop.generation_time(lx0, mx0):.0f} yr')
print(f'damping ratio               {pop.damping_ratio(A0):.4f}')
print(f'memory (to 10% residual)    {pop.memory_years(A0):.0f} yr')
print(f'transient period            {pop.transient_period(A0):.0f} yr '
      f'(cohort echo, near generation time)')
print(f'lifespan, the v1 proxy      {P["baseline_lifespan_years"]} yr')
print(f'\n  Memory exceeds lifespan by '
      f'{100 * (memories[0] / P["baseline_lifespan_years"] - 1):.0f}%. '
      f'At +6 °C it falls to {memories[-1]:.0f} yr,')
print('  while population growth rate slightly RISES — faster turnover, '
      'same')
print('  lifetime output. Warming costs the system its memory before it '
      'costs')
print('  it viability. That is the flywheel, made quantitative.')

rule('Coupled basins')
print(f'critical forcing            {basins.CRITICAL_FORCING:.4f}')
print(f'{"sector":<16}{"forcing":>9}{"alone":>9}{"coupled":>9}'
      f'{"cascade":>9}')
for i, name in enumerate(sp.SECTOR_NAMES):
    iso, cpl = attribution['isolated_step'][i], attribution['coupled_step'][i]
    print(f'{name:<16}{FINAL_FORCING[i]:>9.3f}'
          f'{("hold" if iso < 0 else f"{iso * dt:.0f}"):>9}'
          f'{("hold" if cpl < 0 else f"{cpl * dt:.0f}"):>9}'
          f'{str(bool(attribution["cascade_only"][i])):>9}')

margin_now, margin_after = basins.susceptibility(FINAL_FORCING, D)
print(f'\n{"sector":<16}{"margin":>9}{"if nbrs tip":>13}')
for i, name in enumerate(sp.SECTOR_NAMES):
    print(f'{name:<16}{margin_now[i]:>+9.3f}{margin_after[i]:>+13.3f}')

cascaded = int(attribution['cascade_only'].sum())
print(f'\n  {cascaded} of {n} basins tipped only because a neighbour did.')
print('  Coupling strengths between real Antarctic tipping systems are '
      'not')
print('  established. The literature supports the sign, not the '
      'magnitude —')
print('  sweep `strength` in ring_coupling() rather than trusting 0.06.')

plt.show()
