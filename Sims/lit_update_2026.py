"""Old vs new parameterisation after the August 2026 literature review.

Each panel shows the model's prior behaviour against the
behaviour implied by recent empirical work. The point is the
gap between the pairs, not either curve on its own.

Run from anywhere:  python Sims/lit_update_2026.py

Literature and caveats: Docs/literature.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

from core import (
    metabolic_multiplier,
    adjusted_lifespan,
    maintenance_adjusted_lifespan,
    trophic_energy_interception,
    dynamic_transfer_efficiency,
    accelerating_temperature,
    percolation_connectivity,
    oxygen_availability,
    metabolic_index,
)

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

warming_range = np.linspace(0, 6, 200)
years = np.linspace(0, 60, 200)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
(ax1, ax2), (ax3, ax4) = axes


# --- Forcing shape -------------------------------------------------
# Observed abyssal warming accelerated ~3x between the long-record
# trend and the 2017/18-2023/24 trend (Johnson et al., GRL 2024).

linear = accelerating_temperature(
    P['baseline_temp_C'], years,
    P['abyssal_warming_rate_C_per_year'], 0.0)
quadratic = accelerating_temperature(
    P['baseline_temp_C'], years,
    P['abyssal_warming_rate_C_per_year'],
    P['forcing_acceleration_C_per_year2'])

ax1.plot(years, linear, label='linear ramp (prior)')
ax1.plot(years, quadratic, label='accelerating (2026 review)')
ax1.set_xlabel('Years')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Forcing shape')
ax1.legend()


# --- Apex energy supply --------------------------------------------
# Transfer efficiency enters once per trophic level, so a given
# per-step reduction compounds across the chain.

micro_mult = metabolic_multiplier(P['Q10_microbial'], warming_range)

fixed_tte = np.full_like(warming_range, P['trophic_transfer_efficiency'])
dyn_tte = dynamic_transfer_efficiency(
    P['trophic_transfer_efficiency'], warming_range,
    P['tte_warming_sensitivity_per_C'], P['tte_floor'])

supply_fixed = [trophic_energy_interception(1.0, m, e, 3)
                for m, e in zip(micro_mult, fixed_tte)]
supply_dyn = [trophic_energy_interception(1.0, m, e, 3)
              for m, e in zip(micro_mult, dyn_tte)]

ax2.plot(warming_range, supply_fixed, label='fixed efficiency (prior)')
ax2.plot(warming_range, supply_dyn, label='warming-dependent (2026 review)')
ax2.set_yscale('log')
ax2.set_xlabel('Temperature increase (°C)')
ax2.set_ylabel('Apex energy supply (normalised, log scale)')
ax2.set_title('Apex supply, 3 trophic levels')
ax2.legend()


# --- Lifespan compression ------------------------------------------
# Genomic and cardiac evidence argues longevity involves active
# encoded maintenance, so compression should be damped rather than
# proportional. The decoupling coefficient is unconstrained.

apex_mult = metabolic_multiplier(P['Q10_apex'], warming_range)
strict = adjusted_lifespan(P['baseline_lifespan_years'], apex_mult)
damped = maintenance_adjusted_lifespan(
    P['baseline_lifespan_years'], apex_mult,
    P['longevity_maintenance_decoupling'])

ax3.plot(warming_range, strict, label='rate-of-living (prior)')
ax3.plot(warming_range, damped,
         label=f"damped, decoupling={P['longevity_maintenance_decoupling']}")
ax3.axhline(y=200, color='red', linestyle='--',
            label='generation compression threshold')
ax3.set_xlabel('Temperature increase (°C)')
ax3.set_ylabel('Theoretical lifespan (years)')
ax3.set_title('Lifespan compression (decoupling is a guess — sweep it)')
ax3.legend()


# --- Connectivity and oxygen ---------------------------------------
# Sea ice held a range for four decades, then stepped to a new state.
# Exponential decay cannot produce that at any parameter value.

exp_decay = P['patch_autocorrelation'] * np.exp(
    -P['fragmentation_rate'] * years)
sigmoid = percolation_connectivity(
    years, P['connectivity_threshold_time_years'],
    P['connectivity_transition_steepness'], P['patch_autocorrelation'])

ax4.plot(years, exp_decay, label='exponential decay (prior)')
ax4.plot(years, sigmoid, label='percolation transition (2026 review)')
ax4.axvline(x=P['connectivity_threshold_time_years'], color='grey',
            linestyle=':', label='threshold time')
ax4.set_xlabel('Years')
ax4.set_ylabel('Effective connectivity')
ax4.set_title('Connectivity collapse')
ax4.legend()

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'lit_update_2026_output.png', dpi=150)


# --- Oxygen squeeze, printed ----------------------------------------
# Warming raises demand and lowers supply at the same time. The
# penalty scales with body mass, so it falls hardest on the largest
# slow integrators.
#
# These are normalised values, not absolute aerobic scope. A 1 kg
# animal at dT=0 returns exactly 1.0 by construction. Read the
# relative decline down each column and the ordering across a row.
# Do not read the absolute level as a viability threshold.

print('Metabolic index by body mass (normalised; 1 kg at dT=0 is 1.0)')
print(f"{'dT (°C)':>8} {'O2':>7} {'1 kg':>8} {'100 kg':>8} {'700 kg':>8}")
for dT in [0, 1, 2, 4, 6]:
    o2 = oxygen_availability(
        dT, P['oxygen_baseline_saturation'],
        P['deep_oxygen_committed_loss_fraction'])
    mult = metabolic_multiplier(P['Q10_apex'], dT)
    row = [metabolic_index(o2, mult, mass, P['oxygen_mass_sensitivity'])
           for mass in (1, 100, 700)]
    print(f'{dT:>8} {float(o2):>7.3f} '
          f'{row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f}')

print('\nThe mass ordering is empirical. The exponent is not.')
print('See Docs/literature.md section 4.')

plt.show()
