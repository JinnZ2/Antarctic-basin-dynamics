"""ENSO coupled into the depth, spatial, demographic and basin layers.

Four questions, in order of how much they change how the model
should be read.

1. What does the ENSO signal look like at the depth the model's
   organism actually occupies?
2. Does the Antarctic Dipole reproduce the sector divergence that
   `spatial.py` previously asserted with invented trends?
3. Does compressing ecological memory let high-frequency variation
   propagate further into the food web? geometry.md claims this
   and nothing in the model oscillated, so it had never been
   tested.
4. Can interannual variability tip a basin that would have held
   under the mean forcing?

Run from anywhere:  python Sims/enso_coupling.py

Literature: Docs/literature.md section 10
Structure and caveats: Docs/structure.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import basins
import climate_modes as cm
import population as pop
import spatial as sp

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

rng = np.random.default_rng(2026)
ROWS = P['lattice_rows']
BASELINE_DEPTH_M = 490.0

# Mapping choices, both flagged because neither is measured.
#
# Basin model time is not years -- the double well is normalised.
# One year is taken as 0.05 model time units, which puts basin
# relaxation at roughly two decades. ENSO is then fast relative to
# the basin, which is the regime that matters for noise-induced
# escape.
BASIN_TIME_PER_YEAR = 0.05

# Degrees of subsurface warming per unit of basin forcing. Set so
# the model's default 2 C warming brings a basin close to its
# threshold, which is the configuration the rest of the repo
# already assumes.
FORCING_PER_DEGREE = 0.19

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes


# --- 1. The signal ---------------------------------------------------

years = 1200
index = cm.enso_index(years, rng)
peak = cm.spectral_peak(index)

span = np.arange(150)
ax1.plot(span, index[:150], lw=1.0, color='k')
ax1.axhline(0, color='grey', lw=0.8)
ax1.fill_between(span, 0, index[:150], where=index[:150] > 0,
                 color='C3', alpha=0.45, interpolate=True, label='El Niño')
ax1.fill_between(span, 0, index[:150], where=index[:150] < 0,
                 color='C0', alpha=0.45, interpolate=True, label='La Niña')
ax1.legend(fontsize=8, loc='upper right')
ax1.set_xlabel('Year')
ax1.set_ylabel('ENSO index (standardised)')
ax1.set_title(f'Quasi-periodic, not periodic — spectral peak {peak:.1f} yr')


# --- 2. Depth ---------------------------------------------------------
# The response reverses sign with depth. During El Nino a weaker
# Amundsen Sea Low weakens coastal easterlies, reducing on-shelf
# Ekman transport of cold surface water and admitting warm
# Circumpolar Deep Water. The surface goes the other way.

field = cm.temperature_anomaly(1.0, sp.SECTOR_NAMES, ROWS)
depths = cm.row_depths(ROWS)

mesh = ax2.pcolormesh(np.arange(len(sp.SECTOR_NAMES)), depths, field,
                      cmap='RdBu_r', vmin=-0.5, vmax=0.5, shading='auto')
fig.colorbar(mesh, ax=ax2, label='temperature anomaly (°C)')
ax2.axhline(cm.SUBSURFACE_DEPTH_M, color='k', ls='--', lw=1.5)
ax2.axhline(BASELINE_DEPTH_M, color='lime', ls='-', lw=2)
ax2.text(-0.4, cm.SUBSURFACE_DEPTH_M - 25, 'sign reversal, 150 m', fontsize=8)
ax2.text(-0.4, BASELINE_DEPTH_M - 25, 'model baseline, 490 m',
         fontsize=8, color='darkgreen')
ax2.invert_yaxis()
ax2.set_xticks(np.arange(len(sp.SECTOR_NAMES)))
ax2.set_xticklabels([n[:6] for n in sp.SECTOR_NAMES], rotation=35, fontsize=8)
ax2.set_ylabel('Depth (m)')
ax2.set_title('El Niño anomaly: surface and subsurface disagree')


# --- 3. The dipole ----------------------------------------------------
# Sector divergence with a mechanism behind it, rather than the
# invented monotone trends spatial.py carried.

el_nino = cm.habitat_anomaly(1.0, sp.SECTOR_NAMES)
la_nina = cm.habitat_anomaly(-1.0, sp.SECTOR_NAMES)
assumed = np.array([sp.SECTOR_SUPPLY_TREND[n] for n in sp.SECTOR_NAMES])
assumed = assumed / np.abs(assumed).max()

position = np.arange(len(sp.SECTOR_NAMES))
ax3.bar(position - 0.27, el_nino, 0.27, label='El Niño', color='C3')
ax3.bar(position, la_nina, 0.27, label='La Niña', color='C0')
ax3.bar(position + 0.27, assumed, 0.27, color='grey',
        label='assumed trend (normalised)')
ax3.axhline(0, color='k', lw=0.8)
ax3.set_xticks(position)
ax3.set_xticklabels([n[:6] for n in sp.SECTOR_NAMES], rotation=35, fontsize=8)
ax3.set_ylabel('Habitat anomaly (dimensionless)')
ax3.set_title('The seesaw the assumed trends were imitating')
ax3.legend(fontsize=8)


# --- 4. The population as a filter ------------------------------------
# Adults integrate every recruitment year from maturity to death, so
# they low-pass filter recruitment. The cutoff is set by that span,
# and warming shortens it.

white = rng.standard_normal(16000)
fecundity = pop.calibrate_fecundity()

gains = {}
for label, dT in (('baseline', 0.0), ('+6 °C', 6.0)):
    max_age, maturity, bounds = pop.warmed_ages(
        dT, P['Q10_apex'], decoupling=P['longevity_maintenance_decoupling'])
    A = pop.build_leslie_matrix(fecundity, max_age, maturity, bounds)
    gains[label] = pop.recruitment_transfer(A, white, maturity_age=maturity)

for label, (periods, gain) in gains.items():
    order = np.argsort(periods)
    ax4.loglog(periods[order], gain[order], lw=1.2, label=label)
ax4.axvspan(2, 7, color='C3', alpha=0.18)
ax4.text(2.6, ax4.get_ylim()[0] * 3, 'ENSO band', fontsize=8, color='C3')
ax4.set_xlabel('Period (years)')
ax4.set_ylabel('Gain, adults per unit recruitment forcing')
ax4.set_title('The slow integrator is a low-pass filter')
ax4.legend(fontsize=8)


# --- 5. Filtering degrades under warming -------------------------------

warming = np.linspace(0, 6, 13)
enso_gain, century_gain = [], []
for dT in warming:
    max_age, maturity, bounds = pop.warmed_ages(
        dT, P['Q10_apex'], decoupling=P['longevity_maintenance_decoupling'])
    A = pop.build_leslie_matrix(fecundity, max_age, maturity, bounds)
    periods, gain = pop.recruitment_transfer(A, white, maturity_age=maturity)
    enso_gain.append(pop.band_gain(periods, gain, 2, 7))
    century_gain.append(pop.band_gain(periods, gain, 80, 300))

enso_gain = np.array(enso_gain)
century_gain = np.array(century_gain)
relative = (enso_gain / enso_gain[0] - 1) * 100

ax5.plot(warming, relative, 'o-', ms=4, color='C3',
         label='ENSO-band gain')
ax5.plot(warming, (century_gain / century_gain[0] - 1) * 100, 's-', ms=4,
         color='C0', label='century-band gain')
ax5.axhline(0, color='grey', lw=0.8)
ax5.set_xlabel('Temperature increase (°C)')
ax5.set_ylabel('Change in gain from baseline (%)')
ax5.set_title('Memory compression lets more variance through')
ax5.legend(fontsize=8)


# --- 6. Can interannual variability tip a basin? -----------------------
# The answer turns on a ratio the model does not know: ENSO period
# against basin relaxation time. So it is swept rather than assumed.
#
# Two forcings, both riding a mean ramp that stops short of the
# threshold: zero-mean ENSO, and ENSO offset toward a persistent El
# Nino state. The second is the case that matters, because CMIP6
# projects more frequent extreme Eastern Pacific events -- a change
# in the tail and the mean, not in the variance.

members = 32
relaxation_years = np.array([0.3, 0.5, 1, 2, 3, 5, 10, 30, 100, 300])
noise_scale = cm.ENSO_SUBSURFACE_AMPLITUDE_C * FORCING_PER_DEGREE
dt = 0.01


def tipping_fraction(relaxation, offset):
    """Share of ENSO realisations that cross, given a basin timescale."""
    time_per_year = 1.0 / relaxation
    steps = int(600 * time_per_year / dt)
    year_axis = np.arange(steps) * dt / time_per_year

    mean_forcing = basins.CRITICAL_FORCING * 0.93 * np.clip(
        year_axis / (0.7 * year_axis[-1]), 0, 1)

    ensemble = np.empty((steps, members))
    for member in range(members):
        annual = cm.enso_index(int(year_axis[-1]) + 2,
                               np.random.default_rng(500 + member)) + offset
        ensemble[:, member] = mean_forcing + noise_scale * np.interp(
            year_axis, np.arange(len(annual)), annual)

    traj = basins.simulate(np.full(members, basins.COLD_STATE),
                           ensemble, None, dt)
    return float(np.mean(basins.tipping_steps(traj) >= 0)), mean_forcing[-1]


zero_mean = np.array([tipping_fraction(r, 0.0)[0] for r in relaxation_years])
persistent, peak_forcing = zip(*[tipping_fraction(r, 0.4)
                                 for r in relaxation_years])
persistent = np.array(persistent)
peak_forcing = peak_forcing[0]

ax6.semilogx(relaxation_years, zero_mean * 100, 'o-', ms=5, color='C0',
             label='zero-mean ENSO')
ax6.semilogx(relaxation_years, persistent * 100, 's-', ms=5, color='C3',
             label='persistent El Niño (+0.4σ)')
ax6.axvline(cm.ENSO_PERIOD_YEARS, color='grey', ls=':', lw=1.5)
ax6.text(cm.ENSO_PERIOD_YEARS * 1.15, 50, 'ENSO period', fontsize=8,
         color='grey', rotation=90)
ax6.set_xlabel('Basin relaxation time (years)')
ax6.set_ylabel('Realisations tipping (%)')
ax6.set_title('Variability tips fast basins only;\nslow basins respond to the mean')
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'enso_coupling_output.png', dpi=140)


# --- Printed diagnostics ------------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('Signal')
print(f'spectral peak                {peak:.1f} yr (2-7 yr band)')
print(f'subsurface amplitude         '
      f'±{cm.ENSO_SUBSURFACE_AMPLITUDE_C} °C, 150 m to bottom')
print(f'as a fraction of the default warming_delta_C of '
      f'{P["warming_delta_C"]}:  '
      f'{cm.ENSO_SUBSURFACE_AMPLITUDE_C / P["warming_delta_C"]:.0%}')
print('  A single El Niño delivers a quarter of the total projected')
print('  warming to the depth band the model organism occupies, then')
print('  takes it away again.')

rule('Depth')
row = int(np.argmin(np.abs(depths - BASELINE_DEPTH_M)))
print(f'{"depth":>8}  anomaly at Amundsen/Bellingshausen (°C)')
for depth, value in zip(depths, field[:, -1]):
    marker = '  <- model baseline' if abs(depth - depths[row]) < 1e-9 else ''
    print(f'{depth:>7.0f}m  {value:>+8.3f}{marker}')
print('\n  Surface and subsurface have opposite signs. A model that')
print('  carried one temperature per sector would have to pick one,')
print('  and either choice is wrong for the other half of the column.')

rule('Dipole')
print(f'{"sector":<16}{"El Niño":>9}{"La Niña":>9}{"assumed":>9}')
for i, name in enumerate(sp.SECTOR_NAMES):
    print(f'{name:<16}{el_nino[i]:>+9.2f}{la_nina[i]:>+9.2f}{assumed[i]:>+9.2f}')
correlation = float(np.corrcoef(el_nino, assumed)[0, 1])
print(f'\ncorrelation, El Niño pattern against assumed trends: {correlation:+.2f}')
print('  Read this as a consistency check, not a discovery. Both')
print('  patterns were written from the same reported regional')
print('  contrasts, so a high correlation is partly built in.')
print('  What it does show is that the assumed trends were a standing')
print('  El Niño: they encoded one phase of an oscillation as a')
print('  monotone trend. The dipole supplies the mechanism and, more')
print('  usefully, restores the sign changes the trends could not.')

rule('Ecological memory as a filter')
print(f'{"dT":>4}{"ENSO gain":>12}{"century gain":>14}{"ratio":>10}')
for dT, e, c in zip(warming[::4], enso_gain[::4], century_gain[::4]):
    print(f'{dT:>4.1f}{e:>12.5f}{c:>14.4f}{e / c:>10.5f}')
print(f'\nENSO-band gain rises {relative[-1]:.0f}% from baseline to +6 °C.')
print(f'Attenuation at baseline is {1 / enso_gain[0]:.0f}x; '
      f'at +6 °C it is {1 / enso_gain[-1]:.0f}x.')
print('\n  geometry.md claims that compressing ecological memory lets')
print('  high-frequency variation propagate further into the food web.')
print('  The model now contains an oscillation, so the claim can be')
print('  checked, and it holds — but with a caveat the qualitative')
print('  statement lacks. The filtering is overwhelming at both ends.')
print('  A 67% increase on a thousandfold attenuation is still a')
print('  thousandfold attenuation. The slow integrator does not stop')
print('  being slow; it stops being quite as slow.')

rule('Can variability tip a basin?')
print(f'critical forcing             {basins.CRITICAL_FORCING:.4f}')
print(f'mean forcing peak            {peak_forcing:.4f}')
print(f'margin under mean forcing    '
      f'{basins.CRITICAL_FORCING - peak_forcing:+.4f}')
print(f'ENSO excursion in forcing    ±{noise_scale:.4f} '
      f'({noise_scale / basins.CRITICAL_FORCING:.0%} of critical)')
print(f'\n{"relaxation":>11}{"zero-mean":>12}{"persistent":>12}')
for relaxation, a, b in zip(relaxation_years, zero_mean, persistent):
    print(f'{relaxation:>10.1f}y{a:>11.0%}{b:>12.0%}')

print('\n  The excursion is more than three times the margin, and a slow')
print('  basin still never crosses. This is a negative result and it is')
print('  the most useful thing in this run.')
print('\n  A basin integrates forcing over its own relaxation time. ENSO')
print('  averages to zero over any span longer than a decade, so for a')
print('  basin that responds on decadal or centennial timescales the')
print('  variance contributes nothing at all. Only basins as fast as')
print('  ENSO itself can be tipped by it.')
print('\n  Offsetting ENSO toward a persistent El Niño extends the')
print('  vulnerable range by about an order of magnitude — but that is a')
print('  shift in the MEAN doing the work, not the variability. Which')
print('  is why the parts of the ENSO literature that matter here are')
print('  the ones about changing statistics: more frequent extreme')
print('  Eastern Pacific events, and a teleconnection whose correlation')
print(f'  with the Antarctic Dipole fell from {cm.TELECONNECTION_EARLY} to '
      f'{cm.TELECONNECTION_LATE} after '
      f'{cm.TELECONNECTION_SHIFT_YEAR}.')
print('\n  This also corrects interpretation_notes.md. "Individual bad')
print('  years matter more as the margin narrows" holds for the fast')
print('  stochastic energy balance it was written about. It does not')
print('  hold for slow basins, which cannot see individual bad years.')

plt.show()
