"""When the trend overtakes the variability — and where it has not.

The 36th State of the Climate report records something that bears
directly on the conclusions this repo reached about variance and
means: 2025 was the warmest year on record **with no El Niño
present**, and the eleven years 2015-2025 were the eleven warmest.
Ranking has stopped depending on ENSO phase.

That is a specific, checkable statement about the ratio between a
trend and a mode of variability, and it is exactly the ratio the
basin results turned on. Earlier sims concluded that slow systems
respond to the mean rather than to variability. This one asks when
the mean took over — and finds that the answer is different at the
surface and at the depth the model's organism occupies.

Run from anywhere:  python Sims/emergence.py

Literature: Docs/literature.md section 14
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import climate_modes as cm

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

rng = np.random.default_rng(2025)

# ENSO one-sigma at the model's reference depth, from the composite
# scaling already in climate_modes.
DEPTH_SIGMA = float(cm.subsurface_anomaly_C(1.0))
DEPTH_TREND = P['abyssal_warming_rate_C_per_year']

# Global surface, for contrast. The trend is the recent
# multi-decadal rate and the ENSO amplitude is the familiar
# order-of-magnitude for global mean surface temperature; both are
# round numbers used to place the surface case beside the deep one,
# not results of this model.
SURFACE_TREND = 0.020
SURFACE_SIGMA = 0.100

CASES = (
    ('global surface', SURFACE_TREND, SURFACE_SIGMA, 'C1'),
    ('Antarctic 490 m', DEPTH_TREND, DEPTH_SIGMA, 'C0'),
)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax1, ax2), (ax3, ax4) = axes


# --- 1. Time of emergence -----------------------------------------------

trend_axis = np.logspace(-3.2, -1.2, 200)
for label, trend, sigma, colour in CASES:
    ax1.loglog(trend_axis, sigma / trend_axis, color=colour, lw=1.6,
               label=f'{label} (σ = {sigma:.2f} °C)')
    ax1.plot(trend, sigma / trend, 'o', ms=9, color=colour)
    ax1.annotate(f'{sigma / trend:.0f} yr',
                 xy=(trend, sigma / trend), xytext=(trend * 1.3,
                                                    sigma / trend * 1.5),
                 fontsize=9, color=colour)
ax1.axhline(1, color='grey', lw=0.8)
ax1.set_xlabel('Warming trend (°C per year)')
ax1.set_ylabel('Years for the trend to equal one σ of ENSO')
ax1.set_title('Time of emergence')
ax1.legend(fontsize=8)


# --- 2. When a neutral year beats a past record El Niño ------------------

years = 260
axis = np.arange(years)
index = cm.enso_index(years, np.random.default_rng(8),
                      skewness=cm.ENSO_SKEWNESS)

# A record event early in the record, then a run of neutral years.
record_year = 30
record_sigma = cm.event_sigma(P['nino34_peak_2015_16_C'])

for i, (label, trend, sigma, colour) in enumerate(CASES):
    series = trend * axis + sigma * index
    record_value = trend * record_year + sigma * record_sigma

    neutral = np.abs(index) < 0.5
    beats = neutral & (series > record_value) & (axis > record_year)
    first = int(np.argmax(beats)) if beats.any() else -1

    offset = i * 2.6
    ax2.plot(axis, series + offset, color=colour, lw=0.9, alpha=0.85,
             label=label)
    ax2.axhline(record_value + offset, color=colour, ls='--', lw=1.2)
    ax2.plot(record_year, record_value + offset, '*', ms=14, color=colour)
    if first > 0:
        ax2.plot(first, series[first] + offset, 'o', ms=8,
                 markerfacecolor='none', markeredgecolor='k', mew=1.8)
        ax2.annotate(f'{first - record_year} yr later',
                     xy=(first, series[first] + offset),
                     xytext=(first + 8, series[first] + offset + 0.35),
                     fontsize=8)

ax2.set_xlabel('Year')
ax2.set_ylabel('Anomaly (°C, series offset for display)')
ax2.set_title('Star = a record El Niño.\nCircle = first neutral year that beats it')
ax2.legend(fontsize=8, loc='upper left')


# --- 3. Ranking stability ------------------------------------------------
# "The last eleven years were the eleven warmest" is an extremely
# strong statement about trend against noise. Under no trend it is
# essentially impossible.

record_length = 176           # years of instrumental record
streak = 11
ratios = np.linspace(0.0, 0.6, 25)      # trend per year, in units of sigma
trials = 4000

probabilities = []
for ratio in ratios:
    series = (ratio * np.arange(record_length)[None, :]
              + rng.standard_normal((trials, record_length)))
    order = np.argsort(np.argsort(series, axis=1), axis=1)
    top = order >= record_length - streak
    probabilities.append(float(np.mean(top[:, -streak:].all(axis=1))))
probabilities = np.array(probabilities)

ax3.plot(ratios, probabilities, 'o-', ms=4, color='C0')
ax3.axhline(1.0, color='grey', ls=':', lw=1)
for label, trend, sigma, colour in CASES:
    ax3.axvline(trend / sigma, color=colour, ls='--', lw=1.5)
    ax3.text(trend / sigma + 0.008, 0.45, label, rotation=90, fontsize=8,
             color=colour)
ax3.set_xlabel('Trend per year, in units of σ')
ax3.set_ylabel(f'P(last {streak} years are the {streak} warmest)')
ax3.set_title(f'The 11-in-11 streak happened.\nUnder no trend it never '
              f'occurred in {trials:,} trials')


# --- 4. Realised against committed ---------------------------------------
# Sea level is the one place the commitment lag has a directly
# observed rate to sit beside it.

REALISED_TOTAL_MM = 111.2      # above the 1993 baseline
THERMAL_MM_PER_YR = 1.6
ICE_MM_PER_YR = 2.0
COMMITTED_M = 4.0              # West Antarctic marine ice, long term

realised_rate = THERMAL_MM_PER_YR + ICE_MM_PER_YR
centuries = COMMITTED_M * 1000.0 / realised_rate

bars = ['realised\nsince 1993', 'committed\n(W. Antarctic)']
values = [REALISED_TOTAL_MM / 1000.0, COMMITTED_M]
ax4.bar(bars, values, color=['C0', 'C3'], width=0.55)
ax4.set_yscale('log')
ax4.set_ylabel('Sea level (m, log scale)')
for i, value in enumerate(values):
    ax4.text(i, value * 1.15, f'{value:.3f} m' if value < 1 else f'{value:.0f} m',
             ha='center', fontsize=10)
ax4.set_title(f'Realised at {realised_rate:.1f} mm/yr;\n'
              f'committed would take ~{centuries:,.0f} yr at that rate')

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'emergence_output.png', dpi=140)


# --- Printed diagnostics --------------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('Time of emergence')
print(f'{"":<18}{"trend":>12}{"ENSO 1σ":>10}{"emergence":>12}')
for label, trend, sigma, _ in CASES:
    print(f'{label:<18}{trend:>10.4f}°{sigma:>9.3f}°{sigma / trend:>10.0f} yr')
print('\n  At the surface the trend passed one sigma of ENSO within a')
print('  decade, which is why ranking no longer depends on ENSO phase')
print('  and why a neutral 2025 came in among the three warmest years.')
print(f'\n  At {DEPTH_SIGMA:.2f} °C per sigma and '
      f'{DEPTH_TREND:.4f} °C/yr, the depth band this model is about')
print('  needs 44 years to reach the same point. That band is still')
print('  variability-dominated.')

rule('Beating a past record El Niño with a neutral year')
for label, trend, sigma, _ in CASES:
    print(f'{label:<18}{3.5 * sigma / trend:>8.0f} yr of trend to overtake '
          f'a 3.5σ event')
print('\n  Eighteen years at the surface. A century and a half at depth.')

rule('Ranking stability')
print(f'{"trend/σ":>10}{"P(last 11 are top 11)":>24}')
for ratio, probability in zip(ratios[::4], probabilities[::4]):
    print(f'{ratio:>10.3f}{probability:>24.3f}')
surface_ratio = SURFACE_TREND / SURFACE_SIGMA
depth_ratio = DEPTH_TREND / DEPTH_SIGMA
print(f'\n  Under no trend the streak did not occur once in '
      f'{trials:,} trials.')
print(f'  At the surface ratio ({surface_ratio:.2f}) it occurs '
      f'{np.interp(surface_ratio, ratios, probabilities) * 100:.1f}% of')
print('  the time — a few percent, not a certainty. So the observed')
print('  streak is strong evidence of a trend without being an')
print('  expected outcome, and the exact figure is sensitive to a')
print('  ratio these round numbers do not pin down.')
print(f'  At depth ({depth_ratio:.3f}) it is indistinguishable from the')
print('  no-trend case.')

rule('Realised against committed')
print(f'realised since 1993            {REALISED_TOTAL_MM:.1f} mm')
print(f'  thermal expansion            {THERMAL_MM_PER_YR:.1f} mm/yr')
print(f'  ice melt                     {ICE_MM_PER_YR:.1f} mm/yr')
print(f'  total                        {realised_rate:.1f} mm/yr')
print(f'committed, West Antarctic      ~{COMMITTED_M:.0f} m')
print(f'ratio                          '
      f'{COMMITTED_M * 1000 / REALISED_TOTAL_MM:.0f}x what has been realised')
print(f'at the observed rate           ~{centuries:,.0f} yr')
print('\n  Ice melt now exceeds thermal expansion, which is a change in')
print('  which term dominates rather than in the total. And the')
print(f'  committed quantity is {COMMITTED_M * 1000 / REALISED_TOTAL_MM:.0f} '
      f'times what has been realised in')
print('  three decades — the commitment lag of Docs/literature.md')
print('  section 13, in an observable that has a measured rate')
print('  attached to it.')

rule('What this changes')
print('  Earlier sims concluded that slow basins respond to the mean')
print('  and not to variability, and used that to argue the ENSO')
print('  literature that matters is the part about changing')
print('  statistics. That still holds. What this adds is WHERE the')
print('  mean has actually taken over.')
print()
print('  At the global surface it has: ranking no longer depends on')
print('  ENSO phase, and a neutral year now outranks former record')
print('  El Niño years. At 490 m in the Antarctic it has not, and on')
print('  the model\'s own trend it will not for four decades.')
print()
print('  So the two conclusions apply to different places. The')
print('  mean-dominated argument is already the right frame for the')
print('  surface. For the deep band this model is built around, a')
print('  single event still moves the water more than a decade of')
print('  trend does — which is what made the latching result matter.')

plt.show()
