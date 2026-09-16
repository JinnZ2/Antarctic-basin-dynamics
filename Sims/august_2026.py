"""The record arrives with the El Niño — §14 against its next data point.

`Sims/emergence.py` took the 2025 State of the Climate finding — a
top-three year with no El Niño — and placed the global surface at a
trend/σ ratio of 0.20, where the mean has taken over from ENSO. It
used two round numbers for the surface (0.020 °C/yr, 0.10 °C per σ)
and said so.

August 2026 is the other half of the superposition. ERA5 puts the
month at 1.65 °C above 1850-1900, the joint warmest month ever
measured, with a strong El Niño underneath it: weekly Niño 3.4 at
+2.7 °C in mid-August, CPC giving >90% odds of a very strong event.
The 12-month running mean is 1.48 °C — down from 1.64 °C two years
earlier, through the La Niña.

So the same surface has now been observed with ENSO off and with
ENSO on. This sim asks four things of that pair.

1. Where does the event sit *now* against the model's event ladder,
   and what is it worth at 490 m at this stage?
2. Do the two round surface numbers survive contact with the
   measured excursion?
3. Is "neutral year in the top three, then an El Niño record" the
   ordering a trend-plus-mode superposition produces at trend/σ =
   0.20 — or is the August record evidence that ENSO is back in
   charge?
4. What state is the fast basin in when the trigger arrives?

Run from anywhere:  python Sims/august_2026.py

Literature: Docs/literature.md section 15
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

# --- the surface case, exactly as emergence.py stated it -----------------
SURFACE_TREND = 0.020          # °C/yr, round number
SURFACE_SIGMA = 0.100          # °C per σ of ENSO, round number
SURFACE_RATIO = SURFACE_TREND / SURFACE_SIGMA

# --- August 2026, as observed -------------------------------------------
MONTH_C = P['global_surface_anomaly_aug_2026_preindustrial_C']     # 1.65
MEAN_12MO_C = P['global_surface_12_month_mean_sep2025_aug2026_C']  # 1.48
MEAN_12MO_PRIOR_C = P['global_surface_12_month_mean_sep2023_aug2024_C']  # 1.64
NINO34_NOW_C = P['nino34_aug_2026_observed_C']                     # 2.7
NINO34_FORECAST_C = P['nino34_peak_2026_27_forecast_C']            # 3.6

WARMING_DELTA_C = P['warming_delta_C']

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
(ax1, ax2), (ax3, ax4) = axes


# --- 1. The event ladder, with the observed rung added -------------------

ladder = {
    '1877-78': P['nino34_peak_1877_78_C'],
    '2015-16': P['nino34_peak_2015_16_C'],
    'Aug 2026\n(observed)': NINO34_NOW_C,
    '2026-27\n(forecast)': NINO34_FORECAST_C,
}
names = list(ladder)
sigmas = cm.event_sigma(np.array(list(ladder.values())))
at_depth = cm.subsurface_anomaly_C(sigmas)

colours = ['C7', 'C7', 'C3', 'C1']
ax1.bar(names, at_depth, color=colours, width=0.6)
ax1.axhline(WARMING_DELTA_C, color='k', ls='--', lw=1)
ax1.text(-0.4, WARMING_DELTA_C * 1.02,
         f'default warming_delta_C = {WARMING_DELTA_C:.1f} °C', fontsize=8)
for i, (s, d) in enumerate(zip(sigmas, at_depth)):
    ax1.text(i, d + 0.03, f'{s:.2f}σ\n{d:.2f} °C', ha='center', fontsize=8)
ax1.set_ylabel('Subsurface anomaly at 490 m (°C, linear scaling)')
ax1.set_title('The event as it stands in August,\nagainst the two records and the forecast')


# --- 2. The round numbers against the measured excursion -----------------

excess = MONTH_C - MEAN_12MO_C                       # 0.17
event_sigma_now = float(cm.event_sigma(NINO34_NOW_C))
realised_per_sigma = excess / event_sigma_now        # °C per σ, lower bound

swing_observed = MEAN_12MO_PRIOR_C - MEAN_12MO_C     # 0.16 fall
swing_trend = SURFACE_TREND * 2.0                    # what trend added
swing_enso = swing_observed + swing_trend            # ENSO must have removed
swing_in_sigma = swing_enso / SURFACE_SIGMA          # in units of the round σ

labels = ['monthly excess\nover 12-mo mean\n(per σ, realised)',
          'round number\nSURFACE_SIGMA']
ax2.bar(labels, [realised_per_sigma, SURFACE_SIGMA], color=['C3', 'C0'],
        width=0.5)
for i, v in enumerate([realised_per_sigma, SURFACE_SIGMA]):
    ax2.text(i, v + 0.003, f'{v:.3f} °C/σ', ha='center', fontsize=9)
ax2.set_ylabel('°C per σ of Niño 3.4')
ax2.set_title(f'Realised so far: {realised_per_sigma:.3f} °C/σ (lagged, still rising)\n'
              f'12-month mean swung {swing_enso:.2f} °C ≈ {swing_in_sigma:.1f}σ '
              f'of the round number')


# --- 3. Does superposition produce this ordering? -----------------------
# At trend/σ = 0.20 over a 176-year record: given the final year's
# ENSO state, where does it rank? Neutral (|i| < 0.5) years should
# sit near the top; ≥3σ El Niño years should BE the record. If both
# hold, the August record is the same statement as the neutral 2025
# ranking, not a reversal of it.

rng = np.random.default_rng(2026)
record_length = 176
trials = 40_000

noise = rng.standard_normal((trials, record_length))
series = SURFACE_RATIO * np.arange(record_length)[None, :] + noise
final_rank = (series[:, :-1] > series[:, -1:]).sum(axis=1) + 1   # 1 = record
final_index = noise[:, -1]

neutral = np.abs(final_index) < 0.5
strong = final_index >= 3.0
moderate = (final_index >= 1.0) & (final_index < 2.0)

p_neutral_top3 = float(np.mean(final_rank[neutral] <= 3))
p_neutral_record = float(np.mean(final_rank[neutral] == 1))
p_strong_record = float(np.mean(final_rank[strong] == 1)) if strong.any() else np.nan
p_moderate_record = float(np.mean(final_rank[moderate] == 1))

# and the same under NO trend, for contrast
series0 = noise
rank0 = (series0[:, :-1] > series0[:, -1:]).sum(axis=1) + 1
p0_neutral_top3 = float(np.mean(rank0[neutral] <= 3))
p0_strong_record = float(np.mean(rank0[strong] == 1)) if strong.any() else np.nan

bins = np.arange(1, 12)
for mask, label, colour in ((neutral, 'neutral year (|i| < 0.5)', 'C0'),
                            (moderate, 'moderate El Niño (1-2σ)', 'C2'),
                            (strong, 'strong El Niño (≥ 3σ)', 'C3')):
    hist, _ = np.histogram(np.clip(final_rank[mask], 1, 11), bins=np.arange(1, 13))
    ax3.step(bins, hist / mask.sum(), where='mid', lw=1.6, color=colour,
             label=label)
ax3.set_xlabel('Rank of the final year in a 176-year record (1 = warmest)')
ax3.set_ylabel('Probability')
ax3.set_title(f'trend/σ = {SURFACE_RATIO:.2f}: neutral years land top-3 with '
              f'P = {p_neutral_top3:.2f},\nstrong El Niño years are THE record with '
              f'P = {p_strong_record:.2f}')
ax3.legend(fontsize=8)


# --- 4. The fast basin when the trigger arrives --------------------------
# §12: Antarctic sea ice latched into a low state after 2015-16 and
# the record lows of 2023-25 sit inside it. August 2026 is the 3rd
# lowest August as the next super El Niño arrives — the trigger this
# time lands on a state that is already the latched one.

ant = P['antarctic_sea_ice_aug_2026_Mkm2']
ant_rank = P['antarctic_sea_ice_aug_2026_rank_lowest']
arc = P['arctic_sea_ice_aug_2026_Mkm2']
arc_rank = P['arctic_sea_ice_aug_2026_rank_lowest']

ax4.barh(['Antarctic\n(austral winter)', 'Arctic\n(boreal summer)'],
         [ant, arc], color=['C0', 'C1'], height=0.5)
ax4.text(ant + 0.2, 0, f'{ant:.2f} Mkm²  —  {ant_rank}rd lowest August',
         va='center', fontsize=9)
ax4.text(arc + 0.2, 1, f'{arc:.2f} Mkm²  —  {arc_rank}th lowest August',
         va='center', fontsize=9)
ax4.set_xlim(0, 22)
ax4.set_xlabel('August 2026 monthly mean extent (million km²)')
ax4.set_title('Fast-basin state at trigger arrival\n'
              '(Antarctic: inside the post-2016 latched state)')

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'august_2026_output.png', dpi=140)


# --- Printed diagnostics --------------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('1. The event as it stands')
print(f'{"":<22}{"Niño 3.4":>10}{"σ":>8}{"at 490 m":>11}{"of Δ":>8}')
for name, s, d in zip(names, sigmas, at_depth):
    flat = name.replace('\n', ' ')
    print(f'{flat:<22}{ladder[name]:>9.2f}°{s:>8.2f}{d:>10.2f}°'
          f'{d / WARMING_DELTA_C:>8.0%}')
print(f'\n  In August the event already sits within 0.05 °C of both previous')
print(f'  record peaks, with the peak months ahead of it. Scaled linearly,')
print(f'  that is {at_depth[2]:.2f} °C at 490 m now against {at_depth[3]:.2f} °C at the')
print(f'  forecast peak — {at_depth[2] / WARMING_DELTA_C:.0%} of the default warming step,')
print(f'  delivered inside one season. Upper bound; the response saturates.')

rule('2. The round numbers against the measured excursion')
print(f'  month                 {MONTH_C:.2f} °C above 1850-1900')
print(f'  12-month mean         {MEAN_12MO_C:.2f} °C')
print(f'  excess                {excess:.2f} °C at Niño 3.4 = +{NINO34_NOW_C:.1f} '
      f'({event_sigma_now:.2f}σ)')
print(f'  realised per σ        {realised_per_sigma:.3f} °C/σ   '
      f'(round number: {SURFACE_SIGMA:.3f})')
print(f'\n  12-month mean, Sep 2023-Aug 2024   {MEAN_12MO_PRIOR_C:.2f} °C')
print(f'  12-month mean, Sep 2025-Aug 2026   {MEAN_12MO_C:.2f} °C')
print(f'  fell {swing_observed:.2f} while trend added {swing_trend:.2f}  '
      f'→ ENSO swing ≈ {swing_enso:.2f} °C ≈ {swing_in_sigma:.1f}σ')
print('\n  The global response lags Niño 3.4 by about a season and the')
print('  event is still strengthening, so the per-σ figure is a floor.')
print(f'  The round number is {SURFACE_SIGMA / realised_per_sigma:.1f}× the floor, and the')
print('  two-year swing of the 12-month mean is two of its σ. Both round')
print('  numbers survive at the factor-of-two level. Neither is refined')
print('  here: one event is not a fit.')

rule('3. Is this the ordering superposition produces?')
print(f'{"final-year ENSO state":<28}{"P(top 3)":>10}{"P(record)":>11}'
      f'{"P(top 3), no trend":>20}')
print(f'{"neutral (|i| < 0.5)":<28}{p_neutral_top3:>10.2f}{p_neutral_record:>11.2f}'
      f'{p0_neutral_top3:>20.3f}')
print(f'{"moderate El Niño (1-2σ)":<28}{"":>10}{p_moderate_record:>11.2f}')
print(f'{"strong El Niño (≥ 3σ)":<28}{"":>10}{p_strong_record:>11.2f}'
      f'{p0_strong_record:>20.2f}')
print(f'\n  At trend/σ = {SURFACE_RATIO:.2f} a neutral year is top-three with')
print(f'  P = {p_neutral_top3:.2f} and a ≥3σ El Niño year is the record with')
print(f'  P = {p_strong_record:.2f}. Under no trend the neutral year is top-three')
print(f'  with P = {p0_neutral_top3:.3f} — while the ≥3σ year is still the record')
print(f'  with P = {p0_strong_record:.2f}. So the El Niño record carries almost no')
print('  information about the trend; the neutral top-three carries')
print('  nearly all of it. "Neutral 2025 in the top three, then an El')
print('  Niño record in 2026" is not ENSO taking back control — it is')
print('  the one ordering a trend-plus-mode superposition produces at')
print('  this ratio. §14 predicted the August record; it did not need')
print('  to be revised by it.')

rule('4. The fast basin when the trigger arrives')
print(f'  Antarctic August extent   {ant:.2f} Mkm²   {ant_rank}rd lowest')
print(f'  Arctic August extent      {arc:.2f} Mkm²   {arc_rank}th lowest')
print('\n  §12: the 2015-16 event tipped Antarctic sea ice into a low state')
print('  that 2023, 2024 and 2025 never left. The next super El Niño is')
print('  arriving onto that state, not onto the pre-2016 one. The')
print('  preconditioning-plus-trigger mechanism has its trigger; whether')
print('  a further step exists below the latched state is not something')
print('  this model can say — it has one threshold per basin.')

rule('What this changes')
print('  Nothing in the mechanism. Two observations are recorded against')
print('  it: the surface round numbers hold at the factor-of-two level,')
print('  and the neutral-then-El-Niño ordering is what the superposition')
print('  predicts rather than a counter-example to it. The one number')
print('  that moved is the event itself — from a forecast to an')
print('  observation in progress, 3.48σ and rising.')

plt.show()
