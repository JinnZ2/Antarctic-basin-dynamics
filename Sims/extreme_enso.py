"""A record El Niño against the model's layers.

The 2026-27 event is forecast to peak near 3.6 °C in Niño 3.4,
roughly 1 °C above the previous 149-year record. The 1877-78 and
2015-16 events sit in a statistical dead heat at 2.73 and 2.75 °C,
well inside the uncertainty of nineteenth-century ship data, so the
ranking between those two means little — but a full degree above
both is a different kind of statement.

This asks four things.

1. Can the model's ENSO generator produce such an event at all?
   It cannot, and that is worth showing rather than quietly fixing.
2. How large is it at the depth the model's organism occupies?
3. Does a record event tip a basin? It overshoots the threshold by
   a wide margin while it lasts.
4. Does the slow integrator record it, and when does the record
   become readable?

Run from anywhere:  python Sims/extreme_enso.py

Literature: Docs/literature.md section 10
Caveats: Docs/structure.md
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

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

FORCING_PER_DEGREE = 0.19        # see enso_coupling.py; invented
BASELINE_DEPTH_M = 490.0
RECRUITMENT_LOSS_PER_SIGMA = 0.10

EVENT = '2026-27 (forecast)'
EVENT_SIGMA = cm.event_sigma(cm.HISTORICAL_EVENTS[EVENT])
EVENT_C = cm.subsurface_anomaly_C(EVENT_SIGMA)
EVENT_FORCING = EVENT_C * FORCING_PER_DEGREE

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes


# --- 1. The generator cannot produce it -------------------------------

sample = 400_000
gaussian = cm.enso_index(sample, np.random.default_rng(1), skewness=0.0)
skewed = cm.enso_index(sample, np.random.default_rng(1),
                       skewness=cm.ENSO_SKEWNESS)

bins = np.linspace(-5, 6, 160)
ax1.hist(gaussian, bins=bins, density=True, histtype='step', lw=1.4,
         label='symmetric (as shipped)')
ax1.hist(skewed, bins=bins, density=True, histtype='step', lw=1.4,
         label=f'skewed (a={cm.ENSO_SKEWNESS})')
for name, anomaly in cm.HISTORICAL_EVENTS.items():
    if 'plume' in name:
        continue
    sigma = cm.event_sigma(anomaly)
    ax1.axvline(sigma, color='C3', ls='--', lw=1.2)
    ax1.text(sigma + 0.06, 0.3, name, rotation=90, fontsize=7, color='C3')
ax1.set_yscale('log')
ax1.set_ylim(1e-6, 1)
ax1.set_xlabel('ENSO index (σ)')
ax1.set_ylabel('Density (log)')
ax1.set_title('Where the observed events sit in the modelled tail')
ax1.legend(fontsize=8)


# --- 2. Return periods -------------------------------------------------

sigma_axis = np.linspace(2.0, 5.0, 25)
curves = {}
for strength, label in ((0.0, 'symmetric'),
                        (cm.ENSO_SKEWNESS, f'skewed a={cm.ENSO_SKEWNESS}'),
                        (0.6, 'skewed a=0.6')):
    curves[label] = cm.return_period(sigma_axis, skewness=strength,
                                     n_years=1_000_000,
                                     rng=np.random.default_rng(4))

for label, periods in curves.items():
    finite = np.isfinite(periods)
    ax2.semilogy(sigma_axis[finite], periods[finite], lw=1.5, label=label)

# Two events near 3.5 sigma in a 149-year record. The Poisson
# interval on n=2 is enormous, which is the point.
ax2.axhspan(21, 620, color='grey', alpha=0.2)
ax2.text(2.05, 90, 'observed rate for ~3.5σ events,\n95% Poisson interval on n=2',
         fontsize=7)
ax2.axvline(EVENT_SIGMA, color='C3', ls='--', lw=1.4)
ax2.text(EVENT_SIGMA - 0.5, 2e5, '2026-27', fontsize=8, color='C3')
ax2.set_xlabel('Event size (σ)')
ax2.set_ylabel('Modelled return period (years)')
ax2.set_title('The symmetric generator never produced it\nin a million years')
ax2.legend(fontsize=8)


# --- 3. Size at depth ---------------------------------------------------

names, sigmas, degrees = [], [], []
for name, anomaly in cm.HISTORICAL_EVENTS.items():
    names.append(name.replace(' (', '\n('))
    sigmas.append(cm.event_sigma(anomaly))
    degrees.append(cm.subsurface_anomaly_C(cm.event_sigma(anomaly)))

position = np.arange(len(names))
colours = ['C0' if s < EVENT_SIGMA else 'C3' for s in sigmas]
ax3.bar(position, degrees, 0.6, color=colours)
ax3.axhline(P['warming_delta_C'], color='k', ls='--', lw=1.5)
ax3.text(-0.4, P['warming_delta_C'] * 1.02,
         f'warming_delta_C = {P["warming_delta_C"]} °C', fontsize=8)
ax3.set_xticks(position)
ax3.set_xticklabels(names, fontsize=7)
ax3.set_ylabel(f'Subsurface anomaly at {BASELINE_DEPTH_M:.0f} m (°C)')
ax3.set_title('One event against the total projected warming')


# --- 4. A record pulse against a basin ----------------------------------

relaxations = np.array([0.5, 1, 2, 3, 5, 10, 30, 100, 300])
dt = 0.01
pulse_tips = []
for relaxation in relaxations:
    time_per_year = 1.0 / relaxation
    steps = int(400 * time_per_year / dt)
    year_axis = np.arange(steps) * dt / time_per_year
    pulse = cm.event_pulse(int(year_axis[-1]) + 2, EVENT_SIGMA,
                           start_year=int(year_axis[-1] * 0.5))
    forcing = basins.CRITICAL_FORCING * 0.93 + (
        EVENT_FORCING / EVENT_SIGMA) * np.interp(
            year_axis, np.arange(len(pulse)), pulse)
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt)
    pulse_tips.append(basins.tipping_steps(traj)[0] >= 0)

pulse_tips = np.array(pulse_tips)
ax4.semilogx(relaxations, pulse_tips.astype(int), 'o-', ms=7, color='C3')
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['holds', 'tips'])
ax4.set_ylim(-0.3, 1.3)
ax4.set_xlabel('Basin relaxation time (years)')
ax4.set_title(f'One 2026-class pulse, worth {EVENT_FORCING:.2f} forcing\n'
              f'({EVENT_FORCING / basins.CRITICAL_FORCING:.0%} of critical) '
              f'onto a basin at 93%')


# --- 5. Recurrence is the control variable -------------------------------

intervals = np.array([200, 100, 50, 25, 18, 12, 6, 3])
relaxation = 30.0
time_per_year = 1.0 / relaxation
steps = int(1500 * time_per_year / dt)
year_axis = np.arange(steps) * dt / time_per_year
n_years = int(year_axis[-1]) + 2
margin = basins.CRITICAL_FORCING * 0.07

added, train_tips = [], []
for interval in intervals:
    train = np.zeros(n_years)
    for start in range(20, n_years, int(interval)):
        train += cm.event_pulse(n_years, EVENT_SIGMA, start_year=start)
    scaled = (EVENT_FORCING / EVENT_SIGMA) * np.interp(
        year_axis, np.arange(n_years), train)
    forcing = basins.CRITICAL_FORCING * 0.93 + scaled
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt)
    added.append(scaled.mean())
    train_tips.append(basins.tipping_steps(traj)[0] >= 0)

added = np.array(added)
train_tips = np.array(train_tips)

ax5.semilogx(intervals, added, 'o-', ms=6, color='C0',
             label='time-averaged forcing added')
ax5.axhline(margin, color='C3', ls='--', lw=1.5, label='remaining margin')
tipped = train_tips
ax5.scatter(intervals[tipped], added[tipped], s=140, facecolors='none',
            edgecolors='C3', lw=2, label='tips', zorder=5)
ax5.set_xlabel('Years between 2026-class events')
ax5.set_ylabel('Forcing')
ax5.set_title('The basin responds to the average, not the peak')
ax5.legend(fontsize=8)


# --- 6. The archive records it, slowly -----------------------------------

fecundity = pop.calibrate_fecundity()
horizon = 2000
event_year = 600

for label, delta_T, colour in (('baseline', 0.0, 'C0'), ('+6 °C', 6.0, 'C3')):
    max_age, maturity, bounds = pop.warmed_ages(
        delta_T, P['Q10_apex'],
        decoupling=P['longevity_maintenance_decoupling'])
    A = pop.build_leslie_matrix(fecundity, max_age, maturity, bounds)
    start = pop.stable_age_distribution(A) * 1000

    pulse = cm.event_pulse(horizon, EVENT_SIGMA, start_year=event_year)
    supply = np.clip(1.0 - RECRUITMENT_LOSS_PER_SIGMA * pulse, 0.0, None)

    forced = pop.project(A, start, horizon, supply=supply,
                         maturity_age=maturity)[:, maturity:].sum(axis=1)
    control = pop.project(A, start, horizon,
                          maturity_age=maturity)[:, maturity:].sum(axis=1)
    deviation = (forced / control - 1.0) * 100.0

    elapsed = np.arange(len(deviation)) - event_year
    ax6.plot(elapsed, deviation, lw=1.5, color=colour, label=label)
    trough = int(np.argmin(deviation))
    ax6.plot(elapsed[trough], deviation[trough], 'o', color=colour, ms=6)

ax6.axvline(0, color='grey', ls=':', lw=1.2)
ax6.axhline(0, color='grey', lw=0.8)
ax6.set_xlim(-100, 900)
ax6.set_xlabel('Years since the event')
ax6.set_ylabel('Adult abundance vs unforced control (%)')
ax6.set_title('A biological archive, written now\nand readable in a century')
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'extreme_enso_output.png', dpi=140)


# --- Printed diagnostics --------------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('Events')
print(f'{"event":<22}{"Niño3.4":>9}{"σ":>7}{"at 490 m":>10}')
for name, anomaly in cm.HISTORICAL_EVENTS.items():
    sigma = cm.event_sigma(anomaly)
    print(f'{name:<22}{anomaly:>8.2f}°{sigma:>7.2f}'
          f'{cm.subsurface_anomaly_C(sigma):>9.2f}°')
print(f'\nNiño3.4 σ used for the conversion: {cm.NINO34_SD_C} °C')
print(f'The 2026-27 event is worth {EVENT_C:.2f} °C at 490 m — '
      f'{EVENT_C / P["warming_delta_C"]:.0%} of the model default')
print(f'warming_delta_C of {P["warming_delta_C"]} °C, delivered in about '
      f'two years and then withdrawn.')

rule('Whether the generator can produce it')
for label, periods in curves.items():
    at_event = np.interp(EVENT_SIGMA, sigma_axis, np.nan_to_num(
        periods, posinf=np.nan))
    finite = np.isfinite(periods) & (sigma_axis >= EVENT_SIGMA)
    text = (f'{at_event:,.0f} yr' if finite.any() and np.isfinite(at_event)
            else 'never occurred in 1,000,000 years')
    print(f'  {label:<22}{text}')
print('\n  The shipped symmetric generator assigns an event that is')
print('  currently happening a probability indistinguishable from zero.')
print('  That is a defect, and the fix is only partial: matching the')
print('  observed rate of ~3.5σ events needs more skew than the Niño3.4')
print('  index actually shows. With two events in 149 years the tail is')
print('  not identifiable — the Poisson interval on the rate spans a')
print('  factor of thirty. No skewness value here is calibrated.')
print('\n  Use the model to ask what happens IF an event of this size')
print('  occurs. Do not use it to ask how often one will.')

rule('Does a record event tip a basin?')
print(f'pulse forcing                {EVENT_FORCING:.3f}')
print(f'critical forcing             {basins.CRITICAL_FORCING:.3f}')
print(f'basin starts at              {basins.CRITICAL_FORCING * 0.93:.3f} '
      f'(93% of critical)')
print(f'peak instantaneous forcing   '
      f'{basins.CRITICAL_FORCING * 0.93 + EVENT_FORCING:.3f} — '
      f'{(basins.CRITICAL_FORCING * 0.93 + EVENT_FORCING) / basins.CRITICAL_FORCING:.0%} '
      f'of critical')
print(f'\n{"relaxation":>12}{"outcome":>10}')
for relaxation, tips in zip(relaxations, pulse_tips):
    print(f'{relaxation:>11.1f}y{"TIPS" if tips else "holds":>10}')
print('\n  The pulse carries the instantaneous forcing to 169% of critical,')
print('  on a basin already sitting at 93% of it, and anything slower')
print('  than about three years ignores it entirely. Not because the')
print('  event is small — because it is brief, and a basin integrates')
print('  over its own relaxation time.')

rule('What does move a slow basin')
print(f'basin relaxation 30 yr, remaining margin {margin:.4f}')
print(f'\n{"interval":>10}{"mean added":>13}{"outcome":>10}')
for interval, mean_added, tips in zip(intervals, added, train_tips):
    print(f'{interval:>9.0f}y{mean_added:>13.4f}'
          f'{"TIPS" if tips else "holds":>10}')
print('\n  The crossover sits exactly where the time-averaged addition')
print('  passes the margin, and nowhere near where the peak does. The')
print('  control variable is the recurrence interval, not the')
print('  amplitude — which is why the ENSO literature that matters')
print('  here is the projection of MORE FREQUENT extreme Eastern')
print('  Pacific events, not stronger ones.')

rule('The archive')
print('A 46% recruitment failure from a 2026-class event produces:')
print(f'{"":<10}{"adult deficit":>15}{"peaks after":>13}')
for label, delta_T in (('baseline', 0.0), ('+3 °C', 3.0), ('+6 °C', 6.0)):
    max_age, maturity, bounds = pop.warmed_ages(
        delta_T, P['Q10_apex'],
        decoupling=P['longevity_maintenance_decoupling'])
    A = pop.build_leslie_matrix(fecundity, max_age, maturity, bounds)
    start = pop.stable_age_distribution(A) * 1000
    pulse = cm.event_pulse(horizon, EVENT_SIGMA, start_year=event_year)
    supply = np.clip(1.0 - RECRUITMENT_LOSS_PER_SIGMA * pulse, 0.0, None)
    forced = pop.project(A, start, horizon, supply=supply,
                         maturity_age=maturity)[:, maturity:].sum(axis=1)
    control = pop.project(A, start, horizon,
                          maturity_age=maturity)[:, maturity:].sum(axis=1)
    deviation = (forced / control - 1.0) * 100.0
    trough = int(np.argmin(deviation))
    print(f'{label:<10}{deviation[trough]:>14.2f}%'
          f'{trough - event_year:>12} yr')

print('\n  The strongest El Niño in the instrumental record removes')
print('  roughly one percent of the adult population — and the deficit')
print('  does not appear for a century and a half, because the cohort')
print('  it destroyed has to reach maturity before its absence can be')
print('  counted.')
print('\n  This is what "long-lived species act as biological archives"')
print('  means mechanically. The event is written into the age')
print('  structure immediately and becomes readable much later.')
print('  Warming makes the entry larger and sooner: the archive gets')
print('  more reactive, which is the same thing as saying it stops')
print('  being an archive.')

plt.show()
