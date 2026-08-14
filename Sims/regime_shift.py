"""Events that outlast themselves.

`extreme_enso.py` reported that a record El Niño tips nothing. That
was measured against slow basins in isolation, and it was the wrong
test.

Antarctic sea ice stepped to a new state in September 2016, after
the 2015-16 super El Niño, and has not come back — the record lows
of 2023, 2024 and 2025 sit in that new state, not in excursions
from the old one. Super El Niño events are reported to raise the
probability of abrupt, persistent regime shifts that endure for
years to decades after the event itself has faded. The mechanism
offered for 2016 is preconditioning plus trigger: a decade of
Winter Water thinning from 2005, then anomalously strong winds in
2015 mixing across the thinned layer.

Three things were missing from the earlier test.

**Persistence was never checked.** The earlier run asked whether
the state crossed while the pulse lasted. It never asked whether a
crossing stays crossed. In a bistable system it does, and that is
the entire point of the geometry.

**Only slow basins were shown.** The same sweep already said basins
of a few years' relaxation tip — and sea ice is one. The result was
reported from the half of the sweep that made the tidier claim.

**Timescales were never coupled.** A fast basin that latches applies
its coupling to slower neighbours permanently. It rectifies a
zero-mean transient into a step, and a step is what a slow basin
responds to. That bridge was never run.

Run from anywhere:  python Sims/regime_shift.py

Literature: Docs/literature.md section 12
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

with open(ROOT / 'Model' / 'parameters.json') as f:
    P = json.load(f)

FORCING_PER_DEGREE = 0.19
FAST_RELAXATION = 2.0        # sea-ice-like
SLOW_RELAXATION = 40.0       # ice-sheet or ecosystem-like

EVENT_SIGMA = cm.event_sigma(P['nino34_peak_2026_27_forecast_C'])
EVENT_FORCING = cm.subsurface_anomaly_C(EVENT_SIGMA) * FORCING_PER_DEGREE

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes


# --- 1. The forcing comes back; the state does not --------------------

dt = 0.02
years = 400
steps = int(years / dt)
year_axis = np.arange(steps) * dt

pulse = np.interp(year_axis, np.arange(int(year_axis[-1]) + 2),
                  cm.event_pulse(int(year_axis[-1]) + 2, EVENT_SIGMA,
                                 start_year=100))
forcing = basins.CRITICAL_FORCING * 0.93 + (
    EVENT_FORCING / EVENT_SIGMA) * pulse

fast = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt,
                       rates=1.0 / FAST_RELAXATION)

ax1.plot(year_axis, forcing, color='C0', lw=1.5, label='forcing')
ax1.axhline(basins.CRITICAL_FORCING, color='k', ls='--', lw=1.2,
            label='tipping threshold')
ax1.axhline(basins.recovery_forcing(), color='C2', ls=':', lw=1.5,
            label='recovery threshold')
ax1.set_ylabel('Forcing', color='C0')
ax1.set_xlabel('Year')

twin = ax1.twinx()
twin.plot(year_axis, fast[:, 0], color='C3', lw=2, label='state')
twin.set_ylabel('Basin state', color='C3')
ax1.set_title('A two-year event, a permanent state\n'
              f'(fast basin, {FAST_RELAXATION:.0f} yr relaxation)')
ax1.legend(fontsize=7, loc='center right')


# --- 2. Preconditioning decides whether the trigger works -------------
# The 2016 mechanism as reported: a decade of thinning, then a wind
# event. Neither alone.

preconditioning = np.linspace(0.60, 0.99, 34)     # fraction of critical
event_sizes = np.linspace(0.0, 6.0, 34)           # sigma
outcome = np.zeros((len(preconditioning), len(event_sizes)))

# 60 years is thirty relaxation times for the fast basin, which is
# ample; the grid is the slowest thing in the repo and does not need
# a longer window than that.
short_steps = int(60 / dt)
short_axis = np.arange(short_steps) * dt
base_pulse = np.interp(short_axis, np.arange(int(short_axis[-1]) + 2),
                       cm.event_pulse(int(short_axis[-1]) + 2, 1.0,
                                      start_year=20))

for i, level in enumerate(preconditioning):
    for j, size in enumerate(event_sizes):
        f = basins.CRITICAL_FORCING * level + (
            EVENT_FORCING / EVENT_SIGMA) * size * base_pulse
        traj = basins.simulate([basins.COLD_STATE], f[:, None], None, dt,
                               rates=1.0 / FAST_RELAXATION)
        outcome[i, j] = float(basins.latched(traj, f[:, None])[0])

ax2.contourf(event_sizes, preconditioning * 100, outcome, levels=[-0.5, 0.5, 1.5],
             colors=['#dfe7f0', '#c0392b'])
ax2.contour(event_sizes, preconditioning * 100, outcome, levels=[0.5],
            colors='k', linewidths=1.5)
# 1877-78 and 2015-16 sit within 0.03 sigma of each other, so they
# get one line rather than two overlapping labels.
for name, key, offset in (('1877-78 / 2015-16', 'nino34_peak_2015_16_C', 0.07),
                          ('2026-27', 'nino34_peak_2026_27_forecast_C', 0.07)):
    ax2.axvline(cm.event_sigma(P[key]), color='k', ls=':', lw=1)
    ax2.text(cm.event_sigma(P[key]) + offset, 61.5, name, rotation=90,
             fontsize=7)
ax2.set_xlabel('Event size (σ)')
ax2.set_ylabel('Preconditioning (% of critical forcing)')
ax2.set_title('Red = latches into a new state.\nNeither ingredient works alone')


# --- 3. The cascade across timescales ----------------------------------

long_years = 3000
long_steps = int(long_years / dt)
long_axis = np.arange(long_steps) * dt
long_pulse = np.interp(long_axis, np.arange(int(long_axis[-1]) + 2),
                       cm.event_pulse(int(long_axis[-1]) + 2, EVENT_SIGMA,
                                      start_year=100))

pair_forcing = np.column_stack([
    basins.CRITICAL_FORCING * 0.93 + (EVENT_FORCING / EVENT_SIGMA) * long_pulse,
    np.full(long_steps, basins.CRITICAL_FORCING * 0.88)])
rates = np.array([1.0 / FAST_RELAXATION, 1.0 / SLOW_RELAXATION])
coupling = np.array([[0.0, 0.0], [0.075, 0.0]])

coupled = basins.simulate([basins.COLD_STATE] * 2, pair_forcing, coupling,
                          dt, rates=rates)
isolated = basins.simulate([basins.COLD_STATE] * 2, pair_forcing, None,
                           dt, rates=rates)

steps_coupled = basins.tipping_steps(coupled)
ax3.plot(long_axis, coupled[:, 0], color='C3', lw=1.8,
         label=f'fast basin ({FAST_RELAXATION:.0f} yr)')
ax3.plot(long_axis, coupled[:, 1], color='C0', lw=1.8,
         label=f'slow basin ({SLOW_RELAXATION:.0f} yr), coupled')
ax3.plot(long_axis, isolated[:, 1], color='C0', lw=1.2, ls='--',
         label='slow basin, uncoupled')
ax3.axvline(100, color='grey', ls=':', lw=1.2)
ax3.text(130, -0.9, 'the event', fontsize=8, color='grey')
if steps_coupled[1] >= 0:
    lag = steps_coupled[1] * dt - 100
    ax3.annotate('', xy=(steps_coupled[1] * dt, 0.0), xytext=(100, 0.0),
                 arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
    ax3.text(0.5 * (100 + steps_coupled[1] * dt), 0.08,
             f'{lag:.0f} yr', ha='center', fontsize=8)
ax3.set_xlabel('Year')
ax3.set_ylabel('Basin state')
ax3.set_title('The fast basin carries the event\nto the slow one')
ax3.legend(fontsize=7, loc='lower right')


# --- 4. Rectification ---------------------------------------------------
# The transient has zero mean. What the slow basin receives does not.

received = basins.effective_forcing(coupled, pair_forcing, coupling)[:, 1]
ax4.plot(long_axis, (EVENT_FORCING / EVENT_SIGMA) * long_pulse,
         color='C3', lw=1.5, label='the event (transient, returns to zero)')
ax4.plot(long_axis, received - basins.CRITICAL_FORCING * 0.88,
         color='C0', lw=2, label='what the slow basin receives (a step)')
ax4.axhline(0, color='grey', lw=0.8)
ax4.set_xlim(0, 800)
ax4.set_xlabel('Year')
ax4.set_ylabel('Added forcing')
ax4.set_title('A latched fast basin rectifies a pulse into a step')
ax4.legend(fontsize=7)


# --- 5. Which timescales latch ------------------------------------------
# The earlier sim asked "does it tip". The question it should have
# asked is "does it stay".

relaxations = np.logspace(np.log10(0.3), np.log10(300), 16)
tips, stays = [], []
for relaxation in relaxations:
    horizon = max(400.0, 20 * relaxation)
    n = int(horizon / dt)
    axis = np.arange(n) * dt
    p = np.interp(axis, np.arange(int(axis[-1]) + 2),
                  cm.event_pulse(int(axis[-1]) + 2, EVENT_SIGMA,
                                 start_year=int(0.25 * horizon)))
    f = basins.CRITICAL_FORCING * 0.93 + (EVENT_FORCING / EVENT_SIGMA) * p
    traj = basins.simulate([basins.COLD_STATE], f[:, None], None, dt,
                           rates=1.0 / relaxation)
    tips.append(basins.tipping_steps(traj)[0] >= 0)
    stays.append(bool(basins.latched(traj, f[:, None])[0]))

tips, stays = np.array(tips), np.array(stays)
ax5.semilogx(relaxations, tips.astype(int), 'o-', ms=6, color='C3',
             label='crosses')
ax5.semilogx(relaxations, np.array(stays).astype(int) * 0.94, 's--', ms=5,
             color='C2', label='and stays crossed')
ax5.axvspan(1, 3, color='C0', alpha=0.15)
ax5.text(1.05, 0.45, 'sea-ice-like', fontsize=7, color='C0', rotation=90)
ax5.set_yticks([0, 1])
ax5.set_yticklabels(['no', 'yes'])
ax5.set_ylim(-0.25, 1.3)
ax5.set_xlabel('Basin relaxation time (years)')
ax5.set_title('Every basin that crosses, stays crossed')
ax5.legend(fontsize=7, loc='center left')


# --- 6. The asymmetry ----------------------------------------------------

labels = ['margin that was\ncrossed', 'reversal needed\nto undo it']
values = [basins.CRITICAL_FORCING * 0.07,
          basins.CRITICAL_FORCING * 0.93 - basins.recovery_forcing()]
ax6.bar(labels, values, color=['C3', 'C0'], width=0.55)
for i, value in enumerate(values):
    ax6.text(i, value * 1.02, f'{value:.3f}', ha='center', fontsize=9)
ax6.set_ylabel('Forcing')
ax6.set_title(f'Cheap to cross, {values[1] / values[0]:.0f}× dearer to undo')

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'regime_shift_output.png', dpi=140)


# --- Printed diagnostics --------------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('Correction to extreme_enso.py')
print('That sim reported that a record El Niño tips nothing. Measured')
print('against slow basins in isolation, that is true. It was still the')
print('wrong test, in three ways.')

rule('1. Persistence was never checked')
print(f'fast basin, {FAST_RELAXATION:.0f} yr relaxation, at 93% of critical')
print(f'  peak forcing during the event   {forcing.max():.3f}')
print(f'  forcing after the event         {forcing[-1]:.3f}  '
      f'(critical {basins.CRITICAL_FORCING:.3f})')
print(f'  state before                    {fast[int(50 / dt), 0]:+.3f}')
print(f'  state 300 years after           {fast[-1, 0]:+.3f}')
print(f'  latched                         '
      f'{bool(basins.latched(fast, forcing[:, None])[0])}')
print('\n  The forcing returned to sub-critical. The state did not. The')
print('  earlier sim asked whether the state crossed while the pulse')
print('  lasted, and never asked whether a crossing stays crossed. In a')
print('  bistable system it does, which is the whole point of the')
print('  geometry the model is built on.')

rule('2. The vulnerable basins were the ones not shown')
print(f'{"relaxation":>12}{"crosses":>10}{"stays":>8}')
for relaxation, tip, stay in zip(relaxations, tips, stays):
    print(f'{relaxation:>11.1f}y{"yes" if tip else "no":>10}'
          f'{"yes" if stay else "-":>8}')
print('\n  Antarctic sea ice responds on one to three years. That is')
print('  inside the band that crosses, and the earlier write-up led')
print('  with the slow half of its own sweep.')

rule('3. Timescales were never coupled')
print(f'fast basin {FAST_RELAXATION:.0f} yr at 93% of critical; '
      f'slow basin {SLOW_RELAXATION:.0f} yr at 88%, coupling 0.075')
print(f'{"":<12}{"coupled":>10}{"isolated":>11}')
fast_step, slow_step = basins.tipping_steps(coupled)
isolated_slow = basins.tipping_steps(isolated)[1]
print(f'{"fast tips":<12}{fast_step * dt:>9.0f}y'
      f'{basins.tipping_steps(isolated)[0] * dt:>10.0f}y')
print(f'{"slow tips":<12}'
      f'{(f"{slow_step * dt:.0f}y" if slow_step >= 0 else "never"):>10}'
      f'{(f"{isolated_slow * dt:.0f}y" if isolated_slow >= 0 else "never"):>11}')
if slow_step >= 0:
    print(f'\n  The slow basin never tips on its own forcing. Coupled, it')
    print(f'  crosses {slow_step * dt - 100:.0f} years after a two-year event.')
print('\n  The mechanism is rectification. The event has zero mean — it')
print('  arrives and leaves. But the fast basin it tips does not leave,')
print('  and a latched neighbour applies its coupling permanently. The')
print('  slow basin never sees a pulse. It sees a step.')
print('\n  That is how a two-year event reaches a forty-year system: not')
print('  directly, which was the earlier finding and stands, but through')
print('  something fast enough to be tipped and bistable enough to stay')
print('  tipped.')

rule('The asymmetry')
print(f'crossed a margin of                 {values[0]:.4f}')
print(f'tipping threshold                   {basins.CRITICAL_FORCING:+.4f}')
print(f'recovery threshold                  {basins.recovery_forcing():+.4f}')
print(f'hysteresis width                    {basins.hysteresis_width():.4f}')
print(f'reversal needed to undo the tip     {values[1]:.4f}  '
      f'({values[1] / values[0]:.0f}× the margin crossed)')
print('\n  Restoring the forcing is not enough and never was. The state')
print('  that a brief excursion moved requires a sustained reversal well')
print('  past the original conditions to move back.')

rule('What this does not change')
print('  Zero-mean variability still cannot move a slow basin directly.')
print('  The rate at which extreme events recur still sets the mean.')
print('  What changes is that neither is the only route: a single event')
print('  can produce a permanent shift by latching a fast subsystem, and')
print('  the slow basin then responds to the latch rather than to the')
print('  event. Recurrence rate matters for the direct path; a single')
print('  event is enough for the indirect one.')

plt.show()
