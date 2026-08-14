"""Committed is not the same as collapsed.

"The ice sheets are near collapse" packs two claims together that
the model keeps apart, and they have opposite implications.

The reported picture: conditions that initiate West Antarctic
collapse can be reached in decades and may already have been
reached — around 40% of West Antarctic ice may already be
committed to long-term loss at today's warming — while the
resulting sea-level rise unfolds over centuries to millennia.
Crossing a threshold does not mean collapsing.

So "any day now" is wrong about ice sheets, and it is wrong in a
way that is not reassuring. The transition is slow. The
*commitment* to it is not, and the window in which it can be
called off closes at commitment, long before anything looks
different.

There is also a conflation worth naming. Sea ice and ice sheets
are not the same object. Sea ice is frozen ocean, metres thick,
and can visibly reorganise in a season — 2016 did. An ice sheet
is kilometres of land ice with a response time three orders of
magnitude longer. Headlines move between them freely. The model
does not, because they are different basins with different rate
constants, and that difference is the whole story.

Run from anywhere:  python Sims/commitment.py

Literature: Docs/literature.md section 13
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import basins
import climate_modes as cm

# Relaxation times, in years. Orders of magnitude, not estimates.
# Sea ice from its observed one-to-three-year adjustment; the ice
# sheet from the centuries-to-millennia language in the literature;
# the shelf placed between them.
LADDER = (
    ('sea ice', 2.0, 'C3'),
    ('ice shelf', 30.0, 'C1'),
    ('ice sheet', 1000.0, 'C0'),
)

dt = 0.1
YEARS = 6000
steps = int(YEARS / dt)
year_axis = np.arange(steps) * dt

# One forcing ramp, crossing critical at a known moment, applied to
# every rung of the ladder. Any difference between them is the rate
# constant and nothing else.
CROSSING_YEAR = 500.0
ramp = basins.CRITICAL_FORCING * (0.80 + 0.20 * year_axis / CROSSING_YEAR)
crossing_index = int(np.argmax(ramp >= basins.CRITICAL_FORCING))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes


# --- 1. The ladder ------------------------------------------------------

rates = np.array([1.0 / relaxation for _, relaxation, _ in LADDER])
forcing = np.tile(ramp[:, None], (1, len(LADDER)))
ladder = basins.simulate([basins.COLD_STATE] * len(LADDER), forcing, None,
                         dt, rates=rates)
lags = basins.commitment_lag(ladder, forcing, dt)

for i, (name, relaxation, colour) in enumerate(LADDER):
    ax1.plot(year_axis, ladder[:, i], color=colour, lw=1.8,
             label=f'{name} ({relaxation:.0f} yr)')
ax1.axvline(year_axis[crossing_index], color='k', ls='--', lw=1.2)
ax1.text(year_axis[crossing_index] + 80, -0.9, 'forcing crosses\nthreshold',
         fontsize=8)
ax1.set_xlabel('Year')
ax1.set_ylabel('Basin state')
ax1.set_title('One threshold crossing, three response times')
ax1.legend(fontsize=8, loc='center right')


# --- 2. Commitment lag scales with the rate constant --------------------

relaxations = np.logspace(0, 3.3, 14)
lag_years = []
for relaxation in relaxations:
    traj = basins.simulate([basins.COLD_STATE], ramp[:, None], None, dt,
                           rates=1.0 / relaxation)
    lag_years.append(basins.commitment_lag(traj, ramp[:, None], dt)[0])
lag_years = np.array(lag_years)

finite = np.isfinite(lag_years)
ax2.loglog(relaxations[finite], lag_years[finite], 'o-', ms=5, color='C0')
for name, relaxation, colour in LADDER:
    ax2.axvline(relaxation, color=colour, ls=':', lw=1.5)
    ax2.text(relaxation * 1.1, lag_years[finite].min() * 1.4, name,
             rotation=90, fontsize=7, color=colour)
ax2.set_xlabel('Basin relaxation time (years)')
ax2.set_ylabel('Years from commitment to visible transition')
ax2.set_title('The gap between deciding and showing')


# --- 3. What is visible against what is diagnostic ----------------------
# The equilibrium drifts smoothly toward the saddle and then jumps.
# The drift is visible but looks like an ordinary trend. The
# slowing is what actually says a threshold is close.

fractions = np.linspace(0.0, 0.9999, 300)
forcings = basins.CRITICAL_FORCING * fractions
positions = np.array([basins.equilibria(c)[0] for c in forcings])
recovery = np.array([basins.recovery_rate(c) for c in forcings])

ax3.plot(fractions * 100, positions, color='C0', lw=2,
         label='equilibrium position')
ax3.set_xlabel('Forcing (% of threshold)')
ax3.set_ylabel('Equilibrium state', color='C0')
ax3.axhline(-1 / np.sqrt(3), color='C0', ls=':', lw=1)

twin = ax3.twinx()
twin.plot(fractions * 100, recovery / recovery[0], color='C3', lw=2,
          label='recovery rate')
twin.set_ylabel('Recovery rate (fraction of undisturbed)', color='C3')
ax3.set_title('The state drifts 21% and stops.\nThe recovery rate goes to zero')


# --- 4. Early warning signals -------------------------------------------
# Measured at fixed forcing levels rather than along the ramp.
#
# The first attempt detrended a ramped run and got indicators that
# fell as the threshold approached — the detrending window and the
# measurement window were the same width, so the filter removed
# exactly the variability being measured. At fixed forcing there is
# nothing to detrend and the analytic prediction is available as a
# check.
#
# For an Ornstein-Uhlenbeck process of rate lambda, stationary
# variance goes as 1/lambda and lag-1 autocorrelation as
# exp(-lambda). Both diverge toward the threshold, where lambda
# reaches zero.

ews_relaxation = 30.0
ews_rate = 1.0 / ews_relaxation
ews_dt = 0.05
ews_steps = 300000
ews_rng = np.random.default_rng(5)

levels = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995])
measured_variance, measured_ac, predicted_ac = [], [], []

for level in levels:
    c = basins.CRITICAL_FORCING * level
    f = c + 0.010 * ews_rng.standard_normal(ews_steps)
    traj = basins.simulate([basins.equilibria(c)[0]], f[:, None], None,
                           ews_dt, rates=ews_rate)[:, 0]
    annual = traj[ews_steps // 4:][::int(1.0 / ews_dt)]
    annual = annual - annual.mean()
    measured_variance.append(annual.var())
    measured_ac.append(float(annual[:-1] @ annual[1:] / (annual @ annual)))
    predicted_ac.append(np.exp(-basins.recovery_rate(c, rate=ews_rate)))

measured_variance = np.array(measured_variance)
measured_ac = np.array(measured_ac)

ax4.plot(levels * 100, measured_variance / measured_variance[0], 'o-', ms=5,
         color='C0', label='variance (relative)')
ax4.set_xlabel('Forcing (% of threshold)')
ax4.set_ylabel('Variance, relative to 40% of threshold', color='C0')

twin4 = ax4.twinx()
twin4.plot(levels * 100, measured_ac, 's-', ms=5, color='C3',
           label='lag-1 autocorrelation')
twin4.plot(levels * 100, predicted_ac, ':', color='k', lw=1.5,
           label='analytic exp(−λ)')
twin4.set_ylabel('Lag-1 autocorrelation', color='C3')
twin4.legend(fontsize=7, loc='lower right')
ax4.set_title(f'Early warning, {ews_relaxation:.0f} yr basin\n'
              '(measured against theory)')


# --- 5. The reversibility window ----------------------------------------
# After the threshold is crossed, how long can the forcing be
# restored and still prevent the transition?

# Two earlier attempts got this wrong and both are worth recording.
#
# The first let the ramp keep rising during the delay, so a basin
# waiting ten of its own relaxation times had also accumulated ten
# times more overshoot — geometry confounded with ramp rate. The
# second held overshoot fixed but swept only to six relaxation
# times, and at a 2% overshoot the window had not closed by then, so
# every basin read "still open" and the sweep bound was reported as
# though it were a result.
#
# With overshoot held fixed the three basins are the same system up
# to a rescaling of time, so this is computed ONCE at unit rate and
# converted afterwards. The window is swept against overshoot size,
# which is what actually sets it: escape near a saddle-node slows as
# the overshoot shrinks.

overshoots = np.array([1.01, 1.02, 1.05, 1.10, 1.20, 1.40, 1.80])
unit_dt = 0.005
delay_grid = np.linspace(0.0, 40.0, 90)      # in relaxation times
safe_level = basins.CRITICAL_FORCING * 0.80
start_state = basins.equilibria(safe_level)[0]

windows = []
for overshoot in overshoots:
    closed_at = np.inf
    for delay in delay_grid:
        n = int((delay + 60.0) / unit_dt)
        f = np.full(n, basins.CRITICAL_FORCING * overshoot)
        f[int(delay / unit_dt):] = safe_level
        traj = basins.simulate([start_state], f[:, None], None, unit_dt,
                               rates=1.0)
        if traj[-1, 0] > 0:                  # lost
            closed_at = delay
            break
    windows.append(closed_at)
windows = np.array(windows)

ax5.plot((overshoots - 1) * 100, windows, 'o-', ms=6, color='k')
ax5.set_xscale('log')
ax5.set_xlabel('Overshoot past the threshold (%)')
ax5.set_ylabel('Reversibility window (relaxation times)')
ax5.set_title('The window closes as the overshoot grows —\n'
              'and it is measured in relaxation times')

reference = int(np.argmin(np.abs(overshoots - 1.10)))
window_relax = windows[reference]
window_results = {name: window_relax * relaxation
                  for name, relaxation, _ in LADDER}
for i, (name, relaxation, colour) in enumerate(LADDER):
    ax5.text(0.04, 0.86 - 0.13 * i,
             f'at 10% overshoot — {name}: {window_relax * relaxation:,.0f} yr',
             color=colour, fontsize=8, transform=ax5.transAxes)


# --- 6. All committed; one of them looks it -----------------------------

checkpoints = [50, 100, 500, 1000]
realised = np.zeros((len(LADDER), len(checkpoints)))
for i, (name, relaxation, colour) in enumerate(LADDER):
    for j, elapsed in enumerate(checkpoints):
        index = min(crossing_index + int(elapsed / dt), steps - 1)
        realised[i, j] = np.clip(
            (ladder[index, i] - basins.COLD_STATE) / 2.0, 0, 1)

position = np.arange(len(checkpoints))
width = 0.26
for i, (name, relaxation, colour) in enumerate(LADDER):
    ax6.bar(position + (i - 1) * width, realised[i] * 100, width,
            color=colour, label=name)
ax6.set_xticks(position)
ax6.set_xticklabels([f'{c} yr' for c in checkpoints])
ax6.set_xlabel('Years after commitment')
ax6.set_ylabel('Transition completed (%)')
ax6.set_title('All three crossed at the same moment')
ax6.legend(fontsize=8)

plt.tight_layout()
plt.savefig(ROOT / 'Sims' / 'commitment_output.png', dpi=140)


# --- Printed diagnostics --------------------------------------------------

def rule(text):
    print(f'\n{text}\n' + '-' * len(text))


rule('One crossing, three timescales')
print(f'forcing crosses threshold at year {year_axis[crossing_index]:.0f}\n')
print(f'{"basin":<12}{"relaxation":>12}{"visible after":>15}')
for i, (name, relaxation, _) in enumerate(LADDER):
    lag = lags[i]
    text = f'{lag:,.0f} yr' if np.isfinite(lag) else 'not yet'
    print(f'{name:<12}{relaxation:>11.0f}y{text:>15}')
print('\n  Same forcing, same threshold, same moment of commitment. The')
print('  only difference is the rate constant, and it decides whether')
print('  the transition is a news story or an archaeological one.')

rule('What an observer sees')
print(f'{"basin":<12}' + ''.join(f'{c:>10} yr' for c in checkpoints))
for i, (name, _, _) in enumerate(LADDER):
    print(f'{name:<12}' + ''.join(f'{v * 100:>12.0f}%' for v in realised[i]))
print('\n  A century after committing, sea ice has completed its')
print('  transition and the ice sheet has moved a few percent. Someone')
print('  watching the ice sheet sees a slow trend. The commitment')
print('  already happened.')

rule('Visible against diagnostic')
saddle = -1 / np.sqrt(3)
print(f'equilibrium at zero forcing        {basins.equilibria(0.0)[0]:+.3f}')
print(f'equilibrium at the threshold       {saddle:+.3f}')
print(f'so the state drifts                '
      f'{abs(saddle + 1) / 2 * 100:.0f}% of the full transition, then jumps')
print()
print(f'{"forcing":>10}{"recovery rate":>15}{"relax time":>13}')
for fraction in (0.0, 0.5, 0.9, 0.99, 0.999):
    c = basins.CRITICAL_FORCING * fraction
    print(f'{fraction * 100:>9.1f}%{basins.recovery_rate(c):>15.4f}'
          f'{basins.relaxation_time(c):>13.2f}')
print('\n  The drift is real but reads as an ordinary trend — a smooth')
print('  move to 21% and then nothing more. The recovery rate is what')
print('  actually carries the information, and it falls to zero. At')
print('  99.9% of threshold a basin takes 38 times longer to recover')
print('  from a perturbation than an undisturbed one, while its mean')
print('  state has barely moved since 90%.')

rule('Early warning')
print(f'{ews_relaxation:.0f} yr basin, forcing held at each level\n')
print(f'{"% of threshold":>15}{"recovery rate":>15}{"variance":>12}'
      f'{"lag-1 AC":>11}{"theory":>9}')
for level, var, ac, theory in zip(levels, measured_variance, measured_ac,
                                  predicted_ac):
    print(f'{level * 100:>14.1f}%'
          f'{basins.recovery_rate(basins.CRITICAL_FORCING * level, rate=ews_rate):>15.4f}'
          f'{var / measured_variance[0]:>11.1f}x{ac:>11.4f}{theory:>9.4f}')
print(f'\n  Variance rises {measured_variance[-1] / measured_variance[0]:.0f}x '
      f'and autocorrelation goes {measured_ac[0]:.3f} -> {measured_ac[-1]:.3f},')
print('  matching the analytic exp(−λ) to three decimals. The state')
print('  meanwhile has drifted 21% and looks like a trend.')
print('\n  These are not forecasts. They say a basin is losing')
print('  resilience, not when it will go. During the gap between')
print('  commitment and transition they are the only signal there is.')

rule('The reversibility window')
print(f'{"overshoot":>11}{"window (relaxation times)":>28}')
for overshoot, window in zip(overshoots, windows):
    text = f'{window:.1f}' if np.isfinite(window) else '> sweep'
    print(f'{(overshoot - 1) * 100:>10.0f}%{text:>28}')
print(f'\nat a {(overshoots[reference] - 1) * 100:.0f}% overshoot, '
      f'{window_relax:.1f} relaxation times:\n')
print(f'{"basin":<12}{"relaxation":>12}{"window in years":>18}')
for name, relaxation, _ in LADDER:
    print(f'{name:<12}{relaxation:>11.0f}y{window_results[name]:>17,.0f}')
print('\n  Measured in relaxation times the window is one number for all')
print('  three basins — with overshoot held fixed they are the same')
print('  system up to a rescaling of time. In years they are not, and')
print('  the slow basin looks generous.')
print('\n  That generosity is the trap. During those centuries the state')
print('  has barely moved, so nothing in the observation record says')
print('  the window is open, or that it is closing. By the time the')
print('  transition is visible the window shut long ago, and recovery')
print('  then requires driving forcing to the far side of zero rather')
print('  than merely undoing the overshoot.')

rule('On "any day now"')
print('  For ice sheets it is wrong: the transition takes centuries to')
print('  millennia once begun, and the model reproduces that from the')
print('  rate constant alone.')
print()
print('  For sea ice it is not wrong, and 2016 already happened.')
print('  Conflating the two is what makes the claim sound either')
print('  hysterical or dismissible depending on which one the reader')
print('  has in mind.')
print()
print('  Neither reading is reassuring, because the quantity that')
print('  matters is not when the transition completes but whether the')
print('  commitment is already made — and commitment is invisible by')
print('  construction in exactly the systems where it lasts longest.')

plt.show()
