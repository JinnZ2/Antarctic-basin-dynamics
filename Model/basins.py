"""Multiple coupled basins and cascading destabilisation.

Closes the gap recorded in Docs/literature.md section 8.

The Antarctic Ice Sheet is not one tipping element. It behaves
as several tipping systems interacting across drainage basins,
and most known interactions between tipping elements are
destabilising -- one element tipping makes another more likely
to tip. A single-basin model cannot represent that, and the
cascade is the behaviour the tipping-point literature
identifies as the dangerous one.

Each basin is a normalised double well:

    dx/dt = x - x^3 + c + sum_j d_ij * (x_j + 1) / 2

x = -1 is the cold, structured state; x = +1 is the
reorganised state. c is external forcing. The saddle-node
sits at |c| = 2 / (3 * sqrt(3)) ~= 0.3849, which is the
analytic value used to check the implementation.

Coupling enters as forcing: a neighbour in the cold state
(x = -1) contributes nothing, a neighbour that has tipped
(x = +1) contributes its full d_ij. So a tipped basin
increases the effective forcing on its neighbours, which can
push them past a threshold they would not have crossed alone.

This also closes additions.md item 5. Basin depth here is a
real potential barrier rather than a clipped energy balance,
so it goes negative naturally once the well is gone: negative
depth means the attractor no longer exists, not that the
energy budget is merely bad.

Literature: Docs/literature.md section 8, Docs/structure.md.
"""

import numpy as np

# Saddle-node bifurcation of dx/dt = x - x^3 + c.
CRITICAL_FORCING = 2.0 / (3.0 * np.sqrt(3.0))

COLD_STATE = -1.0
TIPPED_STATE = 1.0


def potential(x, c=0.0):
    """V(x) such that dx/dt = -dV/dx."""
    x = np.asarray(x, dtype=float)
    return -0.5 * x ** 2 + 0.25 * x ** 4 - c * x


def drift(x, c=0.0):
    """Uncoupled rate of change."""
    x = np.asarray(x, dtype=float)
    return x - x ** 3 + c


def equilibria(c):
    """Real roots of x^3 - x - c = 0, ascending.

    Three roots inside the bistable window, one outside.
    """
    roots = np.roots([1.0, 0.0, -1.0, -float(c)])
    real = np.sort(np.real(roots[np.abs(np.imag(roots)) < 1e-9]))
    return real


def basin_depth(c, state=COLD_STATE):
    """Potential barrier from a well to the saddle.

    Positive: the barrier that must be overcome to reorganise.
    This is the quantity `basin_depth` in the scalar model was
    reaching for.

    Negative: the well no longer exists. Reported as the
    distance past the bifurcation rather than clipped to zero,
    so that destabilisation is represented structurally instead
    of being suppressed. This is additions.md item 5.
    """
    c = float(c)
    roots = equilibria(c)
    if len(roots) < 3:
        return -(abs(c) - CRITICAL_FORCING)

    low, saddle, high = roots[0], roots[1], roots[2]
    well = low if state < 0 else high
    return float(potential(saddle, c) - potential(well, c))


def _coupled_drift(x, c, D, rates):
    """Rate of change with destabilising coupling."""
    influence = D @ ((x + 1.0) / 2.0)
    return rates * (x - x ** 3 + c + influence)


def simulate(x0, forcing, D=None, dt=0.01, rates=None):
    """Integrate the coupled system with RK4.

    `forcing` is an (n_steps, n_basins) array of c values.
    `D` is the coupling matrix; D[i, j] is the push basin j
    exerts on basin i once j has tipped. The diagonal is
    ignored.

    `rates` scales each basin's response speed, so basins with
    different relaxation times can be run together. This is not
    cosmetic: a sea-ice basin responds in a year or two while an
    ice-sheet or ecosystem basin takes decades, and whether a
    brief event matters depends entirely on which one it hits.
    Rescaling time leaves the equilibria and the saddle-node
    untouched.

    Returns trajectories of shape (n_steps, n_basins).
    """
    forcing = np.atleast_2d(np.asarray(forcing, dtype=float))
    n_steps, n_basins = forcing.shape

    D = np.zeros((n_basins, n_basins)) if D is None else np.array(D, float)
    np.fill_diagonal(D, 0.0)
    rates = (np.ones(n_basins) if rates is None
             else np.broadcast_to(np.asarray(rates, dtype=float),
                                  (n_basins,)))

    x = np.asarray(x0, dtype=float).copy()
    out = np.empty((n_steps, n_basins))

    for k in range(n_steps):
        c = forcing[k]
        k1 = _coupled_drift(x, c, D, rates)
        k2 = _coupled_drift(x + 0.5 * dt * k1, c, D, rates)
        k3 = _coupled_drift(x + 0.5 * dt * k2, c, D, rates)
        k4 = _coupled_drift(x + dt * k3, c, D, rates)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k] = x

    return out


def recovery_forcing(state=TIPPED_STATE):
    """Forcing required to undo a tip.

    Having crossed at +CRITICAL_FORCING, the upper state does
    not disappear until forcing falls to -CRITICAL_FORCING. The
    two are not the same number, and the gap between them is
    why a transition that took a brief excursion to trigger can
    take a sustained reversal to undo.
    """
    return -CRITICAL_FORCING if state > 0 else CRITICAL_FORCING


def hysteresis_width():
    """Separation between the tipping and recovery thresholds."""
    return 2.0 * CRITICAL_FORCING


def latched(trajectories, forcing, threshold=0.0):
    """Basins left in the upper state while forcing is sub-critical.

    The operational test for a transition that outlasts its
    trigger: the forcing has come back and the state has not.
    """
    traj = np.atleast_2d(trajectories)
    forcing = np.atleast_2d(np.asarray(forcing, dtype=float))
    return (traj[-1] > threshold) & (forcing[-1] < CRITICAL_FORCING)


def tipping_steps(trajectories, threshold=0.0):
    """First step index at which each basin crosses to the upper state.

    Returns -1 for basins that never tip.
    """
    traj = np.atleast_2d(trajectories)
    crossed = traj > threshold
    ever = crossed.any(axis=0)
    first = np.argmax(crossed, axis=0)
    return np.where(ever, first, -1)


def cascade_attribution(x0, forcing, D, dt=0.01, threshold=0.0):
    """Which tips are caused by coupling rather than by forcing.

    Runs the system twice, once with coupling and once without,
    under identical forcing. A basin that tips only in the
    coupled run tipped because a neighbour did. A basin that
    tips earlier in the coupled run was brought forward.

    Returns a dict with per-basin tip steps and the derived
    classification.
    """
    coupled = tipping_steps(simulate(x0, forcing, D, dt), threshold)
    isolated = tipping_steps(simulate(x0, forcing, None, dt), threshold)

    cascade_only = (coupled >= 0) & (isolated < 0)
    brought_forward = np.where(
        (coupled >= 0) & (isolated >= 0), isolated - coupled, 0)

    return {
        'coupled_step': coupled,
        'isolated_step': isolated,
        'cascade_only': cascade_only,
        'steps_brought_forward': brought_forward,
    }


def susceptibility(c, D):
    """Headroom each basin has left, before and after neighbours tip.

    Returns (margin_now, margin_if_neighbours_tip). A negative
    second value means the basin survives its own forcing but
    not its neighbours' tipping -- it is being held up by them
    staying put.

    This is the number that distinguishes a system of
    independent thresholds from a system with a cascade waiting
    in it.

    The second value assumes EVERY neighbour tips. That is the
    worst case, not the next step: a cascade that halts partway
    leaves basins sitting between the two margins, which is why
    a negative margin_after does not by itself predict a tip.
    """
    c = np.asarray(c, dtype=float)
    D = np.array(D, dtype=float)
    np.fill_diagonal(D, 0.0)

    margin_now = CRITICAL_FORCING - c
    margin_after = CRITICAL_FORCING - (c + D.sum(axis=1))
    return margin_now, margin_after


def effective_forcing(trajectories, forcing, D):
    """External forcing plus the push from already-tipped neighbours.

    This is the forcing each basin actually experiences. Basin
    depth computed from the external term alone understates the
    damage during a cascade, because the neighbour contribution
    is exactly what eats the remaining barrier.
    """
    traj = np.atleast_2d(np.asarray(trajectories, dtype=float))
    forcing = np.atleast_2d(np.asarray(forcing, dtype=float))
    D = np.array(D, dtype=float)
    np.fill_diagonal(D, 0.0)
    return forcing + ((traj + 1.0) / 2.0) @ D.T


def recovery_rate(c, state=COLD_STATE, rate=1.0):
    """How fast a basin returns after a small perturbation.

    The linearisation |f'(x*)| = |1 - 3 x*^2| at the stable
    equilibrium, times the basin's own rate constant.

    It is 2 at zero forcing and falls to exactly **zero** at the
    saddle-node, where the equilibrium and the saddle merge.
    That is critical slowing down, and it is analytic here
    rather than fitted: a basin approaching its threshold takes
    longer and longer to recover from anything, without its
    state having visibly moved.

    Returns 0.0 outside the bistable window, where the well
    being asked about no longer exists.
    """
    roots = equilibria(c)
    if len(roots) < 3:
        return 0.0
    well = roots[0] if state < 0 else roots[-1]
    return float(rate * abs(1.0 - 3.0 * well ** 2))


def relaxation_time(c, state=COLD_STATE, rate=1.0):
    """Reciprocal of the recovery rate. Diverges at the threshold."""
    speed = recovery_rate(c, state, rate)
    return np.inf if speed <= 0 else 1.0 / speed


def commitment_lag(trajectories, forcing, dt=0.01, threshold=0.0):
    """Time between forcing crossing critical and the state crossing.

    The gap between committing to a transition and displaying
    one. For a fast basin it is short. For a slow basin it can
    be centuries, during which the state barely moves and
    nothing looks wrong.

    Returns np.inf if the state never crosses, and np.nan if the
    forcing never does.
    """
    traj = np.atleast_2d(trajectories)
    forcing = np.atleast_2d(np.asarray(forcing, dtype=float))

    lags = []
    for i in range(traj.shape[1]):
        past = forcing[:, i] >= CRITICAL_FORCING
        if not past.any():
            lags.append(np.nan)
            continue
        committed = int(np.argmax(past))
        crossed = tipping_steps(traj[:, [i]], threshold)[0]
        lags.append(np.inf if crossed < 0 else (crossed - committed) * dt)

    return np.array(lags)


def rolling_variance(series, window):
    """Variance in a trailing window. Rises as a basin destabilises."""
    series = np.asarray(series, dtype=float)
    window = int(window)
    out = np.full(len(series), np.nan)
    for k in range(window, len(series)):
        out[k] = series[k - window:k].var()
    return out


def rolling_autocorrelation(series, window, lag=1):
    """Lag-1 autocorrelation in a trailing window.

    Approaches 1 as recovery slows. With variance, the standard
    pair of early-warning indicators, and the only way to see a
    commitment while it is being made rather than centuries
    later when the state finally moves.
    """
    series = np.asarray(series, dtype=float)
    window, lag = int(window), int(lag)
    out = np.full(len(series), np.nan)

    for k in range(window, len(series)):
        chunk = series[k - window:k]
        chunk = chunk - chunk.mean()
        denominator = float(chunk @ chunk)
        if denominator <= 0:
            continue
        out[k] = float(chunk[:-lag] @ chunk[lag:]) / denominator

    return out


def ring_coupling(n, strength):
    """Destabilising coupling between circumpolar neighbours.

    Uniform and symmetric, which no real system is. Real
    interaction strengths between Antarctic tipping systems are
    not established -- the literature supports the *sign* (most
    interactions are destabilising), not the magnitude.
    Treat `strength` as the parameter to sweep, and the
    uniformity as a placeholder for a matrix nobody has
    measured yet.
    """
    D = np.zeros((n, n))
    for i in range(n):
        D[i, (i + 1) % n] = strength
        D[i, (i - 1) % n] = strength
    return D
