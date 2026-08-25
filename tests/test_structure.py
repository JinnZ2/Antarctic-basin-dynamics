"""Checks for the structural layers.

Where a quantity has a known analytic value, it is checked against
that value rather than against a previously recorded output. Square
lattice bond percolation has a threshold of 0.5; the saddle-node of
dx/dt = x - x^3 + c sits at 2/(3*sqrt(3)); the potential barrier at
zero forcing is exactly 0.25. Those are the anchors.

Run with pytest, or directly:  python tests/test_structure.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import basins
import climate_modes as cm
import core
import population as pop
import spatial as sp


# --- spatial --------------------------------------------------------


def test_components_on_a_known_graph():
    # 7 nodes: {0,1,2}, {3}, {4,5}, {6}
    edges = np.array([[0, 1], [1, 2], [4, 5]])
    labels, sizes = sp._components(7, edges)
    assert sorted(sizes.tolist()) == [1, 1, 2, 3], sizes
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5]
    assert labels[3] not in (labels[0], labels[4])


def test_lattice_is_periodic_in_longitude_only():
    edges, _, _, (rows, cols) = sp.build_lattice(4, 10)
    # Every node has a longitudinal bond; wrapping makes the count
    # rows*cols. Depth bonds are one fewer per column.
    assert len(edges) == rows * cols + (rows - 1) * cols
    degree = np.bincount(edges.ravel(), minlength=rows * cols)
    assert degree.min() >= 3          # interior rows reach 4


def test_full_lattice_is_one_component():
    edges, _, _, (rows, cols) = sp.build_lattice(5, 12)
    assert sp.giant_fraction(rows * cols, edges) == 1.0


def test_square_lattice_recovers_the_analytic_threshold():
    """Bond percolation on a square lattice has p_c = 0.5."""
    rng = np.random.default_rng(4)
    p = np.linspace(0.30, 0.75, 46)
    giant = sp.percolation_sweep(p, rows=60, cols=60, reps=10, rng=rng)
    p_c = sp.critical_probability(p, giant)
    assert abs(p_c - 0.5) < 0.05, f'p_c = {p_c}'


def test_thin_strip_fragments_before_a_square_does():
    """A narrow band loses continuity at a higher bond probability."""
    rng = np.random.default_rng(5)

    def mean_giant(rows, cols, p, reps=6):
        edges, _, _, _ = sp.build_lattice(rows, cols)
        probability = np.full(len(edges), p)
        return float(np.mean([
            sp.giant_fraction(rows * cols, sp.occupy(edges, probability, rng))
            for _ in range(reps)]))

    assert mean_giant(6, 60, 0.55) < mean_giant(60, 60, 0.55)


def test_redistribution_index_is_one_when_sectors_agree():
    uniform = np.full((1, 6), 0.8)
    assert abs(sp.redistribution_index(uniform)[0] - 1.0) < 1e-9


def test_redistribution_index_detects_cancelling_sectors():
    """Sectors moving oppositely: large gross change, no net change."""
    supply = np.array([[1.2, 1.2, 1.2, 0.8, 0.8, 0.8]])
    assert sp.redistribution_index(supply)[0] > 1e6


def test_fragmentation_reduces_connectivity_monotonically():
    years = np.array([0.0, 30.0, 60.0, 90.0])
    conn = sp.connectivity_trajectory(
        years, 8, 60, reps=6, rng=np.random.default_rng(6))
    assert np.all(np.diff(conn) <= 1e-9), conn


# --- population -----------------------------------------------------


def test_literature_parameters_land_near_stationarity():
    """Independently sourced demography should not be wildly off.

    Litter size, maturity age and lifespan come from separate
    studies; nothing forces them to be consistent. Landing within a
    few percent of replacement is a weak but real check that the
    schedule is not nonsense.
    """
    lam = pop.growth_rate(pop.build_leslie_matrix())
    assert 0.99 < lam < 1.01, lam


def test_calibration_hits_replacement_exactly():
    F = pop.calibrate_fecundity()
    assert abs(pop.growth_rate(pop.build_leslie_matrix(F)) - 1.0) < 1e-6


def test_euler_lotka_agrees_with_the_eigenvalue():
    F = pop.calibrate_fecundity()
    for scale in (0.3, 1.0, 2.0):
        lx, mx = pop.life_table(fecundity=F * scale)
        direct = pop.growth_rate_from_life_table(lx, mx)
        eigen = pop.growth_rate(pop.build_leslie_matrix(F * scale))
        assert abs(direct - eigen) < 1e-3, (scale, direct, eigen)


def test_generation_time_exceeds_age_at_maturity():
    """Parents are on average older than the age they first bred."""
    F = pop.calibrate_fecundity()
    lx, mx = pop.life_table(fecundity=F)
    assert pop.generation_time(lx, mx) > pop.MATURITY_AGE


def test_memory_is_not_lifespan():
    """The point of the demographic layer.

    The damping timescale is a different quantity from lifespan and
    is not close to it. If these ever coincide, the scalar proxy was
    adequate and this module is not earning its cost.
    """
    F = pop.calibrate_fecundity()
    memory = pop.memory_years(pop.build_leslie_matrix(F))
    assert memory > 1.3 * pop.MAX_AGE, memory


def test_transient_period_is_near_generation_time():
    """Cohort echo: perturbations ring at roughly a generation."""
    F = pop.calibrate_fecundity()
    A = pop.build_leslie_matrix(F)
    lx, mx = pop.life_table(fecundity=F)
    period, T = pop.transient_period(A), pop.generation_time(lx, mx)
    assert 0.8 < period / T < 1.25, (period, T)


def test_warming_compresses_the_age_axis():
    cold = pop.warmed_ages(0.0, 2.5)
    warm = pop.warmed_ages(6.0, 2.5)
    assert warm[0] < cold[0] and warm[1] < cold[1]


def test_decoupling_damps_compression():
    strict = pop.warmed_ages(6.0, 2.5, decoupling=0.0)
    damped = pop.warmed_ages(6.0, 2.5, decoupling=0.5)
    none = pop.warmed_ages(6.0, 2.5, decoupling=1.0)
    assert strict[1] < damped[1] < none[1]
    assert none[1] == pop.MATURITY_AGE


def test_allee_is_off_by_default():
    assert pop.allee_multiplier(0.001, 0.0) == 1.0


def test_allee_bites_below_the_threshold():
    assert pop.allee_multiplier(50.0, 100.0) == 0.5
    assert pop.allee_multiplier(150.0, 100.0) == 1.0


def test_stage_aggregation_conserves_total():
    F = pop.calibrate_fecundity()
    w = pop.stable_age_distribution(pop.build_leslie_matrix(F))
    stages = pop.aggregate_to_stages(w)
    assert len(stages) == len(pop.STAGES)
    assert abs(stages.sum() - w.sum()) < 1e-12


def test_supply_shortfall_reduces_growth():
    F = pop.calibrate_fecundity()
    lx, mx = pop.life_table(fecundity=F * 0.5)
    assert pop.growth_rate_from_life_table(lx, mx) < 1.0


# --- basins ---------------------------------------------------------


def test_critical_forcing_is_the_analytic_saddle_node():
    assert abs(basins.CRITICAL_FORCING - 2 / (3 * np.sqrt(3))) < 1e-12


def test_equilibria_at_zero_forcing():
    assert np.allclose(basins.equilibria(0.0), [-1.0, 0.0, 1.0], atol=1e-9)


def test_barrier_at_zero_forcing_is_one_quarter():
    assert abs(basins.basin_depth(0.0) - 0.25) < 1e-12


def test_depth_shrinks_to_zero_at_the_bifurcation():
    depths = [basins.basin_depth(c) for c in (0.0, 0.2, 0.35, 0.384)]
    assert all(np.diff(depths) < 0)
    assert depths[-1] < 1e-3


def test_depth_goes_negative_past_the_bifurcation():
    """additions.md item 5: destabilisation is represented, not clipped."""
    assert basins.basin_depth(basins.CRITICAL_FORCING + 0.1) < 0


def test_an_uncoupled_basin_holds_below_its_threshold():
    forcing = np.full((4000, 1), 0.30)
    traj = basins.simulate([-1.0], forcing, dt=0.01)
    assert traj[-1, 0] < 0
    assert basins.tipping_steps(traj)[0] == -1


def test_an_uncoupled_basin_tips_above_its_threshold():
    forcing = np.full((4000, 1), 0.45)
    traj = basins.simulate([-1.0], forcing, dt=0.01)
    assert traj[-1, 0] > 0


def test_coupling_tips_a_basin_that_would_have_held():
    """The cascade, in its smallest form.

    Basin 0 sits below its own threshold. Basin 1 is above its own.
    Coupled, basin 1 tips and drags basin 0 with it.
    """
    forcing = np.tile([0.34, 0.45], (12000, 1))
    D = np.array([[0.0, 0.08], [0.08, 0.0]])
    result = basins.cascade_attribution([-1.0, -1.0], forcing, D, dt=0.01)
    assert result['isolated_step'][0] == -1
    assert result['coupled_step'][0] >= 0
    assert bool(result['cascade_only'][0])


def test_no_cascade_without_coupling():
    forcing = np.tile([0.34, 0.45], (12000, 1))
    D = np.zeros((2, 2))
    result = basins.cascade_attribution([-1.0, -1.0], forcing, D, dt=0.01)
    assert not result['cascade_only'].any()


def test_effective_forcing_includes_tipped_neighbours():
    D = basins.ring_coupling(3, 0.1)
    cold = basins.effective_forcing(np.full((1, 3), -1.0), np.zeros((1, 3)), D)
    hot = basins.effective_forcing(np.full((1, 3), 1.0), np.zeros((1, 3)), D)
    assert np.allclose(cold, 0.0)
    assert np.allclose(hot, 0.2)


def test_susceptibility_flags_a_basin_held_up_by_its_neighbours():
    D = basins.ring_coupling(3, 0.08)
    now, after = basins.susceptibility(np.full(3, 0.30), D)
    assert np.all(now > 0)
    assert np.all(after < 0)


# --- core neutrality, restated against the new layers ---------------


def test_core_additions_still_reduce_to_prior_behaviour():
    assert np.isclose(core.dynamic_transfer_efficiency(0.10, 3.0, 0.0), 0.10)
    assert np.isclose(core.maintenance_adjusted_lifespan(300, 2.0, 0.0),
                      core.adjusted_lifespan(300, 2.0))
    assert np.isclose(core.mass_dependent_connectivity(0.75, 0.5, 1.0),
                      0.75 ** 0.5)
    assert np.isclose(core.accelerating_temperature(1.3, 10, 0.0075, 0.0),
                      1.3 + 0.075)


# --- climate modes --------------------------------------------------


def test_enso_index_is_standardised():
    index = cm.enso_index(5000, np.random.default_rng(9))
    assert abs(index.std() - 1.0) < 1e-9
    assert abs(index.mean()) < 0.1


def test_enso_peak_falls_in_the_two_to_seven_year_band():
    """The standard description of ENSO. Not a tuned outcome — the
    oscillator is specified by period and damping, and this checks
    the realised spectrum matches the specification."""
    for seed in range(6):
        period = cm.spectral_peak(
            cm.enso_index(2000, np.random.default_rng(seed)))
        assert 2.0 <= period <= 7.0, (seed, period)


def test_enso_is_broadband_not_a_sine():
    """A pure sinusoid would make the filtering results look sharper
    than the real forcing warrants."""
    index = cm.enso_index(4000, np.random.default_rng(11))
    spectrum = np.abs(np.fft.rfft(index - index.mean())) ** 2
    spectrum[0] = 0.0
    # A sine puts essentially all power in one bin.
    assert spectrum.max() / spectrum.sum() < 0.05


def test_depth_response_reverses_sign_at_the_transition():
    """Surface and subsurface must not share a sign: during El Niño
    the subsurface warms while the surface goes the other way."""
    assert cm.depth_weight(20.0) < 0
    assert cm.depth_weight(490.0) > 0
    assert abs(cm.depth_weight(cm.SUBSURFACE_DEPTH_M)) < 1e-9


def test_model_baseline_depth_sits_in_the_warming_band():
    """490 m is where the reference organism was observed and where
    the reported subsurface warming applies."""
    field = cm.temperature_anomaly(1.0, sp.SECTOR_NAMES, 8)
    row = int(np.argmin(np.abs(cm.row_depths(8) - 490.0)))
    peak = field[row, list(sp.SECTOR_NAMES).index('Bellingshausen')]
    assert peak > 0.9 * cm.ENSO_SUBSURFACE_AMPLITUDE_C, peak


def test_dipole_is_a_seesaw():
    """Ross and Bellingshausen must respond oppositely, or it is not
    a dipole."""
    anomaly = cm.habitat_anomaly(1.0, sp.SECTOR_NAMES)
    ross = anomaly[list(sp.SECTOR_NAMES).index('Ross')]
    bellingshausen = anomaly[list(sp.SECTOR_NAMES).index('Bellingshausen')]
    assert ross > 0 and bellingshausen < 0


def test_la_nina_reverses_el_nino():
    assert np.allclose(cm.habitat_anomaly(-1.0, sp.SECTOR_NAMES),
                       -cm.habitat_anomaly(1.0, sp.SECTOR_NAMES))


def test_teleconnection_weakens_after_the_decadal_shift():
    assert cm.teleconnection_strength(1990) > cm.teleconnection_strength(2015)


def test_warming_shortens_the_enso_period():
    assert cm.warmed_period(6.0) < cm.warmed_period(0.0)


def test_amplitude_default_is_no_change():
    """CMIP6 amplitude projections span both signs with an ensemble
    mean near zero, so the model must not pick one."""
    assert cm.warmed_amplitude(6.0) == cm.warmed_amplitude(0.0)


# --- memory as a filter ---------------------------------------------


def _gain_at(delta_T, forcing):
    F = pop.calibrate_fecundity()
    max_age, maturity, bounds = pop.warmed_ages(delta_T, 2.5, decoupling=0.5)
    A = pop.build_leslie_matrix(F, max_age, maturity, bounds)
    return pop.recruitment_transfer(A, forcing, maturity_age=maturity)


def test_the_slow_integrator_attenuates_enso_heavily():
    forcing = np.random.default_rng(13).standard_normal(8000)
    periods, gain = _gain_at(0.0, forcing)
    enso = pop.band_gain(periods, gain, 2, 7)
    century = pop.band_gain(periods, gain, 80, 300)
    assert enso < century / 10, (enso, century)


def test_warming_lets_more_high_frequency_variance_through():
    """geometry.md's central claim, which was untestable until the
    model contained an oscillation."""
    forcing = np.random.default_rng(13).standard_normal(8000)
    cold = pop.band_gain(*_gain_at(0.0, forcing), 2, 7)
    warm = pop.band_gain(*_gain_at(6.0, forcing), 2, 7)
    assert warm > cold, (cold, warm)


def test_attenuation_remains_large_even_when_degraded():
    """The claim holds directionally but the effect is a change in a
    very large number, not a breach of the filter."""
    forcing = np.random.default_rng(13).standard_normal(8000)
    warm = pop.band_gain(*_gain_at(6.0, forcing), 2, 7)
    assert warm < 0.01


def test_a_slow_basin_ignores_zero_mean_variability():
    """A basin integrates over its relaxation time, and ENSO averages
    to zero across any span longer than a decade."""
    dt, steps = 0.01, 60000
    time_per_year = 1.0 / 50.0
    year_axis = np.arange(steps) * dt / time_per_year
    mean_forcing = basins.CRITICAL_FORCING * 0.93
    annual = cm.enso_index(int(year_axis[-1]) + 2, np.random.default_rng(3))
    forcing = mean_forcing + 0.095 * np.interp(
        year_axis, np.arange(len(annual)), annual)
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt)
    assert basins.tipping_steps(traj)[0] == -1


def test_a_fast_basin_can_be_tipped_by_variability():
    dt, steps = 0.01, 60000
    time_per_year = 1.0 / 0.4
    year_axis = np.arange(steps) * dt / time_per_year
    annual = cm.enso_index(int(year_axis[-1]) + 2, np.random.default_rng(3))
    forcing = basins.CRITICAL_FORCING * 0.93 + 0.095 * np.interp(
        year_axis, np.arange(len(annual)), annual)
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt)
    assert basins.tipping_steps(traj)[0] >= 0


# --- extreme events -------------------------------------------------


def test_event_sigma_conversion():
    """1877-78 and 2015-16 both land near 3.5 sigma."""
    assert 3.4 < cm.event_sigma(2.73) < 3.6
    assert 3.4 < cm.event_sigma(2.75) < 3.6
    assert cm.event_sigma(3.6) > 4.5


def test_rectification_produces_positive_skew():
    plain = cm.enso_index(200000, np.random.default_rng(21), skewness=0.0)
    skewed = cm.enso_index(200000, np.random.default_rng(21),
                           skewness=cm.ENSO_SKEWNESS)

    def skew(x):
        return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)

    assert abs(skew(plain)) < 0.1
    assert skew(skewed) > 0.2


def test_rectification_preserves_unit_variance():
    for strength in (0.0, 0.08, 0.6):
        index = cm.enso_index(50000, np.random.default_rng(22),
                              skewness=strength)
        assert abs(index.std() - 1.0) < 1e-9


def test_skewness_defaults_off_so_earlier_results_are_unchanged():
    a = cm.enso_index(500, np.random.default_rng(23))
    b = cm.enso_index(500, np.random.default_rng(23), skewness=0.0)
    assert np.allclose(a, b)


def test_symmetric_generator_cannot_produce_a_record_event():
    """The defect the 2026-27 forecast exposed.

    A symmetric generator assigns an event that is currently
    happening a return period beyond any useful horizon.
    """
    period = cm.return_period(cm.event_sigma(3.6), skewness=0.0,
                              n_years=200000,
                              rng=np.random.default_rng(24))
    assert period > 100000 or not np.isfinite(period)


def test_skew_shortens_the_return_period():
    sigma = cm.event_sigma(2.73)
    plain = cm.return_period(sigma, skewness=0.0, n_years=300000,
                             rng=np.random.default_rng(25))
    skewed = cm.return_period(sigma, skewness=0.6, n_years=300000,
                              rng=np.random.default_rng(25))
    assert skewed < plain


def test_event_pulse_peaks_at_the_requested_size():
    pulse = cm.event_pulse(100, 4.65, start_year=40)
    assert abs(pulse.max() - 4.65) < 1e-9
    assert int(np.argmax(pulse)) == 40
    assert pulse[0] < 1e-6


def test_event_pulse_decays_slower_than_it_rises():
    """Extreme events are projected to show faster onset, slower decay."""
    pulse = cm.event_pulse(100, 1.0, onset_years=1.0, decay_years=1.5,
                           start_year=50)
    assert pulse[53] > pulse[47]


def test_record_event_is_a_large_fraction_of_total_warming():
    """The reason a single event matters at all: at 490 m it is
    comparable to the model's whole projected warming."""
    degrees = cm.subsurface_anomaly_C(cm.event_sigma(3.6))
    assert 0.6 < degrees / 2.0 < 1.0


def test_a_record_pulse_does_not_tip_a_slow_basin():
    """The central negative result, at the largest event on record.

    The pulse takes instantaneous forcing well past critical. A
    basin with a decadal relaxation time still holds, because it
    integrates rather than tracks.
    """
    dt, relaxation = 0.01, 30.0
    time_per_year = 1.0 / relaxation
    steps = int(400 * time_per_year / dt)
    year_axis = np.arange(steps) * dt / time_per_year

    sigma = cm.event_sigma(3.6)
    forcing_amplitude = cm.subsurface_anomaly_C(sigma) * 0.19
    pulse = cm.event_pulse(int(year_axis[-1]) + 2, sigma,
                           start_year=int(year_axis[-1] * 0.5))
    forcing = basins.CRITICAL_FORCING * 0.93 + (
        forcing_amplitude / sigma) * np.interp(
            year_axis, np.arange(len(pulse)), pulse)

    assert forcing.max() > basins.CRITICAL_FORCING      # overshoots
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt)
    assert basins.tipping_steps(traj)[0] == -1          # and holds anyway


def test_the_deficit_appears_only_after_maturity():
    """A destroyed cohort cannot be counted among adults until it
    would have matured."""
    F = pop.calibrate_fecundity()
    A = pop.build_leslie_matrix(F)
    start = pop.stable_age_distribution(A) * 1000
    horizon, event_year = 1200, 400

    pulse = cm.event_pulse(horizon, cm.event_sigma(3.6),
                           start_year=event_year)
    supply = np.clip(1.0 - 0.10 * pulse, 0.0, None)
    forced = pop.project(A, start, horizon, supply=supply)[:, pop.MATURITY_AGE:].sum(axis=1)
    control = pop.project(A, start, horizon)[:, pop.MATURITY_AGE:].sum(axis=1)
    deviation = forced / control - 1.0

    assert abs(deviation[event_year + 50]) < 1e-6      # nothing yet
    trough = int(np.argmin(deviation))
    assert trough - event_year > pop.MATURITY_AGE      # only after maturity
    assert deviation[trough] < 0


# --- persistence and the timescale cascade ---------------------------
# These encode a correction. An earlier run asked only whether a state
# crossed while a pulse lasted, never whether a crossing stays
# crossed, and never coupled a fast basin to a slow one.


def _pulse_forcing(relaxation, dt, years, sigma, precondition=0.93,
                   start_fraction=0.25):
    steps = int(years / dt)
    axis = np.arange(steps) * dt
    pulse = np.interp(axis, np.arange(int(axis[-1]) + 2),
                      cm.event_pulse(int(axis[-1]) + 2, sigma,
                                     start_year=int(start_fraction * years)))
    amplitude = cm.subsurface_anomaly_C(sigma) * 0.19 / sigma
    return basins.CRITICAL_FORCING * precondition + amplitude * pulse


def test_rates_rescale_time_without_moving_the_threshold():
    """A slower basin takes longer but tips at the same forcing."""
    steps = 40000
    forcing = np.full((steps, 1), 0.45)
    fast = basins.simulate([-1.0], forcing, None, 0.01, rates=1.0)
    slow = basins.simulate([-1.0], forcing, None, 0.01, rates=0.1)
    assert fast[-1, 0] > 0 and slow[-1, 0] > 0
    assert basins.tipping_steps(fast)[0] < basins.tipping_steps(slow)[0]


def test_recovery_threshold_is_not_the_tipping_threshold():
    assert basins.recovery_forcing() == -basins.CRITICAL_FORCING
    assert basins.hysteresis_width() == 2 * basins.CRITICAL_FORCING


def test_a_crossing_outlasts_the_event_that_caused_it():
    """The check the earlier sim never made.

    A two-year pulse tips a fast basin; the forcing returns to
    sub-critical and the state does not follow it back.
    """
    dt = 0.02
    forcing = _pulse_forcing(2.0, dt, 400, cm.event_sigma(3.6))
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt,
                           rates=1.0 / 2.0)
    assert forcing[-1] < basins.CRITICAL_FORCING       # forcing came back
    assert traj[-1, 0] > 0                             # the state did not
    assert bool(basins.latched(traj, forcing[:, None])[0])


def test_sea_ice_timescales_are_inside_the_vulnerable_band():
    """Antarctic sea ice responds on one to three years, which is
    exactly where a brief event does tip a basin."""
    dt = 0.02
    for relaxation in (1.0, 2.0):
        forcing = _pulse_forcing(relaxation, dt, 400, cm.event_sigma(3.6))
        traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None,
                               dt, rates=1.0 / relaxation)
        assert bool(basins.latched(traj, forcing[:, None])[0]), relaxation


def test_preconditioning_and_trigger_are_both_required():
    """The reported 2016 mechanism: a decade of thinning, then a wind
    event. Neither ingredient does it alone."""
    dt = 0.02

    # Trigger without preconditioning.
    weak = _pulse_forcing(2.0, dt, 200, cm.event_sigma(3.6),
                          precondition=0.60)
    traj = basins.simulate([basins.COLD_STATE], weak[:, None], None, dt,
                           rates=0.5)
    assert not bool(basins.latched(traj, weak[:, None])[0])

    # Preconditioning without a trigger.
    steps = int(200 / dt)
    flat = np.full((steps, 1), basins.CRITICAL_FORCING * 0.93)
    traj = basins.simulate([basins.COLD_STATE], flat, None, dt, rates=0.5)
    assert not bool(basins.latched(traj, flat)[0])

    # Both.
    both = _pulse_forcing(2.0, dt, 200, cm.event_sigma(3.6),
                          precondition=0.93)
    traj = basins.simulate([basins.COLD_STATE], both[:, None], None, dt,
                           rates=0.5)
    assert bool(basins.latched(traj, both[:, None])[0])


def test_a_latched_fast_basin_tips_a_slow_one():
    """Rectification across timescales.

    A brief event cannot reach a slow basin directly — that earlier
    finding stands. It reaches it through a fast basin that latches
    and then applies its coupling permanently, turning a pulse into
    a step.
    """
    dt = 0.02
    years = 3000
    steps = int(years / dt)
    axis = np.arange(steps) * dt
    sigma = cm.event_sigma(3.6)
    amplitude = cm.subsurface_anomaly_C(sigma) * 0.19 / sigma
    pulse = np.interp(axis, np.arange(int(axis[-1]) + 2),
                      cm.event_pulse(int(axis[-1]) + 2, sigma,
                                     start_year=100))

    forcing = np.column_stack([
        basins.CRITICAL_FORCING * 0.93 + amplitude * pulse,
        np.full(steps, basins.CRITICAL_FORCING * 0.88)])
    rates = np.array([1 / 2.0, 1 / 40.0])
    D = np.array([[0.0, 0.0], [0.075, 0.0]])

    coupled = basins.tipping_steps(
        basins.simulate([-1.0, -1.0], forcing, D, dt, rates=rates))
    isolated = basins.tipping_steps(
        basins.simulate([-1.0, -1.0], forcing, None, dt, rates=rates))

    assert isolated[1] == -1          # slow basin holds on its own
    assert coupled[1] >= 0            # and crosses once the fast one latches
    assert coupled[1] * dt - 100 > 100  # centuries later


def test_the_slow_basin_still_ignores_the_event_directly():
    """The earlier result, retained. Only the framing was wrong."""
    dt = 0.02
    forcing = _pulse_forcing(40.0, dt, 3000, cm.event_sigma(3.6))
    traj = basins.simulate([basins.COLD_STATE], forcing[:, None], None, dt,
                           rates=1.0 / 40.0)
    assert basins.tipping_steps(traj)[0] == -1


# --- commitment, slowing down, reversibility -------------------------


def test_recovery_rate_matches_the_analytic_endpoints():
    """2 at zero forcing, exactly 0 at the saddle-node."""
    assert abs(basins.recovery_rate(0.0) - 2.0) < 1e-9
    assert basins.recovery_rate(basins.CRITICAL_FORCING) == 0.0


def test_recovery_rate_falls_monotonically_toward_the_threshold():
    rates = [basins.recovery_rate(basins.CRITICAL_FORCING * f)
             for f in (0.0, 0.5, 0.9, 0.99, 0.999)]
    assert all(np.diff(rates) < 0)


def test_relaxation_time_diverges_at_the_threshold():
    near = basins.relaxation_time(basins.CRITICAL_FORCING * 0.999)
    rest = basins.relaxation_time(0.0)
    assert near > 30 * rest
    assert not np.isfinite(basins.relaxation_time(basins.CRITICAL_FORCING))


def test_the_state_barely_moves_before_it_jumps():
    """Commitment is invisible: the equilibrium drifts to -1/sqrt(3),
    21% of the full transition, and then jumps the rest."""
    drift = abs(basins.equilibria(basins.CRITICAL_FORCING * 0.9999)[0]
                - basins.COLD_STATE)
    assert 0.40 < drift < 0.45          # of a 2.0-unit transition
    assert abs(drift / 2.0 - 0.21) < 0.02


def test_commitment_lag_grows_with_relaxation_time():
    """One crossing, three rate constants, three very different
    times until anything is visible."""
    dt = 0.1
    steps = int(6000 / dt)
    axis = np.arange(steps) * dt
    ramp = basins.CRITICAL_FORCING * (0.80 + 0.20 * axis / 500.0)

    lags = []
    for relaxation in (2.0, 30.0, 1000.0):
        traj = basins.simulate([basins.COLD_STATE], ramp[:, None], None, dt,
                               rates=1.0 / relaxation)
        lags.append(basins.commitment_lag(traj, ramp[:, None], dt)[0])

    assert all(np.isfinite(lags))
    assert lags[0] < lags[1] < lags[2]
    assert lags[2] > 1000            # ice-sheet-like: millennia


def test_early_warning_indicators_rise_toward_the_threshold():
    """Variance and lag-1 autocorrelation both increase, and the
    autocorrelation matches exp(-lambda).

    Measured at fixed forcing. An earlier version detrended a ramped
    run with a filter as wide as the measurement window and got
    indicators that fell — this test exists so that cannot recur
    silently.
    """
    dt, rate, steps = 0.05, 1 / 30.0, 120000
    rng = np.random.default_rng(5)

    variances, autocorrs = [], []
    for level in (0.5, 0.9, 0.995):
        c = basins.CRITICAL_FORCING * level
        forcing = c + 0.010 * rng.standard_normal(steps)
        traj = basins.simulate([basins.equilibria(c)[0]], forcing[:, None],
                               None, dt, rates=rate)[:, 0]
        annual = traj[steps // 4:][::int(1.0 / dt)]
        annual = annual - annual.mean()
        variances.append(annual.var())
        measured = float(annual[:-1] @ annual[1:] / (annual @ annual))
        autocorrs.append(measured)
        predicted = np.exp(-basins.recovery_rate(c, rate=rate))
        assert abs(measured - predicted) < 0.01, (level, measured, predicted)

    assert variances[0] < variances[1] < variances[2]
    assert autocorrs[0] < autocorrs[1] < autocorrs[2]
    assert variances[2] > 5 * variances[0]


def test_reversibility_window_shrinks_as_overshoot_grows():
    """Escape near a saddle-node slows as the overshoot shrinks, so a
    small overshoot leaves a long window."""
    dt = 0.005
    safe = basins.CRITICAL_FORCING * 0.80
    start = basins.equilibria(safe)[0]

    def window(overshoot):
        for delay in np.linspace(0.0, 40.0, 40):
            n = int((delay + 60.0) / dt)
            f = np.full(n, basins.CRITICAL_FORCING * overshoot)
            f[int(delay / dt):] = safe
            traj = basins.simulate([start], f[:, None], None, dt, rates=1.0)
            if traj[-1, 0] > 0:
                return delay
        return np.inf

    assert window(1.02) > window(1.20) > window(1.80)


def test_restoring_forcing_too_late_does_not_help():
    """Past the window, returning to the original forcing leaves the
    system in the new state."""
    dt = 0.005
    safe = basins.CRITICAL_FORCING * 0.80
    n = int(80.0 / dt)
    f = np.full(n, basins.CRITICAL_FORCING * 1.20)
    f[int(20.0 / dt):] = safe
    traj = basins.simulate([basins.equilibria(safe)[0]], f[:, None], None,
                           dt, rates=1.0)
    assert traj[-1, 0] > 0


# --- trend against variability ---------------------------------------


def test_emergence_is_later_at_depth_than_at_the_surface():
    """The qualification the 2025 State of the Climate report forces.

    Ranking at the surface no longer depends on ENSO phase because
    the trend passed one sigma of ENSO within about a decade. At the
    model's reference depth, where ENSO delivers ~0.33 C per sigma
    against a 0.0075 C/yr trend, it has not.
    """
    depth_sigma = float(cm.subsurface_anomaly_C(1.0))
    depth_trend = 0.0075
    surface_sigma, surface_trend = 0.10, 0.020

    surface_emergence = surface_sigma / surface_trend
    depth_emergence = depth_sigma / depth_trend

    assert surface_emergence < 10
    assert depth_emergence > 40
    assert depth_emergence > 4 * surface_emergence


def test_one_sigma_at_depth_comes_from_the_composite_scaling():
    """0.5 C at a 1.5-sigma composite implies 1/3 C per sigma."""
    assert abs(cm.subsurface_anomaly_C(1.0)
               - cm.ENSO_SUBSURFACE_AMPLITUDE_C
               / cm.COMPOSITE_EVENT_SIGMA) < 1e-12
    assert 0.30 < cm.subsurface_anomaly_C(1.0) < 0.36


def test_a_warming_streak_is_essentially_impossible_without_trend():
    """"The last eleven years were the eleven warmest" is a strong
    statement about trend against noise."""
    rng = np.random.default_rng(3)
    length, streak, trials = 176, 11, 2000

    def probability(ratio):
        series = (ratio * np.arange(length)[None, :]
                  + rng.standard_normal((trials, length)))
        order = np.argsort(np.argsort(series, axis=1), axis=1)
        return float(np.mean((order >= length - streak)[:, -streak:]
                             .all(axis=1)))

    assert probability(0.0) == 0.0
    assert probability(0.5) > probability(0.2) > probability(0.0)


def test_committed_sea_level_dwarfs_what_has_been_realised():
    """Section 13's commitment lag, in an observable with a rate."""
    import json
    with open(ROOT / 'Model' / 'parameters.json') as handle:
        params = json.load(handle)

    realised_m = params['sea_level_mm_above_1993'] / 1000.0
    committed_m = params['committed_west_antarctic_sea_level_m']
    rate = (params['sea_level_thermal_mm_per_year']
            + params['sea_level_ice_mm_per_year'])

    assert committed_m / realised_m > 30
    assert rate == 3.6
    assert params['sea_level_ice_mm_per_year'] > \
        params['sea_level_thermal_mm_per_year']
    assert committed_m * 1000 / rate > 1000        # centuries at this rate


def test_arctic_multiyear_ice_loss_is_a_memory_loss():
    """Recorded as the same structure in another system: an
    integrator stripped of its integration window. Outside the
    model's domain and not evidence for it."""
    import json
    with open(ROOT / 'Model' / 'parameters.json') as handle:
        params = json.load(handle)

    before = params['arctic_multiyear_ice_km2_1980s']
    after = params['arctic_multiyear_ice_km2_2025']
    assert 1 - after / before > 0.9


if __name__ == '__main__':
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f'  pass  {name}')
        except AssertionError as exc:
            failures.append((name, exc))
            print(f'  FAIL  {name}: {exc}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    sys.exit(1 if failures else 0)
