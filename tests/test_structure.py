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
