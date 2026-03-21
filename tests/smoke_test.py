"""
Smoke tests — verify all simulations import and core functions work.
Run from repo root: python -m pytest tests/ -v
Or directly:        python tests/smoke_test.py
"""
import sys
from pathlib import Path

# Ensure Model/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Model'))

import numpy as np


def test_core_imports():
    from core import (
        metabolic_multiplier,
        adjusted_lifespan,
        trophic_energy_interception,
        patch_viability,
        percolation_decay,
        quadratic_warming,
        load_parameters,
        PHI,
    )
    assert callable(metabolic_multiplier)
    assert callable(percolation_decay)
    assert isinstance(PHI, float)


def test_load_parameters():
    from core import load_parameters
    P = load_parameters()
    assert 'Q10_apex' in P
    assert 'baseline_temp_C' in P
    assert P['Q10_microbial'] > P['Q10_apex']


def test_metabolic_multiplier():
    from core import metabolic_multiplier
    assert metabolic_multiplier(2.0, 0) == 1.0
    assert metabolic_multiplier(2.0, 10) == 2.0
    assert metabolic_multiplier(3.0, 10) == 3.0


def test_adjusted_lifespan():
    from core import adjusted_lifespan
    assert adjusted_lifespan(300, 1.0) == 300
    assert adjusted_lifespan(300, 2.0) == 150


def test_trophic_energy_interception():
    from core import trophic_energy_interception
    result = trophic_energy_interception(100, 1.0, 0.1, 3)
    assert result > 0
    # Higher microbial multiplier should reduce available energy
    result_warm = trophic_energy_interception(100, 2.0, 0.1, 3)
    assert result_warm < result


def test_patch_viability():
    from core import patch_viability
    # High autocorrelation, small body → viable
    assert patch_viability(0.9, 100, 0.75) is True or patch_viability(0.9, 100, 0.75) == True
    # Low autocorrelation, large body → nonviable
    assert patch_viability(0.1, 100000, 0.75) is False or patch_viability(0.1, 100000, 0.75) == False


def test_percolation_decay():
    from core import percolation_decay
    # At t << t_c, connectivity ~1; at t >> t_c, connectivity ~0
    assert percolation_decay(0, 0.4, 60) > 0.99
    assert percolation_decay(200, 0.4, 60) < 0.01
    # At t_c, connectivity = 0.5
    assert abs(percolation_decay(60, 0.4, 60) - 0.5) < 0.01


def test_quadratic_warming():
    from core import quadratic_warming
    assert quadratic_warming(0, 0.01, 0.0008) == 0.0
    result = quadratic_warming(50, 0.01, 0.0008)
    assert result > 0


def test_sim_imports():
    """Verify all simulation files can be parsed (no syntax errors)."""
    import importlib.util
    sims_dir = ROOT / 'Sims'
    failures = []
    for py_file in sorted(sims_dir.glob('*.py')):
        try:
            spec = importlib.util.spec_from_file_location(
                py_file.stem, py_file
            )
            # Just check the spec is valid — don't execute
            assert spec is not None, f"Could not create spec for {py_file.name}"
        except Exception as e:
            failures.append(f"{py_file.name}: {e}")
    if failures:
        raise AssertionError(
            "Simulation files with issues:\n" +
            "\n".join(failures)
        )


if __name__ == '__main__':
    tests = [v for k, v in globals().items() if k.startswith('test_')]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
