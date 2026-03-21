# CLAUDE.md — Antarctic Basin Dynamics

## Project Overview

A geometric dynamical modeling framework for examining Antarctic basin ecosystem stability under compound environmental forcing (warming + fragmentation). This is a **geometric model**, not a predictive model — it makes structural relationships visible rather than forecasting specific outcomes.

The framework examines how temperature forcing interacts with trophic structure, spatial continuity, and organism time constants to create regime shifts and stability thresholds in deep-ocean Antarctic ecosystems.

Licensed under CC0 1.0 Universal (public domain).

## Repository Structure

```
Antarctic-basin-dynamics/
├── Model/                  # Core model infrastructure
│   ├── core.py             # Base functions (metabolic scaling, lifespan, energy, patch viability)
│   └── parameters.json     # Central configuration (10 baseline parameters)
├── Sims/                   # Simulation scripts (standalone, each produces visualizations)
│   ├── integrative_sim.py          # Master multi-strategy integration (slow vs. fast life history)
│   ├── integrative_sim_v2.py       # Percolation-based variant with mass-dependent scaling
│   ├── energy_basin.py             # Differential metabolic acceleration visualization
│   ├── trophic_mismatch.py         # Core mismatch mechanism (microbial Q10 > apex Q10)
│   ├── patch_connectivity.py       # Spatial viability surface across body mass & autocorrelation
│   ├── leverage_strategies.py      # Management strategy comparison (4 approaches)
│   ├── shadow_geometry.py          # Parallel basin architecture with nudge engine
│   ├── shadow_geometry_v2.py       # Adaptive shadow acceleration + 625-sim phase map
│   ├── responsive_curvature.py     # Adaptive curvature + survival region mapping
│   ├── V3.py                       # Refined shadow geometry with percolation decay
│   ├── V3.1.py                     # V3 variant with phi-modulated oscillations
│   └── calibration_pipeline.py     # Empirical data integration scaffold (CCAMLR, OBIS, etc.)
├── Docs/                   # Documentation
│   ├── geometry.md         # Core geometric framing (basin depth, frequency, mismatch)
│   ├── interpretation_notes.md  # Model philosophy + AI guidance for interpreting outputs
│   └── variables.md        # Parameter definitions with empirical ranges
├── additions.md            # Proposed model extensions (10 directions)
├── README.md
└── LICENSE
```

## Tech Stack

- **Language:** Python 3
- **Dependencies:** `numpy`, `matplotlib`, `pandas` (calibration only), standard library (`json`, `pathlib`, `dataclasses`)
- **No package manager** — no requirements.txt, pyproject.toml, or virtual environment config exists
- **No CI/CD** — no test suite, linting, or build pipeline

## Running Simulations

Each simulation in `Sims/` is a standalone script:

```bash
python Sims/integrative_sim.py
python Sims/shadow_geometry_v2.py
# etc.
```

Some simulations produce `.png` output files (e.g., `energy_basin_output.png`). All produce matplotlib visualizations and console output.

Simulations import from `Model/core.py` using relative paths — run from the repository root.

## Key Concepts

- **Differential Q10 response** is structurally central: microbial Q10 (3.5) > midtrophic Q10 (3.0) > apex Q10 (2.5). This differential drives energy supply deficits under warming.
- **Shadow geometry**: parallel basin architecture where a shadow basin grows as the primary connectivity decays, representing alternative stability configurations.
- **Percolation dynamics**: sigmoidal connectivity transitions model habitat fragmentation thresholds.
- **Two life-history strategies**: "slow integrator" (long-lived, high memory) vs. "fast cycler" (short-lived, responsive) — compared throughout.

## Code Conventions

- **Naming:** snake_case for variables/functions, UPPERCASE for constants (Q10, PHI)
- **Parameters:** Semantic prefixes (`baseline_temp_C`, `warming_delta_C`, `patch_autocorrelation`)
- **Section dividers:** `# ── Section Name ──────` comments
- **Configuration:** `@dataclass`-based config objects in advanced simulations; `parameters.json` for baseline values
- **Population bounds:** Explicit clamping with `min`/`max` or `np.clip`
- **Visualization:** Dark backgrounds (#0a0a0a), consistent color coding (blue=slow, orange/green=fast), multi-panel gridspec layouts

## Mathematical Patterns

- Q10 scaling: `Q10 ** (delta_T / 10)`
- Percolation decay: `1 / (1 + exp(k * (t - t_c)))`
- Logistic growth: `pop + r_rate * pop * (1 - pop/K) * dt`
- Energy interception: cascading multiplicative loss through trophic levels
- Bounded rates via `np.tanh(energy_balance)`

## Important Guidance

- Read `Docs/interpretation_notes.md` before modifying model behavior — it contains explicit notes for AI systems on how to interpret outputs.
- Diverging trophic curves indicate rate mismatch, not model error.
- Threshold crossings indicate bifurcations — these are the key structural features.
- Lifespan compression represents ecological memory reduction and is intentional.
- The `calibration_pipeline.py` is deliberately incomplete (scaffold for future data integration).
- Respect the geometric framing: do not add predictive claims or forecast-oriented language to outputs.
