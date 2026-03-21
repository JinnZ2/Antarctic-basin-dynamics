# CLAUDE.md — Antarctic Basin Dynamics

## Project Overview

A geometric dynamical modeling framework for examining Antarctic basin ecosystem stability under compound environmental forcing (warming + fragmentation). This is a **geometric model**, not a predictive model — it makes structural relationships visible rather than forecasting specific outcomes.

The framework examines how temperature forcing interacts with trophic structure, spatial continuity, and organism time constants to create regime shifts and stability thresholds in deep-ocean Antarctic ecosystems.

Licensed under CC0 1.0 Universal (public domain).

## Repository Structure

```
Antarctic-basin-dynamics/
├── Model/                  # Core model infrastructure
│   ├── core.py             # Shared functions: metabolic scaling, lifespan, energy,
│   │                       #   patch viability, percolation_decay, quadratic_warming,
│   │                       #   load_parameters, PHI constant
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
│   ├── shadow_percolation.py       # Shadow geometry with percolation decay
│   ├── shadow_phi_modulated.py     # Shadow geometry with phi-modulated oscillations
│   └── calibration_pipeline.py     # Empirical data integration scaffold (CCAMLR, OBIS, etc.)
├── Docs/                   # Documentation
│   ├── geometry.md         # Core geometric framing (basin depth, frequency, mismatch)
│   ├── interpretation_notes.md  # Model philosophy + AI guidance for interpreting outputs
│   └── variables.md        # Parameter definitions with empirical ranges
├── tests/
│   └── smoke_test.py       # Core function tests + import validation
├── output/                 # Simulation output files (.png, .csv)
├── additions.md            # Proposed model extensions (10 directions)
├── requirements.txt        # Python dependencies with version pins
├── CONTRIBUTING.md         # Contribution guidelines
├── README.md
└── LICENSE
```

## Tech Stack

- **Language:** Python 3
- **Dependencies:** `numpy`, `matplotlib`, `pandas` (calibration only) — see `requirements.txt`
- **No CI/CD** — no linting or build pipeline

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simulation
python Sims/integrative_sim.py

# Run smoke tests
python tests/smoke_test.py
```

## Running Simulations

Each simulation in `Sims/` is a standalone script. All simulations use `sys.path` to import from `Model/core.py`, so they work from any working directory. Output `.png` files are saved to `output/`.

## Key Concepts

- **Differential Q10 response** is structurally central: microbial Q10 (3.5) > midtrophic Q10 (3.0) > apex Q10 (2.5). This differential drives energy supply deficits under warming.
- **Shadow geometry**: parallel basin architecture where a shadow basin grows as the primary connectivity decays, representing alternative stability configurations.
- **Percolation dynamics**: sigmoidal connectivity transitions model habitat fragmentation thresholds.
- **Two life-history strategies**: "slow integrator" (long-lived, high memory) vs. "fast cycler" (short-lived, responsive) — compared throughout.

## Code Conventions

- **Shared logic** belongs in `Model/core.py` — import `percolation_decay`, `metabolic_multiplier`, `quadratic_warming`, `PHI`, etc. from there rather than redefining.
- **Parameters** should be loaded via `load_parameters()` from `Model/core.py` rather than hardcoded.
- **Imports** use `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'Model'))` for portable imports.
- **Naming:** snake_case for variables/functions, UPPERCASE for constants (Q10, PHI).
- **Parameters:** Semantic prefixes (`baseline_temp_C`, `warming_delta_C`, `patch_autocorrelation`).
- **Section dividers:** `# ── Section Name ──────` comments.
- **Configuration:** `@dataclass`-based config objects in advanced simulations; `parameters.json` for baseline values.
- **Population bounds:** Explicit clamping with `min`/`max` or `np.clip`.
- **Visualization:** Dark backgrounds (#0a0a0a), consistent color coding (blue=slow, orange/green=fast), multi-panel gridspec layouts.
- **Outputs** go to `output/` directory, not the repo root.

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
