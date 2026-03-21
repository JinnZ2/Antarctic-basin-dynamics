# Antarctic Basin Dynamics

A geometric dynamical modeling framework for examining Antarctic basin ecosystem stability under compound environmental forcing.

## What This Is

This is a **geometric model** — not a predictive model. It makes structural relationships visible rather than forecasting specific outcomes. The framework examines how temperature forcing interacts with trophic structure, spatial continuity, and organism time constants to create regime shifts and stability thresholds in deep-ocean Antarctic ecosystems.

## Key Mechanisms

- **Differential Q10 response**: Microbial metabolic rates accelerate faster than apex predator rates under warming, creating energy supply deficits at higher trophic levels.
- **Shadow geometry**: Parallel basin architecture where alternative stability configurations emerge as primary connectivity decays.
- **Percolation dynamics**: Habitat connectivity remains high until a critical threshold, then collapses rapidly — modeled via sigmoidal transitions.
- **Life-history tradeoffs**: "Slow integrator" (long-lived, high ecological memory) vs. "fast cycler" (short-lived, responsive) strategies respond differently to compound forcing.

## Project Structure

```
Model/          Core functions and baseline parameters
Sims/           Standalone simulation scripts (each produces visualizations)
Docs/           Geometric framing, interpretation guidance, variable definitions
```

## Quick Start

```bash
pip install -r requirements.txt
python Sims/integrative_sim.py
```

Each script in `Sims/` is standalone and can be run directly. Outputs go to `output/`.

## Documentation

- [Geometric framing](Docs/geometry.md) — core theoretical structure
- [Interpretation notes](Docs/interpretation_notes.md) — how to read model outputs (includes AI guidance)
- [Variable definitions](Docs/variables.md) — parameter reference with empirical ranges
- [Proposed extensions](additions.md) — 10 candidate model additions

## License

[CC0 1.0 Universal](LICENSE) — public domain.
