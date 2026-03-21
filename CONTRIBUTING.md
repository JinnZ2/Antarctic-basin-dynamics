# Contributing

This project is released under [CC0 1.0 Universal](LICENSE) (public domain). Contributions are welcome.

## Getting Started

```bash
git clone <repo-url>
cd Antarctic-basin-dynamics
pip install -r requirements.txt
```

## Running Simulations

Each script in `Sims/` is standalone. Run from the repo root:

```bash
python Sims/integrative_sim.py
```

Outputs are saved to `output/`.

## Running Tests

```bash
python tests/smoke_test.py
```

## Project Conventions

- **Core logic** belongs in `Model/core.py` — avoid duplicating functions in simulation files.
- **Parameters** should be loaded from `Model/parameters.json` via `load_parameters()` rather than hardcoded.
- **Imports** use `sys.path` to reference `Model/` so scripts work from any directory.
- **Outputs** go to the `output/` directory, not the repo root.
- **Naming** uses snake_case for variables/functions, UPPERCASE for constants.
- **Visualization** uses dark backgrounds and consistent color coding (blue=slow, orange/green=fast).

## What to Work On

See [additions.md](additions.md) for proposed model extensions. Each can be implemented and tested independently.

The [calibration pipeline](Sims/calibration_pipeline.py) has documented data source stubs ready for implementation.

## Philosophy

This is a geometric model. Read [Docs/interpretation_notes.md](Docs/interpretation_notes.md) before modifying model behavior. Respect the geometric framing — do not add predictive claims or forecast-oriented language.
