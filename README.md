# Antarctic-basin-dynamics

A geometric model of energy basin stability under
compound forcing in Antarctic marine systems.

It is not a predictive model. It makes structural
relationships visible — where thresholds sit, how
variables couple, what shape a collapse takes.
See `Docs/interpretation_notes.md` for what that
distinction means and what the outputs do and do not
support.

## Layout

| Path | Contents |
|---|---|
| `Model/core.py` | Model functions |
| `Model/parameters.json` | Parameter values and provenance |
| `Docs/geometry.md` | Conceptual framing |
| `Docs/variables.md` | Variable definitions and known weak points |
| `Docs/interpretation_notes.md` | How to read the outputs |
| `Docs/literature.md` | Empirical basis and revision history |
| `additions.md` | Candidate extensions, with implementation status |
| `Sims/` | Simulation scripts |

## Literature status

Last reviewed **August 2026**. Every parameter's
source, and every change deliberately *not* made, is
recorded in `Docs/literature.md`.

The most recent review made trophic transfer
efficiency dynamic under warming, added an oxygen
constraint, added an accelerating forcing option and
a percolation-style connectivity transition, and
weakened the rate-of-living assumption on lifespan
compression. Each addition defaults to a neutral
value that reproduces the model's prior behaviour, so
existing simulations are unaffected until a mechanism
is engaged deliberately.

Four parameters are flagged as heuristic or derived
rather than measured. They are listed under
`_heuristic_parameters` and `_derived_parameters` in
`parameters.json`. Sweep them; do not trust them.

`python Sims/lit_update_2026.py` plots the prior
parameterisation against the revised one.

## Requirements

`numpy`, `matplotlib`.
