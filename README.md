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
| `Model/core.py` | Model primitives |
| `Model/spatial.py` | Circumpolar habitat lattice, percolation |
| `Model/population.py` | Age-structured demography |
| `Model/basins.py` | Coupled basins, cascading destabilisation |
| `Model/climate_modes.py` | ENSO and SAM, depth-resolved teleconnection |
| `Model/parameters.json` | Parameter values and provenance |
| `Docs/geometry.md` | Conceptual framing |
| `Docs/variables.md` | Variable definitions and known weak points |
| `Docs/interpretation_notes.md` | How to read the outputs |
| `Docs/literature.md` | Empirical basis and revision history |
| `Docs/structure.md` | What the structural layers changed |
| `additions.md` | Candidate extensions, with implementation status |
| `Sims/` | Simulation scripts |
| `tests/` | Checks against analytic values |

## Where the model stands

Reviewed against the literature in **August 2026**;
structural layers added immediately after. Every
parameter's source, and every change deliberately
*not* made, is in `Docs/literature.md`.

The literature review made trophic transfer efficiency
dynamic under warming, added an oxygen constraint and
an accelerating forcing option, and weakened the
rate-of-living assumption on lifespan. Those additions
each default to a neutral value reproducing the model's
prior behaviour.

The structural update then replaced three scalars with
structure — space, age, and the basin itself. Eight of
the ten items in `additions.md` are now implemented.
Three results worth knowing before reading any output:

- **Ecological memory is not lifespan.** The damping
  timescale from the projection matrix is ~498 years
  against a 300-year lifespan, and perturbations ring
  at roughly the generation time rather than simply
  decaying.
- **Warming costs memory before viability.** Taken
  alone, age compression slightly *raises* population
  growth rate while memory falls by a third.
- **Thresholds are emergent and plural.** Connectivity
  collapse comes out of lattice geometry rather than an
  assumed sigmoid, and coupled basins let a sector
  reorganise for reasons absent from its own forcing.

ENSO was then added, giving the model its first
oscillation and two more results:

- **The ENSO signal reverses sign with depth**, and the
  model's 490 m baseline sits inside the band where El Niño
  warms the shelf by up to 0.5 °C — a quarter of the default
  total warming, delivered and withdrawn every few years.
- **Slow basins cannot be tipped by variability.** They
  integrate over their own relaxation time, and ENSO
  averages to zero across it. An excursion three times the
  remaining margin never crosses. Only the *mean* moves a
  slow basin, which corrects "individual bad years matter
  more as the margin narrows" — true for the fast stochastic
  balance it was written about, false here.

`Docs/structure.md` covers all of these, including what
became *less* certain: roughly twenty parameters now
stand where three scalars did, and a minority are
grounded. `parameters.json` marks the rest under
`_heuristic_parameters`. Sweep them; do not trust them.

## Running

```
python Sims/structural_v4.py       # the three structural layers
python Sims/enso_coupling.py       # ENSO: depth, dipole, filtering, tipping
python Sims/forcing_isolation.py   # isolated drivers, interaction surface
python Sims/lit_update_2026.py     # prior vs revised parameterisation
python tests/test_structure.py     # checks (also runs under pytest)
```

Checks are against analytic values where they exist:
square-lattice bond percolation at 0.5, the saddle-node
of dx/dt = x − x³ + c at 2/(3√3), the potential barrier
at zero forcing of exactly 0.25.

## Requirements

`numpy`, `matplotlib`.
