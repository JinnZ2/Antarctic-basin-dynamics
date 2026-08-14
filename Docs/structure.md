# Structural changes

Three things the model previously collapsed into scalars have
been given structure: space, age, and the basin itself.

The August 2026 literature review
(`literature.md`) recorded these as the highest-value gaps
it could not close, on the grounds that faking them with
scalars would be worse than leaving them visible. This
document is what closing them produced, including the
results that were not expected and the places where adding
structure made the model *less* certain rather than more.

A fourth followed: ENSO, which gave the model its first
oscillation and so its first chance to test the claim
`geometry.md` is built on.

Modules: `Model/spatial.py`, `Model/population.py`,
`Model/basins.py`, `Model/climate_modes.py`.
Simulations: `Sims/structural_v4.py`,
`Sims/forcing_isolation.py`, `Sims/enso_coupling.py`.
Checks: `tests/test_structure.py`.

⸻

## 1. Space

**What replaced what.** `patch_autocorrelation`, a single
number in [0, 1], is now a lattice: rows of depth or
latitude bands, columns of longitude wrapping at the date
line, partitioned into six circumpolar sectors. Habitat
continuity is measured as the fraction of the domain in the
largest connected component.

### The threshold is now emergent

`core.percolation_connectivity()` produces a sharp
connectivity collapse by writing a sigmoid. That is
assuming the answer. The lattice is driven by bond
occupation declining *linearly* and the sharp transition
falls out of the geometry.

This is checkable rather than decorative, because square
lattice bond percolation has a known threshold of 0.5. The
implementation recovers 0.50 on a 60x60 square lattice
(`tests/test_structure.py`), which is what licenses
believing the numbers it produces on geometries with no
analytic answer.

### The model's own geometry raises the threshold

The Southern Ocean habitat band is thin in depth and long
in longitude. On an 8x60 strip the measured threshold is
about 0.55, not 0.50.

That is a result, not an error. A habitat band that is
narrow in one dimension loses continuity at a *higher* bond
probability than a compact region of the same area — there
are fewer alternative paths around any given break. The
scalar model had no way to express that the shape of the
habitat matters independently of how much of it there is.

### Heterogeneity smears the threshold into a staircase

Feeding per-sector fragmentation rates in produces
something neither earlier form predicted: not one sharp
collapse and not a smooth decay, but a staged decline. Each
sector crosses its own threshold at its own time, and the
circumpolar curve is the sum of several separate
transitions.

The habitat does not fail all at once. It retreats to the
sectors that hold together longest.

### Redistribution is not decline

`redistribution_index()` reports gross sector change
divided by net circumpolar change. It is 1.0 when every
sector moves the same way, and large when they move in
opposite directions.

On the default trends it reaches about 2.6. The circumpolar
mean shows a 13% decline while individual sectors have
moved up to 72% in opposing directions. A scalar model run
on the same inputs would report a modest, survivable
change, and would be describing a state that no sector is
actually in.

This is the diagnostic to check first. If it sits near 1.0
on a given run, the spatial layer is not earning its cost
and a scalar would do.

### What is still assumed

Sector trend *signs* are grounded — krill decline in the
Atlantic-Bellingshausen sector contrasts with increase in
the Ross-Pacific sector, and in 2023 the Amundsen was the
only region substantially above normal in sea ice while the
outer Weddell, Ross and entire Bellingshausen showed
extreme lows. The *magnitudes* in
`SECTOR_SUPPLY_TREND` and `SECTOR_FRAGMENTATION` are
illustrative. They are shaped to make divergence visible on
a decadal plot, not measured.

The lattice dimensions are also a choice. Threshold values
depend on them.

⸻

## 2. Age

**What replaced what.** A scalar abundance with a scalar
recruitment rate is now an age-classed projection matrix
running from age 0 to a bounded maximum, with reproduction
beginning at maturity.

Age classes rather than stage classes, deliberately.
Fixed-duration stage models blur a sharp maturity age, and
maturity age is one of the few quantities actually measured
for this animal. It should not be smeared for notational
convenience. Stage labels survive only as a reporting
aggregation.

### The demography is externally sourced

Growth of about 1 cm per year, sexual maturity at
156 ± 22 years, 200–324 pups per pregnancy depending on
maternal size, and a radiocarbon lifespan estimate of
392 ± 120 years. These come from separate studies and
nothing forces them to be mutually consistent.

Assembled without tuning, they give an annual growth rate
of **0.9995** — within 0.05% of exact replacement.

That is a weak check, not a validation: annual survival by
age is *not* measured for this species and is the
heuristic input doing most of the work. But an arbitrary
schedule would not land there, and it is worth knowing that
the independently sourced numbers are close to consistent.

The baseline is then calibrated to exactly replacement so
forcing experiments read against a flat reference. That is
a convention for legibility, not a claim about the real
population.

### Ecological memory is not lifespan

This is the substantive result.

`geometry.md` operationalised ecological memory as the
lifespan of the dominant long-lived species. The projection
matrix gives three quantities where there was one:

| Quantity | Baseline | At +6 °C |
|---|---|---|
| Lifespan (the old proxy) | 300 yr | 173 yr |
| Generation time | 206 yr | 123 yr |
| Damping timescale (memory) | 498 yr | 327 yr |

The damping timescale — how long a perturbation takes to
decay to a tenth of its size — exceeds lifespan by about
66% at baseline. The old proxy understated the integration
window substantially, and the three quantities do not
track each other under warming.

There is a fourth number worth having. The subdominant
eigenvalue is complex, so perturbations do not merely
decay, they *ring*, with a period of about 209 years —
essentially the generation time. Cohort echo. A single bad
recruitment year is still detectable in the age structure
two centuries later, and returns as a weakened echo after
that.

None of this is visible to a scalar, and none of it is
lifespan.

### Warming raises the growth rate

Compressing the age axis while holding annual vital rates
fixed makes the population turn over faster with the same
lifetime output. Growth rate rises slightly.

This is not a claim that warming is good for the animal.
It is the honest consequence of the demographic channel
taken alone, and the reason it matters is that it isolates
what warming actually costs here: **the system loses its
memory before it loses its viability.** Memory falls by a
third while growth rate ticks up.

`interpretation_notes.md` already described this
qualitatively — cold systems as heavy flywheels, warming
reducing rotational inertia, the system becoming more
responsive and more prone to overshoot. The demographic
layer turns that analogy into two numbers that move in
opposite directions.

Viability declines only once the energy channel is
connected, which is where the isolation experiments come
in.

### What is still assumed

Annual survival by stage is the largest unmeasured input
and the one the results are most sensitive to. Breeding
interval is a placeholder. The Allee mechanism is
implemented and **off by default**: mate limitation is the
most commonly reported Allee mechanism in general, but
detection in marine fishes carries known analytical biases,
some depleted shark populations recovered faster than
demographic models predicted, and mating-system variation
appears unlikely to be a major determinant of extinction
vulnerability in elasmobranchs. Turning it on states a
hypothesis; it does not correct an error.

⸻

## 3. Basins

**What replaced what.** One basin depth, clipped at zero,
is now several coupled double-well potentials with
destabilising interactions.

Each basin follows

    dx/dt = x - x³ + c + Σⱼ dᵢⱼ (xⱼ + 1) / 2

where x = −1 is the cold structured state and x = +1 the
reorganised one. The saddle-node sits at
|c| = 2/(3√3) ≈ 0.3849, which is the analytic anchor the
implementation is checked against.

### Coupling enters as forcing

A neighbour in the cold state contributes nothing. A
neighbour that has tipped contributes its full coupling
strength. So a tipped basin raises the effective forcing on
its neighbours, which can push them past a threshold they
would not have crossed alone.

This is the minimum structure that can represent what the
tipping-point literature describes: the ice sheet as
several interacting systems across drainage basins rather
than one element, with most known interactions
destabilising.

### Cascades are attributable

`cascade_attribution()` runs the system twice under
identical forcing, once coupled and once not. A basin that
tips only in the coupled run was tipped by a neighbour. On
the default configuration one sector crosses its own
threshold and two more follow — a chain three deep that
then halts where the remaining margin is wide enough to
absorb one tipped neighbour.

Two of six basins reorganise for reasons entirely absent
from their own forcing. A single-basin model cannot
produce that outcome at any parameter value, which is the
whole argument for the change.

`susceptibility()` reports the margin each basin has now
against the margin it would have if every neighbour tipped.
A basin with a positive first value and a negative second
is being held up by its neighbours staying put. It looks
safe and is not.

### Negative depth means something now

`additions.md` item 5 asked for basin depth to be allowed
to go negative rather than clipped. With a real potential
this happens naturally: once forcing passes the
saddle-node the well does not exist, and the reported depth
becomes the distance past the bifurcation.

Negative depth is not a bad energy budget. It is the
absence of an attractor.

Depth must be read off the forcing each basin *actually*
experiences, external plus the contribution from already
tipped neighbours. Computing it from the external term
alone understates the damage during a cascade, because the
neighbour contribution is precisely what eats the remaining
barrier.

### What is still assumed

Coupling strengths between real Antarctic tipping systems
are not established. The literature supports the *sign* —
most interactions are destabilising — not the magnitude.
`ring_coupling()` is uniform and symmetric, which no real
system is; it is a placeholder for a matrix nobody has
measured. Sweep the strength rather than trusting 0.06.

The mapping from a normalised double well to a physical
sector is qualitative throughout. Model time is not years.

⸻

## 4. What the structure made possible

### Forcing isolation, and why it needed the structure first

`additions.md` item 8 asked for isolated forcing
experiments and interaction surfaces. Attempting it on a
static viability index reveals why it had to wait.

An index built from supply, demand and reach is
*multiplicatively separable* in warming and fragmentation.
Its interaction term is exactly zero — not small, zero, by
construction, for any input values. That experiment would
have produced a clean surface meaning nothing at all.

Age structure breaks the separability for a statable
reason: **warming shortens generation time, so a given
supply shortfall is paid off over fewer years and costs
more per year.** The interaction is a property of the
demography, and it did not exist in the model before.

Running it (`Sims/forcing_isolation.py`), in percentage
population change per century:

| Driver | Full ramp |
|---|---|
| Thermal (all channels) | −95% |
| — of which transfer efficiency | −91% |
| — of which microbial differential | −30% |
| Fragmentation | −65% |
| Combined | −98.5% |
| Additive null | −98.4% |

Two things fall out.

**Transfer efficiency dominates the microbial term.**
Efficiency enters once per trophic level; interception
enters once. Before efficiency was made dynamic in the 2026
review, the microbial differential was carrying the entire
trophic mismatch mechanism by itself, and it is the weaker
of the two by a wide margin.

**The interaction peaks in the middle.** Departure from
additivity is strongest near +2–3 °C at low connectivity,
and fades at both ends because the outcome saturates. That
places the maximum compounding effect at roughly the
model's default `warming_delta_C` of 2.0 — the region where
the drivers are individually survivable and jointly are
not.

### Where the model got less certain

Adding structure did not only sharpen things.

The spatial layer introduced lattice dimensions, six
sector trend rates and six fragmentation rates, none of
them measured. The demographic layer introduced a survival
schedule that does not exist in the literature for this
species. The basin layer introduced a coupling matrix whose
sign is supported and whose magnitude is invented.

Three scalars were replaced by roughly twenty parameters,
of which a minority are grounded. The model now represents
behaviours it could not represent before and it makes more
assumptions to do so. `parameters.json` marks the
unmeasured ones under `_heuristic_parameters`.

That trade is worth making for a geometric model, whose
purpose is to show what shapes are possible rather than to
predict values. It would not obviously be worth making for
a predictive one.

⸻

## 4b. Interannual variability: ENSO

**What replaced what.** Nothing oscillated. Forcing was a
trend, optionally accelerating. ENSO and SAM now supply the
high-frequency component (`Model/climate_modes.py`,
`Sims/enso_coupling.py`, `additions.md` item 10).

### It uses the depth dimension the lattice already had

During El Niño a weaker Amundsen Sea Low weakens coastal
easterlies, which reduces on-shelf Ekman transport of cold
surface water and lets warm Circumpolar Deep Water onto the
shelf. Subsurface warming from 150 m to the shelf bottom
reaches 0.5 °C. The surface response runs the other way.

The model's baseline is 1.3 °C at 490 m — inside that band.
So a single El Niño delivers roughly **a quarter of the
default 2.0 °C `warming_delta_C`** to the exact depth the
reference organism occupies, and then removes it again.

The sign reversal is the structural point. A model carrying
one temperature per sector must pick a sign, and either
choice is wrong for half the water column. The lattice rows
were depth bands that nothing had used; now they carry a
profile.

### The assumed sector trends were a standing El Niño

`spatial.py` carried monotone per-sector supply trends with
grounded signs and invented magnitudes. The Antarctic
Dipole is what they were imitating: a seesaw in which El
Niño raises sea ice in the Ross sector and lowers it in the
Bellingshausen and northern Weddell.

Both patterns were written from the same reported regional
contrasts, so their agreement is a consistency check and
not an independent discovery. What the dipole genuinely
adds is a mechanism, and the sign changes that a monotone
trend cannot produce.

### The memory claim, tested at last

`geometry.md` asserts that compressing ecological memory
lets high-frequency environmental variation propagate
further into the food web. Nothing in the model oscillated,
so the claim had never been checkable.

Adults integrate every recruitment year between maturity
and death, which makes them a low-pass filter with a cutoff
set by that span — and warming shortens the span.
Driving recruitment with white noise and measuring the
adult response:

| | Baseline | +6 °C |
|---|---|---|
| ENSO-band gain (2–7 yr) | 0.00086 | 0.00149 |
| Attenuation | 1156× | 669× |

The claim holds — about 73% more ENSO-band variance gets
through. But the qualitative version omits the scale. A
73% increase on a thousandfold attenuation is still a
thousandfold attenuation. The slow integrator does not stop
being slow; it stops being quite as slow.

The gain spectrum also shows a resonance bump near 200
years, which is the cohort echo from `structure.md`
section 2 appearing in a second, independent measurement.

### The basin result is negative, and more useful

A basin integrates forcing over its own relaxation time,
and ENSO averages to zero over any span longer than a
decade. Sweeping relaxation time:

| Basin relaxation | Zero-mean ENSO | Persistent El Niño |
|---|---|---|
| 0.5 yr | 100% tip | 100% |
| 2 yr | 0% | 100% |
| 10 yr | 0% | 88% |
| 30 yr and slower | 0% | 0% |

Only basins about as fast as ENSO itself can be tipped by
it. At decadal or centennial relaxation an excursion **more
than three times the remaining margin** never crosses the
threshold.

Offsetting toward a persistent El Niño extends the
vulnerable range roughly tenfold — but that is a shift in
the *mean*, not the variability, doing the work.

The conclusion is a constraint on which literature matters.
For slow systems, ENSO variance is irrelevant. What matters
is anything that shifts ENSO's mean or tail: more frequent
extreme Eastern Pacific events, or the teleconnection
weakening from 0.72 to 0.21 after 2002.

It also corrects `interpretation_notes.md`. "Individual bad
years matter more as the margin narrows" holds for the fast
stochastic energy balance it was written about. It is false
for slow basins, which cannot see individual bad years.

### What is still assumed

The ENSO oscillator's damping sets bandwidth and is
tuned by eye to the 2–7 year description. Sector patterns
have grounded signs and shaped magnitudes, as before. The
depth transition at 150 m is empirical but the sharpness of
the profile is a stand-in for Ekman and bathymetric detail.

Two mappings are invented and both matter: degrees of
warming per unit basin forcing, and basin model time per
year. The basin conclusions are conclusions about a
*ratio*, which is why that ratio is swept rather than
chosen.

The teleconnection shift is modelled as a step because that
is how it is reported. The real transition was presumably
not instantaneous.

Amplitude change under warming defaults to zero. This is
the honest default: CMIP6 spans both signs with an
ensemble mean near zero, so the model does not pick one.

⸻

## 5. Still not represented

- Behavioural adaptation and range shift. The spatial
  layer has patches but no agents moving between them,
  which was the more ambitious reading of
  `additions.md` item 7.
- Evolutionary response.
- Two-way coupling between layers. Supply flows spatial →
  demographic → basin, but a reorganised basin does not
  feed back into habitat structure. Real cascades run in
  both directions.
- Dynamic trophic depth (`additions.md` item 3, second
  half). The literature supports warming-driven trophic
  compression but gives no defensible coefficient.
- Seasonal and stochastic forcing components
  (`additions.md` item 10). Only the long-term trend term
  is non-linear.
- Ice-albedo and methane feedbacks. Forcing is still
  externally imposed.
