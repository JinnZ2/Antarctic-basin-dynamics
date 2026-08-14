# Variable definitions

Empirical basis and revision history for every value
here: `literature.md`. Last reviewed August 2026.

## Temperature parameters

**baseline_temp_C**
Starting water temperature in degrees Celsius.
Antarctic deep water reference: ~1.3°C at 490m depth.
Corroborated 2025: the first footage of a sleeper
shark in Antarctic waters recorded the animal at
roughly 490m in water near 1.1°C.

**warming_delta_C**
Temperature increase applied to baseline.
Range of interest: 0–6°C over decadal to centennial timescales.
Note that the default of 2.0 is not a conservative
choice — the first West Antarctic ice threshold is
placed potentially as low as 1–2°C above
pre-industrial.

## Metabolic parameters

**Q10_apex**
Metabolic rate multiplier per 10°C increase for apex predators.
Reference organism: Somniosus-class sleeper sharks.
Empirical range: 2.0–2.5. Default: 2.5.

**Q10_midtrophic**
Q10 for mid-trophic organisms: fish, cephalopods, 
large invertebrates.
Empirical range: 2.5–3.2. Default: 3.0.

**Q10_microbial**
Q10 for microbial and bacterial processes.
Empirical range: 2.8–4.0. Default: 3.5.
This value drives trophic mismatch dynamics.
Higher than apex Q10 by design — this differential
is the structurally important feature.

**baseline_lifespan_years**
Reference lifespan for proxy organism under baseline 
temperature conditions.
Greenland shark empirical estimate: 250–400 years.
Default: 300 years.
The radiocarbon estimate of 392 ± 120 years remains
the reference figure. Unchanged by the 2026 review.

## Energy parameters

**trophic_transfer_efficiency**
Fraction of energy passing between trophic levels.
Empirical range: 0.05–0.15. Default: 0.10.
Now a *baseline* rather than a constant. Seven years
of experimental warming reduced transfer efficiency
by up to 56% at +4°C, so the value is no longer held
fixed under forcing — see
`tte_warming_sensitivity_per_C` below and
`dynamic_transfer_efficiency()` in `core.py`.
The 10% figure remains a rule of thumb that masks
large variation across systems, levels, and
currencies.

## Spatial parameters

**patch_autocorrelation**
Moran's I analog. Degree to which neighboring habitat 
patches share ecological properties.
1.0 = perfect continuity.
0.0 = complete fragmentation.
Default: 0.75, representing moderate historical continuity.

**fragmentation_rate**
Rate of autocorrelation decay per time unit.
Driven by land use change, ice loss, fisheries pressure,
infrastructure introduction.
Default: 0.02 per year as a first-order approximation.

**body_mass_scaling_exponent**
Kleiber-adjacent exponent for home range scaling.
Home range ∝ body mass ^ exponent.
Empirical range: 0.7–0.8. Default: 0.75.

## Derived quantities

**metabolic_multiplier**
Q10 ^ (delta_T / 10)
Rate change factor relative to baseline temperature.

**adjusted_lifespan**
baseline_lifespan / metabolic_multiplier
First-order estimate under rate-of-living framework.
Known limitation: does not account for cold-adaptation
mechanisms that may partially decouple longevity from
metabolic rate.
This limitation now has direct evidence behind it.
Greenland shark genomics recovers expansions in
NF-κB pathway gene families alongside DNA repair and
cancer resistance signatures, and cardiac tissue
shows resilience rather than age-related
degeneration. Longevity in this lineage involves
active encoded maintenance, not only slow chemistry.
Use `maintenance_adjusted_lifespan()` with
`longevity_maintenance_decoupling` to damp the
compression. The original function is retained
unchanged.

**effective_patch_size**
autocorrelation² × minimum viable landscape
Nonlinear because connectivity loss compounds.
Below 50% of MVL: viability threshold crossed.

**energy_balance**
Normalized supply minus normalized demand at apex level.
Negative values indicate budget deficit conditions.
Stochastic variant adds prey availability noise.

⸻

# Parameters added August 2026

Each of these has a neutral value at which the model
reduces to its previous behaviour. Nothing already in
the repo changes output unless a mechanism is
deliberately engaged.

## Forcing shape

**abyssal_warming_rate_C_per_year**
Recent observed abyssal warming trend in the
Antarctic sector: 0.0075 °C/yr
(7.5 ± 0.9 m°C/yr, eastern Bellingshausen Basin,
2017/18–2023/24).

**abyssal_warming_rate_prior_C_per_year**
Longer-record trend from the same basin for
comparison: 0.0028 °C/yr (2.8 ± 0.2 m°C/yr, since
1992/95). The ratio between the two is the evidence
that forcing is accelerating.

**forcing_acceleration_C_per_year2**
Quadratic coefficient *b* in
T = T₀ + a·t + b·t².
Default 2.0e-4, derived by treating each trend above
as the instantaneous rate at its record midpoint
(≈2008.5, ≈2020.5), differencing over the 12-year
gap, and halving.
Derived, not measured. Assumes smooth acceleration
and that a single-basin trend generalises.
Neutral value: 0.0 (linear ramp).

## Trophic transfer

**tte_warming_sensitivity_per_C**
Fractional reduction in transfer efficiency per °C.
Default 0.14, a linear fit to "up to 56% at +4°C"
from a seven-year warming experiment.
The experiment was freshwater mesocosms, and "up to"
is an upper bound. Sweep rather than trust.
Neutral value: 0.0 (fixed efficiency).

**tte_floor**
Lower clamp on efficiency so it cannot reach zero or
go negative under large forcing. Default 0.01.

## Oxygen

**oxygen_baseline_saturation**
Normalized reference oxygen availability. Default 1.0.
This is a normalisation, not a concentration.

**deep_oxygen_committed_loss_fraction**
Fraction of pre-industrial oxygen content the deep
ocean is committed to lose. Default 0.10, from the
finding that >10% is lost below 2000m even under
immediate emissions cessation, with ~80% of committed
global loss occurring below that depth.
This matters here because it is where the apex proxy
lives.

**oxygen_mass_sensitivity**
Exponent scaling oxygen demand with body mass in the
metabolic index. Default 0.25.
The direction is empirical — oxygen availability and
body mass jointly modulate ectotherm warming
responses, so the constraint is not mass-neutral.
The exponent is not empirical. Heuristic.
Neutral value: 0.0 (mass-neutral).

## Connectivity

**connectivity_threshold_time_years**
Inflection time t_c for sigmoidal connectivity
collapse. Default 7.0, anchored on the September 2016
Antarctic sea ice regime break and the February 2023
record low seven years later.

**connectivity_transition_steepness**
Steepness k of the same transition. Default 0.6.
The shape is assumed, not observed. Heuristic.

**connectivity_mass_sensitivity**
Multiplier α on the body-mass exponent in
`mass_dependent_connectivity()`. Default 1.0, which
with a 0.5 scaling reproduces the symmetric ac**0.5
used previously. Values above 1.0 penalise
large-bodied strategies more sharply.

## Longevity

**longevity_maintenance_decoupling**
Degree to which longevity is decoupled from metabolic
rate by active maintenance, in [0, 1].
0.0 → identical to rate-of-living.
1.0 → lifespan independent of metabolic rate.
Default 0.5.
No study gives a compression coefficient for this
lineage. The mid-point is chosen to make the
uncertainty visible rather than bury it in a harder
assumption. Heuristic.

⸻

# Parameters added in the structural update

Full rationale and results: `structure.md`.

## Demography (`Model/population.py`)

**maturity_age_years**
Age of first reproduction. Default 150.
Empirical: 156 ± 22 years.

**max_age_years**
Working maximum lifespan. Default 300.
Empirical: 392 ± 120 years reported; 300 is the
conservative working value carried from
`baseline_lifespan_years`.

**pups_per_pregnancy**
Default 262, the midpoint of an empirical 200–324
range that varies with maternal size.

**breeding_interval_years**
Years between pregnancies. Default 10.
Not known for this species. Placeholder.

**stage_age_bounds**
Age boundaries of the four reporting stages:
[0, 15, 75, 150]. Only the 150 boundary is empirical;
the internal splits are a reporting convenience.

**stage_annual_survival**
Annual survival within each stage:
[0.85, 0.95, 0.98, 0.99].
No survival schedule exists for an animal that matures
at 150 years. This is the standard shape for a large
slow elasmobranch. Heuristic, and the input the
demographic results are most sensitive to.

**allee_critical_density**
Adult density below which recruitment is penalised.
Default 0.0, meaning **off**.
Evidence in elasmobranchs is mixed: detection of Allee
effects in marine fishes carries known analytical
biases, some depleted shark populations recovered
faster than demographic models predicted, and
mating-system variation appears unlikely to be a major
determinant of extinction vulnerability. Turning it on
states a hypothesis rather than correcting an error.

Note: assembled from the empirical values above
without tuning, the schedule gives an annual growth
rate of 0.9995 — within 0.05% of replacement. A weak
consistency check, not a validation, since survival is
the unmeasured input.

## Spatial lattice (`Model/spatial.py`)

**lattice_rows**, **lattice_cols**
Habitat lattice dimensions, default 8 × 60. Rows are
depth or latitude bands; columns are longitude and
wrap. Threshold values depend on these. Heuristic.

**n_sectors**
Circumpolar sectors, default 6: Weddell, Indian, West
Pacific, Ross, Amundsen, Bellingshausen.

**initial_bond_probability**
Starting habitat continuity, default 0.95.

Per-sector supply and fragmentation rates live in
`spatial.py` rather than here, because the *signs* are
grounded in observed regional contrasts while the
*magnitudes* are illustrative. Keeping them in code
with that caveat attached is more honest than listing
them as parameters.

## Coupled basins (`Model/basins.py`)

**basin_coupling_strength**
Destabilising push a tipped basin exerts on each
neighbour. Default 0.06.
The literature supports the sign — most interactions
between tipping elements are destabilising — not the
magnitude. Heuristic. Sweep it.

**basin_critical_forcing**
Saddle-node of dx/dt = x − x³ + c, at 2/(3√3) ≈ 0.3849.
Analytic. Listed for reference and used as the check
value in `tests/test_structure.py`; the code computes
it rather than reading it.

## Climate modes (`Model/climate_modes.py`)

**enso_period_years**
Centre of the ENSO spectral band. Default 4.0.
The 2–7 year band is the standard description; 4 is its
centre. The realised spectrum is checked against that band
in `tests/test_structure.py`.

**enso_damping**
Bandwidth of the oscillator. Default 0.72.
Near 1 gives a narrow, almost periodic signal, which ENSO
is not. Tuned by eye to produce a broad peak. Heuristic.

**enso_subsurface_amplitude_C**
Peak El Niño subsurface warming on the West Antarctic
shelf. Default 0.5 °C. Empirical.
Worth holding next to `warming_delta_C` of 2.0: a single
El Niño delivers a quarter of the total default warming to
the depth band the reference organism occupies, then
removes it.

**enso_subsurface_depth_m**
Depth at which the response reverses sign. Default 150 m.
Empirical — the reported subsurface warming band starts
there. Above it, the surface response runs the other way.
The *sharpness* of the transition is a stand-in for Ekman
and bathymetric detail.

**lattice_max_depth_m**
Depth spanned by the lattice rows. Default 1000 m, chosen
to run past the 490 m reference depth. Heuristic.

**enso_frequency_sensitivity_per_C**
Fractional increase in ENSO frequency per °C. Default 0.02.
Direction is supported — most CMIP6 models project
increasing frequency and a shift toward Eastern Pacific
events. The coefficient is a placeholder. Heuristic.

**enso_amplitude_sensitivity_per_C**
Default 0.0, meaning **no change**.
This default is the honest one rather than a shortcut.
CMIP6 amplitude projections span an increase of up to
0.6 °C in standard deviation to a decrease of up to 0.4 °C,
with an ensemble mean near zero. The sign is unknown, so
the model does not pick one.

**teleconnection_correlation_early / _late / _shift_year**
0.72, 0.21, 2002.
The reported ENSO–Antarctic Dipole correlation before and
after a documented decadal shift. Modelled as a step
because that is how it is reported; the real transition was
presumably not instantaneous.
The consequence matters: the same ENSO now delivers less
Antarctic signal, so the environmental cue degrades even
where ENSO itself is unchanged.

**sam_period_years**, **sam_trend_per_year**
1.6 years and 0.0. SAM varies faster than ENSO and carries
a real positive trend — it is reported as being in its most
positive state in roughly a thousand years — but the trend
in index units per year is a modelling choice, so it
defaults to off.

Sector patterns for the ice dipole and the subsurface CDW
response live in `climate_modes.py` rather than here, for
the same reason as the spatial trends: signs grounded,
magnitudes shaped.

⸻

⸻

# Known weak points in these definitions

**Sea ice as habitat continuity.**
`patch_autocorrelation` and the lattice bond
probability are anchored partly on sea ice
observations, but ice extent is a proxy for habitat
continuity, not a measurement of it. The mapping from
ice state to effective connectivity for a deep-water
species is assumed. Adding a lattice did not fix this
— it gave the assumption more structure to be wrong
in.

**Model time in the basin layer.**
The double-well formulation is normalised. Model time
is not years, and the mapping from a normalised basin
to a physical sector is qualitative throughout.

Two mappings in `Sims/enso_coupling.py` are invented and
both matter for the basin results: degrees of warming per
unit basin forcing, and basin model time per year. The
conclusions there are conclusions about a *ratio*, which
is why the ratio is swept rather than chosen.

**Parameter count.**
The structural update replaced three scalars with
roughly twenty parameters, of which a minority are
grounded. `parameters.json` marks the rest under
`_heuristic_parameters`. The model now represents
behaviours it previously could not, and makes more
assumptions to do so.

⸻

# Superseded

**Ecological memory as lifespan.**
Was the operational definition in `geometry.md`. The
buffering mechanism is sound — overlapping generations
damp short-term variation — but lifespan is one
component of a multi-dimensional life-history
position, and recent work cautions against reading
sensitivity to environmental change off that position.

Replaced by three quantities from the projection
matrix: damping timescale (498 yr at baseline),
generation time (206 yr), and transient period
(209 yr). Against a lifespan of 300 yr. They do not
track each other under warming.

**Circumpolar scalars for mid-trophic supply.**
Replaced by per-sector supply with a
`redistribution_index()` diagnostic. On the default
trends the circumpolar mean reports a 13% decline
while sectors move up to 72% in opposing directions.
