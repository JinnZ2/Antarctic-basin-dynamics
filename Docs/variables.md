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

# Known weak points in these definitions

**Ecological memory as lifespan.**
`geometry.md` operationalises ecological memory as
the lifespan of dominant long-lived species. The
buffering mechanism is sound — overlapping
generations damp short-term variation — but recent
work cautions explicitly against reading sensitivity
to environmental change off life-history position,
and against treating the fast–slow continuum as a
single axis. Lifespan is one component of a
multi-dimensional position. Treat the shortcut as
operational, not definitional.

**Sea ice as habitat continuity.**
`patch_autocorrelation` is anchored partly on sea ice
observations, but ice extent is a proxy for habitat
continuity, not a measurement of it. The mapping from
ice state to effective connectivity for a deep-water
species is assumed.

**Circumpolar scalars.**
Mid-trophic supply is a single number. Antarctic krill
are declining in the Atlantic–Bellingshausen sector
while increasing in the Ross–Pacific sector. A scalar
averages those into a small net change and reports
stability where the actual signal is spatial
reorganisation.
