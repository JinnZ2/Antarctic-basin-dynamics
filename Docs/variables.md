# Variable definitions

## Temperature parameters

**baseline_temp_C**
Starting water temperature in degrees Celsius.
Antarctic deep water reference: ~1.3°C at 490m depth.

**warming_delta_C**
Temperature increase applied to baseline.
Range of interest: 0–6°C over decadal to centennial timescales.

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

## Energy parameters

**trophic_transfer_efficiency**
Fraction of energy passing between trophic levels.
Empirical range: 0.05–0.15. Default: 0.10.
Note: warming may reduce this value by increasing 
microbial loop interception. Not currently dynamic 
in this model — a known simplification.

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

**effective_patch_size**
autocorrelation² × minimum viable landscape
Nonlinear because connectivity loss compounds.
Below 50% of MVL: viability threshold crossed.

**energy_balance**
Normalized supply minus normalized demand at apex level.
Negative values indicate budget deficit conditions.
Stochastic variant adds prey availability noise.
