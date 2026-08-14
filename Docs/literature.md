# Literature update — August 2026

This document records the empirical basis for the
model's parameters and states where recent work has
changed, tightened, or contradicted an earlier
assumption.

The model remains geometric, not predictive
(see `interpretation_notes.md`). Literature is used
here to constrain the *shape* of relationships and
the *plausible range* of parameters — not to claim
forecast skill.

Each section below follows the same structure:

**Finding** — what the recent work reports
**Implication** — what it means for this model
**Change** — what was actually altered in the repo

Entries marked `[proposed]` were not implemented;
they are recorded so the reasoning is not lost.

⸻

## 1. Forcing: the warming ramp is accelerating, and it is deep

**Finding.**
Abyssal warming in the Antarctic sector has
accelerated sharply. In the eastern Bellingshausen
Basin the trend over 2017/18–2023/24 is
7.5 ± 0.9 m°C yr⁻¹, roughly triple the
2.8 ± 0.2 m°C yr⁻¹ trend measured since 1992/95.
The descent rate of the coldest water nearly
quadrupled, from 7.8 to 28 m yr⁻¹
([Johnson et al., *GRL*, 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL109937)).

Antarctic Bottom Water volume declined by about 3%
over 2002–2023, with the loss rate increasing
roughly fourfold after 2015, concentrated in the
Weddell sector. AABW is simultaneously warming,
freshening, and losing oxygen
([*Nature Reviews Earth & Environment*, 2025](https://www.nature.com/articles/s43017-025-00750-2);
[*Nature Climate Change*, 2023](https://www.nature.com/articles/s41558-023-01695-4)).

A large pool of Circumpolar Deep Water has expanded
and moved closer to the continental shelf over the
past two decades, detected by combining ship
hydrography, Argo floats, and machine learning
([2026 synthesis](https://www.sciencedaily.com/releases/2026/04/260429102023.htm)).
Global ocean heat content set records in both 2024
and 2025
([*Advances in Atmospheric Sciences*, 2026](https://link.springer.com/article/10.1007/s00376-026-5876-0)).

**Implication.**
The model's linear warming ramp is the wrong shape.
Observed deep warming is not linear — the rate itself
is increasing. `additions.md` item 6 (accelerating
forcing) is no longer a speculative extension; it is
the better-supported default.

The relevant forcing for a deep-water apex proxy is
subsurface and abyssal warming, not surface anomaly.
Those are different numbers, and the deep signal is
the one that matters here.

**Change.**
Added `abyssal_warming_rate_C_per_year` (0.0075) and
`abyssal_warming_rate_prior_C_per_year` (0.0028) to
`parameters.json`, plus
`forcing_acceleration_C_per_year2` (2.0e-4), derived
by treating each reported trend as the instantaneous
rate at its own record midpoint (≈2008.5 and ≈2020.5)
and differencing:
(0.0075 − 0.0028) / 12 ≈ 3.9e-4 °C yr⁻², halved to
give the quadratic coefficient *b* in
T = T₀ + a·t + b·t².
Added `accelerating_temperature()` to
`Model/core.py`.

This derivation assumes the acceleration is smooth
and that a single-basin trend generalises. Neither is
established. It is a shape, not a projection.

The linear ramp is retained as a comparison case.
The difference between the two trajectories is the
point.

⸻

## 2. The apex proxy: rate-of-living is now the weakest assumption in the model

**Finding.**
The Greenland shark genome has been sequenced at
chromosome level (5.9 Gb, scaffold N50 233 Mb,
96.7% completeness). Analyses of gene family
expansion and positive selection recover expansions
in *TNF*, *TLR*, and *LRRFIP* — all NF-κB pathway
activators — alongside signatures in DNA repair and
cancer resistance, relative to short-lived sharks
([*PNAS*, 2026](https://www.pnas.org/doi/10.1073/pnas.2601272123);
[preprint](https://www.biorxiv.org/content/10.1101/2025.02.19.638963v1.full)).

A 2026 study provides the first histological and
molecular analysis of cardiac aging in the species
and reports resilience rather than the expected
age-related degeneration
([Chiavacci et al., *Aging Cell*, 2026](https://onlinelibrary.wiley.com/doi/10.1111/acel.70505)).
The visual system has also now been described
([*Nature Communications*, 2025](https://www.nature.com/articles/s41467-025-67429-6)).

The radiocarbon age estimate of 392 ± 120 years
remains the reference figure. The 250–400 year band
used in this model is still defensible.

Taxonomically, molecular work recovers
*S. antarcticus* nested within the *S. pacificus*
clade, supporting synonymization
([*Journal of Heredity*, 2023](https://academic.oup.com/jhered/article/114/2/152/6881712)).

In January 2025 a sleeper shark was filmed in
Antarctic waters off the South Shetland Islands at
roughly 490 m depth in water near 1.1 °C — the first
such footage
([Smithsonian, 2025](https://www.smithsonianmag.com/smart-news/see-the-first-known-footage-of-an-elusive-southern-sleeper-shark-swimming-in-antarcticas-near-freezing-waters-180988227/)).

**Implication.**
Two separate things happened here.

The trivial one: the model's baseline of ~1.3 °C at
~490 m, chosen as a plausible reference, turns out to
sit almost exactly where a Somniosus-class animal was
actually observed. The baseline stands.

The non-trivial one: `adjusted_lifespan = baseline /
metabolic_multiplier` is a rate-of-living statement.
It says longevity is a passive consequence of slow
metabolism. The genomic and cardiac evidence says
otherwise — extreme longevity in this lineage
involves active, encoded maintenance (DNA repair,
tumour suppression, inflammatory regulation, cardiac
resilience) that is not obviously a function of
ambient temperature.

Warming should still compress lifespan. But the
compression should be *damped*, not proportional. The
model previously flagged this in `variables.md` as a
"known limitation." It is now a limitation with
direct evidence behind it, so it gets a parameter
instead of a caveat.

**Change.**
Added `longevity_maintenance_decoupling` (default
0.5) to `parameters.json` and
`maintenance_adjusted_lifespan()` to `Model/core.py`.
At 0.0 the function reduces exactly to the existing
rate-of-living behaviour, so nothing that depends on
the old curve breaks.

The value 0.5 is **not** empirically calibrated. No
study gives a compression coefficient for this
lineage. It is a deliberate mid-point that makes the
uncertainty visible rather than hiding it inside a
harder assumption. Treat it as a knob to sweep, not
a result.

⸻

## 3. Trophic transfer efficiency is not a constant

**Finding.**
The clearest single result relevant to this model:
in a seven-year experimental warming study, 4 °C of
warming decreased trophic transfer efficiency by up
to 56% relative to ambient. Phytoplankton and
zooplankton biomass were both lower in warmed
treatments
([Barneche et al., *Nature* 592:76–79, 2021](https://www.nature.com/articles/s41586-021-03352-2)).

Carbon flow through the microbial loop increases with
temperature, though bacterial growth efficiency falls
as communities adapt — the two effects partly oppose
each other
([*PMC*, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8940968/)).

In the Southern Ocean specifically, transfer of
diatom material through the subpolar twilight zone is
inefficient
([2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11732746/)),
and warming interacts with iron supply to shift
phytoplankton size structure and composition, which
in turn sets transfer efficiency
([*Biogeosciences* 21:4637, 2024](https://bg.copernicus.org/articles/21/4637/2024/)).
Diatom-dominated communities yield shorter, more
efficient chains; small-cell communities lengthen
them
([*Frontiers in Marine Science*, 2022](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.916140/full)).

Transfer efficiency — not primary productivity — is
what determines trophic cascade strength
([Zhou et al., *Ecology*, 2025](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecy.4482)).
Reviews continue to stress that the 10% rule of thumb
masks large variation across systems, levels, and
currencies
([Song et al., *Limnology & Oceanography*, 2026](https://aslopubs.onlinelibrary.wiley.com/doi/10.1002/lno.70343)).

**Implication.**
`trophic_transfer_efficiency` being a fixed 0.10 is
now the model's largest quantified underestimate.
`interpretation_notes.md` already said trophic
mismatch was "probably underestimated." Barneche et
al. lets us say by roughly how much.

Note the direction of the compounding: efficiency
enters `trophic_energy_interception()` once per
trophic level. A 56% reduction in per-step efficiency
across three levels is not a 56% reduction in apex
supply — it is far larger. This is the mechanism the
model exists to make visible, and holding efficiency
constant was suppressing it.

**Change.**
Added `tte_warming_sensitivity_per_C` (0.14) to
`parameters.json` and `dynamic_transfer_efficiency()`
to `Model/core.py`. The value is a first-order linear
fit to "up to 56% at 4 °C" (0.56 / 4 ≈ 0.14), floored
so efficiency cannot go non-positive.

Caveats worth stating plainly: the experiment was
freshwater mesocosms, not the Southern Ocean; "up to"
is an upper bound, not a mean; and a linear fit to a
single endpoint is a crude functional form. The
sensitivity is exposed as a parameter precisely so it
can be swept rather than trusted.

⸻

## 4. Oxygen: the axis the model was missing

**Finding.**
Committed ocean oxygen loss is roughly fourfold what
has already been realised, and about 80% of it occurs
below 2000 m. The deep ocean loses more than 10% of
its pre-industrial oxygen content even under
immediate emissions cessation
([*Nature Communications*, 2021](https://www.nature.com/articles/s41467-021-22584-4)).

The Metabolic Index framework — the ratio of oxygen
supply to temperature-dependent resting demand —
continues to be the standard formalism for this
constraint
([Deutsch et al., *Science*, 2015](https://www.science.org/doi/10.1126/science.aaa1605);
[*Biogeosciences* 21:3477, 2024](https://bg.copernicus.org/articles/21/3477/2024/)).

Critically for this model: oxygen availability and
body mass jointly modulate ectotherm responses to
warming — the constraint is not mass-neutral
([*Nature Communications*, 2023](https://www.nature.com/articles/s41467-023-39438-w)).

AABW is deoxygenating alongside its warming and
freshening ([Section 1](#1-forcing-the-warming-ramp-is-accelerating-and-it-is-deep)).

**Implication.**
`interpretation_notes.md` listed oxygen as "adds to
mismatch but is not included." Three things make that
exclusion harder to justify now: the deep ocean is
where most of the loss lands, the deep ocean is where
this model's apex proxy lives, and the penalty scales
with body mass — which is exactly the asymmetry the
rest of the model is built to represent.

Warming raises demand and lowers supply
simultaneously. That is a squeeze, not a sum, and it
falls hardest on the largest-bodied slow integrators.

**Change.**
Added `oxygen_baseline_saturation`,
`deep_oxygen_committed_loss_fraction` (0.10), and
`oxygen_mass_sensitivity` to `parameters.json`.
Added `oxygen_availability()` and `metabolic_index()`
to `Model/core.py`, implementing `additions.md`
item 9.

⸻

## 5. Body size: the allometric penalty has empirical support

**Finding.**
Roughly 9,000 body-size changes drawn from fossil,
historical, and modern records show that size
reduction is a general response of marine ectotherms
to environmental crisis, with warming events eliciting
especially strong shifts
([*PNAS*, 2025](https://www.pnas.org/doi/abs/10.1073/pnas.2505564123)).

Over 80% of ectotherm species examined follow the
temperature–size rule — faster juvenile growth,
smaller adult size
([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7821163/);
[Frizot et al., *Ecology Letters*, 2025](https://onlinelibrary.wiley.com/doi/10.1111/ele.70273)).
Projections grounded in oxygen mass-scaling limits
give 14–39% size reductions by 2050 for tropical reef
fishes
([*Nature Communications*, 2024](https://www.nature.com/articles/s41467-024-49459-8)).

The response is not universal. In one broad analysis
about 55% of species were smaller in warmer water and
45% were larger
([2020](https://pubmed.ncbi.nlm.nih.gov/32251381/)).

**Implication.**
This supports `additions.md` item 2 — mass-dependent
connectivity scaling — from a different direction
than expected. Large-bodied strategies are penalised
under warming through at least three partly
independent channels: home-range demand under
fragmentation, oxygen supply limits under
deoxygenation, and direct size-at-maturity reduction.

The 45% counter-case is a genuine caution against
treating this as a clean monotonic law. It is a
tendency with real variance, not a rule.

**Change.**
Added `mass_dependent_connectivity()` to
`Model/core.py` with
`connectivity_mass_sensitivity` in `parameters.json`
(implements `additions.md` item 2). The 14–39%
figure is recorded here as a range of interest but is
**not** used as a default — it is derived from
tropical reef fish and does not transfer to a polar
deep-water elasmobranch.

⸻

## 6. Connectivity: sea ice is the fragmentation driver, and it is behaving like a threshold

**Finding.**
Antarctic sea ice extent reached 1.77 million km² on
19 February 2023 — 36% below the 1979–2022 mean
minimum. February 2024 tied for second-lowest, and
low coverage persisted through the winters of 2023
and 2024
([NOAA Climate.gov](https://www.climate.gov/news-features/event-tracker/antarctic-sea-ice-summer-minimum-ties-second-lowest-record-2024);
[Antarctic Environments Portal](https://environments.aq/publications/antarctic-sea-ice-4-record-lows-between-2022-and-2025/)).

Statistical analysis over 1979–2022 identifies three
distinct regimes, the most recent beginning in
September 2016 and coinciding with Southern Ocean
warming — interpreted as a new sea ice state rather
than an excursion
([*Communications Earth & Environment*, 2023](https://www.nature.com/articles/s43247-023-00961-9)).

The 2023 record low drove unprecedented wintertime
turbulent ocean heat loss, enhanced storminess, and
altered dense water formation
([*Nature*, 2024](https://www.nature.com/articles/s41586-024-08368-y)).
The record low was assessed as extremely unlikely
without anthropogenic climate change
([2024 attribution study](https://www.sciencedaily.com/releases/2024/05/240520122718.htm)).
Compound drivers link sea ice loss to Southern Ocean
destratification
([*Science Advances*, 2026](https://www.science.org/doi/10.1126/sciadv.aeb0166)).

**Implication.**
This is the strongest support in the update for
`additions.md` item 1. A system that holds a range
for four decades, then steps to a new state in a
single year and stays there, is not decaying
exponentially. It is crossing a threshold.

`ac(t) = BASE_AC * exp(-FRAG_RATE * t)` cannot
produce that behaviour at any parameter value. The
sigmoidal form can.

The regime break also gives the transition an
approximate anchor: if 2016 is treated as t=0, the
observed step is ~7 years in, which is what
`connectivity_threshold_time_years` is set from.

Two honest caveats. First, sea ice extent is a proxy
for habitat continuity, not a measurement of it —
the mapping from ice state to effective connectivity
for a deep-water species is assumed, not observed.
Second, "percolation threshold" in the sea ice
literature usually refers to brine flow through the
ice matrix (~10% porosity), a physical process at a
completely different scale from habitat percolation.
The mathematical form is shared; the phenomenon is
not. Do not cite one as evidence for the other.

**Change.**
Added `percolation_connectivity()` to
`Model/core.py` with
`connectivity_threshold_time_years` (7.0) and
`connectivity_transition_steepness` (0.6) in
`parameters.json`. The exponential decay path is
retained — `fragmentation_rate` is unchanged — so the
two forms can be compared directly.

One flaw worth naming: the sigmoid proposed in
`additions.md` asymptotes to zero, but the observed
sea ice step was to a *lower state*, not to nothing.
The function takes an optional `floor` for residual
connectivity, defaulting to 0.0 to match the proposed
form. The default is almost certainly too severe at
long horizons.

⸻

## 7. Mid-trophic: krill are redistributing, not simply declining

**Finding.**
*Euphausia superba* distribution in its southwest
Atlantic population centre has contracted southward
over ~90 years, with sharp density declines near the
northern limit, concentration toward the shelves, and
increasing mean body length reflecting weak juvenile
recruitment
([*Nature Climate Change*, 2019](https://www.nature.com/articles/s41558-018-0370-z)).

Recent work reports declines in the
Atlantic–Bellingshausen sector alongside increases in
the Ross–Pacific sector, driven by regional
differences in temperature and ice
([*Nature Reviews Earth & Environment*, 2023](https://www.nature.com/articles/s43017-023-00504-y);
[habitat suitability projections, 2025](https://www.sciencedirect.com/science/article/abs/pii/S0025326X25006174)).
Responses are stage-specific: juveniles are more
vulnerable, while spawning adults show compensating
strategies
([*2026*](https://www.sciencedirect.com/science/article/abs/pii/S0301479726015057)).

**Implication.**
Two things the model currently cannot express.

Redistribution is not decline. A circumpolar scalar
for mid-trophic supply averages over sectors moving
in opposite directions and reports a small net change
where the actual signal is spatial reorganisation.
This is a direct argument for `additions.md` item 7
(explicit spatial representation), independent of the
fragmentation argument.

Stage structure matters. The strongest signal is in
recruitment, and the model has no age or stage
structure at all — recruitment enters only as a
scalar rate. `additions.md` item 4 (Allee effects)
touches this but does not solve it.

**Change.**
Originally recorded as `[proposed]` — the main
structural gap, too large for that update to make.

**Closed.** `Model/spatial.py` replaces the scalar
with a circumpolar lattice partitioned into six
sectors, and `Model/population.py` replaces scalar
abundance with an age-classed projection matrix.
Sector trend signs are grounded; magnitudes are not.
See `structure.md` sections 1 and 2.

The redistribution diagnostic confirms the concern
quantitatively: on the default trends the circumpolar
mean reports a 13% decline while sectors move up to
72% in opposing directions.

⸻

## 8. Thresholds: Antarctica is several tipping systems, not one

**Finding.**
A first threshold, potentially as low as 1–2 °C above
pre-industrial, is associated with long-term collapse
of about 40% of marine ice volume in West Antarctica.
Marine-based East Antarctic sectors representing
~5 m of potential sea-level rise are at risk of
losing stability at 2–5 °C
([*Nature Climate Change*, 2025](https://www.nature.com/articles/s41558-025-02554-0)).

The ice sheet does not behave as a single tipping
element but as several interacting systems across
drainage basins, and interactions between tipping
systems are frequently destabilising — one tipping
makes another more likely
([Global Tipping Points Report 2025](https://www.natureinsights.earth/post/global-tipping-points-report-2025-the-world-has-entered-a-new-climate-reality);
[Kubiszewski et al., *Ambio*, 2024](https://www.robertcostanza.com/wp-content/uploads/2025/01/2024_J_Kubiszewski-et-al.-Antarctic-tipping-points-Ambio-2024.pdf)).
Projected ice loss remains subject to deep
uncertainty
([*Science*, 2025](https://www.science.org/doi/abs/10.1126/science.adt9619)).

**Implication.**
The model's framing — basins in phase space, with
depth as distance to bifurcation — turns out to match
how the physical system is now being described. That
is a convenient alignment, and convenient alignments
deserve suspicion, so state it carefully: the
geometry is a useful lens, not a validated
correspondence.

The concrete lesson is *multiple coupled basins*. The
current model has one. Cascading destabilisation
between basins cannot appear in a single-basin
formulation, and it is precisely the behaviour the
tipping-point literature identifies as the dangerous
one.

The 1–2 °C figure also sits inside the model's
`warming_delta_C` default of 2.0. The default is not
a conservative choice.

**Change.**
Originally recorded as `[proposed]` — the
highest-value structural extension.

**Closed.** `Model/basins.py` implements several
coupled double-well potentials with destabilising
interactions, checked against the analytic saddle-node
at 2/(3√3). On the default configuration one sector
crosses its own threshold and two more follow.

The coupling *sign* is what the literature supports.
The magnitude is invented — `ring_coupling()` is a
placeholder for a matrix nobody has measured. See
`structure.md` section 3.

⸻

## 9. Ecological memory: keep the mechanism, weaken the shortcut

**Finding.**
Long-lived, multi-aged populations with overlapping
generations buffer short-term environmental
perturbation; short-lived populations lack that
buffer and track seasonal variation directly.

But recent work cautions explicitly against reading
sensitivity to environmental change off life-history
position
([Rademaker et al., *Journal of Animal Ecology*, 2024](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.14050)),
and against treating the fast–slow continuum as a
single axis at all
([*Trends in Ecology & Evolution*, 2024](https://www.cell.com/trends/ecology-evolution/fulltext/S0169-5347(24)00139-3)).
Demographic buffering varies along a continuum set by
survival–reproduction trade-offs
([*Nature Communications*, 2026](https://www.nature.com/articles/s41467-026-73720-x)).

**Implication.**
The buffering mechanism in `geometry.md` is sound.
The operational shortcut — "ecological memory =
lifespan of dominant long-lived species" — is the
part now under pressure. Lifespan is one component of
a multi-dimensional life-history position, and the
literature is explicit that the single-axis reading
misleads.

**Change.**
Originally documentation only — `geometry.md` and
`variables.md` flagged the shortcut as an operational
simplification with a known failure mode.

**Closed.** The age-classed matrix in
`Model/population.py` supplies measures that do not
depend on reading sensitivity off a single
life-history axis: generation time (206 yr), the
damping timescale (498 yr), and the period of the
transient oscillation (209 yr).

The lifespan proxy understated the integration window
by about 66%, and the three quantities do not track
each other under warming. See `structure.md`
section 2.

⸻

## 10. ENSO: the interannual mode, and it is a depth signal

**Finding.**
The Antarctic Dipole is a seesaw in sea ice between the
Ross–Amundsen and Bellingshausen–Weddell sectors. During
austral spring an ENSO-generated Pacific–South American
wave train shifts and deepens the Amundsen Sea Low,
producing increased sea ice in the northern Ross Sea and
decreases in the Bellingshausen and northern Weddell.
La Niña reverses it
([*Atmosphere*, 2023](https://doi.org/10.3390/atmos14111659);
[*Journal of Climate*, 2025](https://journals.ametsoc.org/view/journals/clim/39/1/JCLI-D-24-0190.1.xml)).

The signal reverses sign with depth. During El Niño a
weaker Amundsen Sea Low weakens coastal easterlies,
reducing on-shelf Ekman transport of cold surface water
and admitting warm Circumpolar Deep Water. Subsurface
warming **between 150 m and the shelf bottom reaches
0.5 °C**; La Niña produces the opposite, with a stronger
low, stronger Ekman transport, less cross-shelf CDW and
subsurface cooling
([Huguenin et al., *GRL*, 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL104518)).
Thermocline depth variability at the Pine Island and
Thwaites ice shelf fronts correlates strongly with basal
melt rates
([*Nature Communications*, 2024](https://www.nature.com/articles/s41467-024-47084-z)).

The teleconnection is non-stationary. Correlation between
mature-phase ENSO and the subsequent Antarctic Dipole fell
from **0.72 over 1979–2001 to 0.21 over 2002–2020**
([*Atmosphere*, 2023](https://doi.org/10.3390/atmos14111659);
[*Climate Dynamics*, 2022](https://link.springer.com/article/10.1007/s00382-022-06364-4)),
with consequences for sea ice predictability
([*GRL*, 2026](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2026GL123531)).

Under warming, most CMIP6 models project increasing ENSO
frequency and a shift toward Eastern Pacific events, with
extreme EP El Niño becoming more likely. Amplitude
projections span an increase of up to 0.6 °C in standard
deviation to a decrease of up to 0.4 °C, ensemble mean
near zero
([*GRL*, 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL116541);
[*npj Climate and Atmospheric Science*, 2022](https://www.nature.com/articles/s41612-022-00324-9);
[*Nature Climate Change*, 2023](https://www.nature.com/articles/s41558-023-01610-x)).

Krill recruitment and Southern Ocean ecosystem
fluctuations are linked to both ENSO and the Southern
Annular Mode
([*Science Advances*, 2023](https://www.science.org/doi/10.1126/sciadv.adh4584);
[*Proc. R. Soc. B*, 2007](https://royalsocietypublishing.org/doi/10.1098/rspb.2007.1180)).
SAM is reported to be in its most positive state in roughly
a thousand years
([*Nature Reviews Earth & Environment*, 2025](https://www.nature.com/articles/s43017-025-00746-y)).

**Implication.**
Three, in ascending order of how much they change the
model.

The 0.5 °C subsurface amplitude applies to the band from
150 m to the shelf bottom. The model's baseline is 1.3 °C
at 490 m. So ENSO delivers **a quarter of the model's
default 2.0 °C `warming_delta_C`** to the exact depth the
reference organism occupies, then takes it back.

The sign reverses with depth. A model carrying one
temperature per sector has to pick a sign, and either
choice is wrong for the other half of the water column.
The lattice already had depth rows and nothing had used
them.

`spatial.py` carried per-sector supply trends that were
monotone, with grounded signs and invented magnitudes.
The dipole is what those trends were imitating — one phase
of an oscillation frozen into a trend. Since both were
written from the same reported regional contrasts, their
agreement is a consistency check rather than a discovery;
what the dipole adds is the mechanism and the sign changes
a monotone trend cannot produce.

**Change.**
Added `Model/climate_modes.py`: a noise-driven damped
oscillator for ENSO with a broad 2–7 year spectral peak
(not a sine — ENSO has a band, not a frequency), a depth
profile reversing sign at 150 m, separate sector patterns
for the ice dipole and the subsurface CDW response, the
documented decadal weakening of the teleconnection, and
frequency shift under warming. SAM is included with the
same machinery and a trend defaulting to zero.
Amplitude sensitivity defaults to zero because the CMIP6
sign is genuinely unknown.

Added `population.recruitment_transfer()` and
`Sims/enso_coupling.py`. Implements `additions.md` item 10.

**Results, including one that corrects an earlier claim.**

`geometry.md` asserts that compressing ecological memory
lets high-frequency variation propagate further into the
food web. Nothing in the model oscillated, so this had
never been tested. It holds: ENSO-band gain into the adult
population rises about 73% between baseline and +6 °C.

But the qualitative statement omits the magnitude.
Attenuation at baseline is roughly 1150-fold and at +6 °C
roughly 670-fold. A 73% increase on a thousandfold
attenuation is still a thousandfold attenuation. The slow
integrator does not stop being slow; it stops being quite
as slow.

The basin result is a flat negative and is the more useful
one. A basin integrates forcing over its own relaxation
time, and ENSO averages to zero over any span longer than
a decade. Sweeping basin relaxation time shows that only
basins as fast as ENSO itself — a few years — can be
tipped by it. At decadal-to-centennial relaxation, an
excursion **more than three times the remaining margin**
never crosses the threshold.

Offsetting ENSO toward a persistent El Niño extends the
vulnerable range by about an order of magnitude, but that
is a shift in the mean doing the work rather than the
variability.

So the parts of the ENSO literature that matter for slow
systems are the ones about *changing statistics* — more
frequent extreme Eastern Pacific events, a weakening
teleconnection — not the ones about variance.

This corrects `interpretation_notes.md`. "Individual bad
years matter more as the margin narrows" was written about
the stochastic energy-balance variant, where it holds. It
does not hold for slow basins, which cannot see individual
bad years at all.

⸻

## Summary of parameter changes

| Parameter | Value | Status | Source |
|---|---|---|---|
| `baseline_temp_C` | 1.3 | unchanged, corroborated | §2 — 2025 Antarctic sighting at ~490 m, ~1.1 °C |
| `baseline_lifespan_years` | 300 | unchanged | §2 — 392 ± 120 yr remains reference |
| `trophic_transfer_efficiency` | 0.10 | unchanged as baseline | §3 — now a *baseline*, not a constant |
| `fragmentation_rate` | 0.02 | unchanged | §6 — retained for comparison against sigmoid |
| `Q10_*` | 2.5 / 3.0 / 3.5 | unchanged | no new work narrows these ranges |
| `tte_warming_sensitivity_per_C` | 0.14 | **new** | §3 — Barneche et al. 2021, 56% at 4 °C |
| `abyssal_warming_rate_C_per_year` | 0.0075 | **new** | §1 — Johnson et al. 2024 |
| `abyssal_warming_rate_prior_C_per_year` | 0.0028 | **new** | §1 — Johnson et al. 2024 |
| `forcing_acceleration_C_per_year2` | 2.0e-4 | **new, derived** | §1 — from the two trends above |
| `oxygen_baseline_saturation` | 1.0 | **new** | §4 — normalised reference |
| `deep_oxygen_committed_loss_fraction` | 0.10 | **new** | §4 — >10% below 2000 m |
| `oxygen_mass_sensitivity` | 0.25 | **new, heuristic** | §4 — direction empirical, magnitude not |
| `connectivity_threshold_time_years` | 7.0 | **new** | §6 — 2016 regime break to 2023 step |
| `connectivity_transition_steepness` | 0.6 | **new, heuristic** | §6 — shape assumed |
| `connectivity_mass_sensitivity` | 1.0 | **new** | §5 — 1.0 reproduces prior behaviour |
| `longevity_maintenance_decoupling` | 0.5 | **new, heuristic** | §2 — no empirical coefficient exists |

Every new default is chosen so the model reduces to
its previous behaviour at the neutral value of the
new parameter. Nothing already in the repo changes
its output unless a new parameter is deliberately
engaged.

Three values above are marked heuristic and one is
derived. They are reasoning with numbers attached,
not measurements. `parameters.json` lists them under
`_heuristic_parameters` and `_derived_parameters` so
they stay identifiable at a glance. Sweep them.

⸻

## Still not represented

- Behavioural adaptation and range shift. The spatial
  layer has patches but no agents dispersing between
  them.
- Evolutionary response on centennial timescales
- Two-way coupling between layers — supply flows
  spatial → demographic → basin, but a reorganised
  basin does not feed back into habitat structure
- Dynamic trophic depth (`additions.md` item 3,
  second half)
- Ice–albedo and methane feedbacks; the warming
  trajectory is still externally imposed

The oxygen constraint and dynamic transfer efficiency
moved off this list in the August 2026 review. Age
structure (§7), spatial geometry (§7) and coupled
basins (§8) followed in the structural update — see
`structure.md`, including what got *less* certain as a
result. Three scalars became roughly twenty
parameters, a minority of them grounded.

⸻

## Review cadence

Reviewed August 2026.

The fastest-moving inputs are Antarctic sea ice state
(§6), abyssal warming rates (§1), and Southern Ocean
ecosystem redistribution (§7). Those are worth
re-checking annually. Q10 values, allometric
exponents, and the longevity reference are stable
and do not need frequent revisiting.

When updating: add the finding, state the
implication, and record the change — including
`[proposed]` entries where the reasoning was to *not*
change something. The record of rejected changes is
as useful as the record of accepted ones.
