Possible Model Extensions and Structural Additions

This section outlines candidate extensions to increase realism, introduce nonlinear regime shifts, and more explicitly represent spatial geometry in the Antarctic Basin Dynamics framework.

Status lines were added in the August 2026 literature review. Items 1, 2, 6 and 9 are implemented in `Model/core.py`; the rest remain proposals. Evidence for each is recorded in `Docs/literature.md`.

⸻

1. Nonlinear Fragmentation Dynamics (Percolation-Based Connectivity)

**Status: implemented.** `percolation_connectivity()`. Strongly supported — Antarctic sea ice held a range for four decades, then stepped to a new state in 2016 and stayed there. Exponential decay cannot produce that at any parameter value.

Current implementation:
Spatial autocorrelation decays exponentially:

ac(t) = BASE_AC * exp(-FRAG_RATE * t)

Limitation:
Real habitat fragmentation behaves more like a percolation transition. Connectivity remains high until a critical threshold, then collapses rapidly.

Proposed addition:
Replace exponential decay with a sigmoidal or percolation-style function:


ac(t) = 1 / (1 + exp(k * (t - t_c)))


Where:
	•	t_c = connectivity threshold time
	•	k = steepness of transition

This allows abrupt connectivity collapse and more realistic regime shifts.

⸻

2. Mass-Dependent Connectivity Scaling

**Status: implemented.** `mass_dependent_connectivity()`. Supported from an unexpected direction — large bodies are penalised under warming through home range demand, oxygen supply limits, and direct size-at-maturity reduction. Not a clean law: roughly 45% of species in one broad analysis were larger, not smaller, in warmer water.

Current implementation:
Effective connectivity scales as:

connectivity_factor = ac ** 0.5

Limitation:
Both strategies experience fragmentation symmetrically.

Proposed addition:
Introduce body-mass-dependent scaling:
connectivity_factor = ac ** (α * body_mass_scaling)

Where:
	•	α = scaling sensitivity parameter
	•	body_mass_scaling derived from allometric exponent

This penalizes large-bodied strategies more sharply under fragmentation.

⸻

3. Dynamic Trophic Compression

**Status: partly addressed.** Transfer *efficiency* is now dynamic under warming (`dynamic_transfer_efficiency()`), which captures the energy-loss half of this item. Trophic *depth* is still fixed at 3. The literature supports warming-driven microbial interception clearly; it does not give a defensible coefficient for β or γ, so the depth equation stays proposed.

Current implementation:
trophic_levels = 3 (fixed)

Limitation:
Warming and fragmentation often increase microbial respiration and shorten effective trophic chains.

Proposed addition:
Make trophic depth a function of temperature or connectivity:

trophic_levels = 3 + β * dT - γ * (1 - ac)

Where:
	•	Higher temperature increases microbial interception
	•	Lower connectivity increases edge-driven energy loss

This allows apex energy supply to degrade nonlinearly.

⸻

4. Density Dependence and Allee Effects

**Status: proposed.** Raised in priority. The strongest observed signal in Antarctic mid-trophic species is weak juvenile recruitment, and this model has no stage structure at all. Allee effects touch that gap but do not close it — proper stage structure is the real fix.

Current implementation:
Recruitment ∝ turnover_rate × effective_supply

Limitation:
No low-density collapse mechanism.

Proposed addition:
Introduce an Allee threshold:

if population < critical_density:
    recruitment *= (population / critical_density)

    This models mate-finding limitations and social collapse in large-bodied species.

⸻

5. Negative Basin Depth Representation

**Status: proposed.** No new evidence bears on this. It is a representational choice rather than an empirical question.

Current implementation:

basin_depth = clip(energy_balance, 0, None) * ac * lifespan_ratio

Limitation:
Energy deficits are suppressed rather than represented structurally.

Proposed addition:
Allow negative basin depth:

basin_depth = energy_balance * ac * lifespan_ratio

Negative values represent attractor destabilization rather than neutral absence.

⸻

6. Accelerating Forcing

**Status: implemented.** `accelerating_temperature()`. No longer speculative — observed abyssal warming in the Antarctic sector accelerated roughly threefold between the long-record trend and the 2017/18–2023/24 trend, with the descent rate of the coldest water nearly quadrupling. The quadratic form is now the better-supported default; the linear ramp is retained for comparison.

Current implementation:
Linear warming ramp.

Proposed addition:
Quadratic or exponential forcing:

temperature = baseline + a*t + b*t^2

This better reflects nonlinear climate trajectories and tests stability margins.

⸻

7. Explicit Spatial Representation

**Status: proposed — now the highest-value extension.** Two independent arguments converged on it. Antarctic krill are declining in the Atlantic–Bellingshausen sector while increasing in the Ross–Pacific sector, so a circumpolar scalar reports stability where the signal is spatial reorganisation. And the ice sheet is described not as one tipping element but as several interacting systems across drainage basins, where one tipping makes another more likely. Neither behaviour can appear in a scalar, single-basin formulation.

Current implementation:
Autocorrelation treated as scalar.

Proposed addition:
Introduce 2D lattice or graph-based habitat grid:
	•	Nodes represent habitat patches
	•	Edges represent dispersal corridors
	•	Percolation threshold computed dynamically

Agents disperse spatially rather than responding to scalar connectivity.

This enables visualization of corridor loss and patch isolation.

⸻

8. Forcing Isolation Experiments

**Status: proposed.** More necessary than before. This update added three new coupled mechanisms — accelerating forcing, dynamic transfer efficiency, oxygen limitation — on top of the existing ones. Without isolation runs there is no way to tell which one is driving a given trajectory.

Run controlled experiments isolating drivers:
	1.	Warming only
	2.	Fragmentation only
	3.	Microbial amplification only
	4.	Combined forcing

Map nonlinear interaction surfaces rather than single trajectories.

⸻

9. Oxygen Limitation Coupling 

**Status: implemented.** `oxygen_availability()` and `metabolic_index()`. Three things made the omission hard to justify: about 80% of committed ocean oxygen loss lands below 2000m, that is where this model's apex proxy lives, and the penalty scales with body mass rather than being mass-neutral. Implemented via the Metabolic Index framing rather than the divisor sketched below, since that formalism is the standard one.

Introduce temperature-dependent oxygen solubility constraint:

oxygen_availability = f(temperature)
metabolic_cost_adjusted = met_mult / oxygen_availability

This captures demand–supply mismatch under warming.

⸻

10. Multi-Timescale Forcing

**Status: proposed.** Partly addressed — the long-term trend term is now non-linear (item 6). Seasonal and stochastic components remain absent.

Introduce slow and fast oscillatory components:




temperature = baseline
             + long_term_trend
             + seasonal_cycle
             + stochastic_variability



             This allows analysis of resilience under compound variability.

⸻

Conceptual Goal of Extensions

These additions shift the model from smooth parameter degradation toward:
	•	Threshold dynamics
	•	Percolation collapse
	•	Asymmetric scaling
	•	Nonlinear attractor deformation

The objective is not to increase complexity for its own sake, but to better represent the geometry of dependency collapse under compound forcing.

⸻

If implemented incrementally, each addition can be tested independently to map which structural changes most strongly alter slow-integrator stability.

⸻

## Status summary (August 2026)

| # | Extension | Status |
|---|---|---|
| 1 | Percolation connectivity | implemented |
| 2 | Mass-dependent connectivity | implemented |
| 3 | Dynamic trophic compression | partly — efficiency yes, depth no |
| 4 | Density dependence / Allee | proposed, priority raised |
| 5 | Negative basin depth | proposed |
| 6 | Accelerating forcing | implemented |
| 7 | Explicit spatial representation | proposed, highest value |
| 8 | Forcing isolation experiments | proposed, more necessary |
| 9 | Oxygen limitation coupling | implemented |
| 10 | Multi-timescale forcing | partly — trend only |

The four implemented items each default to a neutral parameter value that reproduces the model's prior behaviour. They change nothing until engaged deliberately.

The three highest-value remaining items — spatial representation, stage structure, and multiple coupled basins — are all the same kind of change: the model currently collapses things that the observations say are pulling in different directions.

