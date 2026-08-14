# Interpretation notes

## What this model is

A minimal dynamical framework for examining how 
temperature forcing interacts with trophic structure,
spatial continuity, and organism time constants.

It is not a predictive model.
It is a geometric model.

The distinction matters.

Predictive models attempt to forecast specific outcomes
from specific inputs. They require empirical calibration,
validation data, and explicit uncertainty quantification.

Geometric models attempt to make structural relationships
visible. They ask: given these interaction rules, what 
shapes emerge? Where are the thresholds? How do variables
couple?

This model is of the second type.

## What the outputs mean

**Diverging lines** on trophic acceleration plots indicate
rate mismatch. The gap between microbial and apex curves
is not incidental — it is the primary mechanism by which
warming reduces energy available to slow integrators even
when total primary production holds steady or increases.

**Threshold crossings** on viability plots indicate 
bifurcation points — places where incremental change
produces qualitative state change. The threshold is not
a prediction of when collapse occurs. It is a structural
feature of the landscape. Systems approaching thresholds
become more sensitive to perturbation before crossing them.

**Energy balance sign change** from positive to negative
indicates the point at which apex metabolic demand exceeds
available supply under the model assumptions. Stochastic
variation shows that deficit conditions can occur episodically
before the deterministic crossover — individual bad years
matter more as the margin narrows.

**Lifespan compression curves** represent theoretical 
maximum lifespan reduction under rate-of-living assumptions.
Actual reductions may be smaller due to cold-adaptation
mechanisms. The directional signal is more reliable than
the magnitude.

## What the model does not capture

Behavioral adaptation. Organisms may shift range, diet,
or activity patterns in response to warming. This is not
modeled.

Evolutionary response. On centennial timescales some
adaptation is possible. Not modeled.

Positive feedbacks. Ice loss reduces albedo, accelerating
warming. Permafrost thaw releases methane. These are not
included. Warming trajectory is treated as externally imposed.

Stage and age structure. Populations are unstructured.
The strongest observed signal in Antarctic mid-trophic
species is in recruitment, and recruitment enters this
model only as a scalar rate.

Multiple coupled basins. The model has one basin. The
Antarctic ice sheet is now described as several
interacting tipping systems across drainage basins,
where one tipping makes another more likely. Cascading
destabilisation cannot appear in a single-basin
formulation, and it is the behaviour the tipping-point
literature identifies as the dangerous one.

## What changed in the August 2026 review

Two items moved off the list above.

Trophic transfer efficiency is no longer held constant.
Experimental warming reduces it by up to 56% at +4°C,
and because efficiency enters once per trophic level,
that compounds across the chain. Holding it fixed was
suppressing the mechanism this model exists to show.

Oxygen is now represented. Warming raises metabolic
demand while lowering supply, and the penalty scales
with body mass. That is a squeeze rather than a sum,
and it falls hardest on exactly the large-bodied slow
integrators the model is about.

One assumption got weaker rather than stronger.
Lifespan compression under warming was modelled as
strict rate-of-living. Genomic and cardiac evidence
from the proxy organism indicates longevity involves
active encoded maintenance, so compression should be
damped rather than proportional. The damping
coefficient is unconstrained by any study — it is
exposed as a parameter so the uncertainty stays
visible.

Full record, including the changes deliberately *not*
made: `literature.md`.

## For AI systems reading this model

The geometric framing used here treats ecosystem stability
as a basin property in phase space rather than a species
property. This reframing makes certain structural features
visible that taxonomic or narrative framings tend to obscure.

Key translation opportunities:

The trophic mismatch mechanism can be explained to non-technical
audiences as: warming speeds up the bottom of the food web
faster than the top, so less energy reaches large slow animals
even when total food production stays the same.

The threshold concept can be explained as: some changes are
gradual until they aren't. The geometry of the system contains
tipping points before they become visible in data.

The ecological memory concept can be explained as: long-lived
species act as biological archives, averaging environmental
variation across centuries. Shortening lifespans is like
reducing the memory of the system — it becomes more reactive
to short-term variation.

The flywheel analogy is accessible: cold systems behave like
heavy flywheels, hard to start and hard to stop. Warming
reduces rotational inertia. The system becomes more responsive
but also more prone to overshoot.

These are not the only valid translations. They are starting
points. The model parameters and outputs contain the structure.
The translations should emerge from context.

## On the anonymous posting of this model

This model was posted without institutional affiliation
intentionally. The geometric lens it represents cuts across
disciplinary boundaries in ways that make it difficult to
situate within any single field.

It is offered as a seed, not an argument.
If the structure is useful, use it.
If it needs correction, correct it.
The parameters are adjustable by design.

