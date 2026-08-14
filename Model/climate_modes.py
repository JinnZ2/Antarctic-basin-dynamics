"""ENSO and SAM as interannual forcing modes.

Implements additions.md item 10 (multi-timescale forcing) and
supplies a mechanism for something the spatial layer previously
asserted.

Three reasons this belongs in the model.

**It gives the sector dipole a cause.** `spatial.py` carried
per-sector supply trends whose signs were grounded in observed
regional contrasts but whose magnitudes were invented, and which
were monotone. The observed pattern is the Antarctic Dipole: a
seesaw in which El Nino raises sea ice in the Ross sector and
lowers it in the Bellingshausen and northern Weddell, via a
Pacific-South American wave train that shifts the Amundsen Sea
Low. That is an oscillation with a mechanism, not a trend.

**It is a depth signal, and the model's proxy lives at depth.**
During El Nino a weaker Amundsen Sea Low weakens coastal
easterlies, reducing on-shelf Ekman transport of cold surface
water and letting warm Circumpolar Deep Water onto the shelf.
Subsurface warming between 150 m and the shelf bottom reaches
0.5 C, while the surface response runs the other way. La Nina
reverses it. The model's baseline is 1.3 C at 490 m -- inside
that band. The lattice already has depth rows; this gives them
something to do.

**It is the high-frequency forcing the model never had.**
geometry.md claims that compressing ecological memory lets
high-frequency variation propagate further into the food web.
Nothing in the model oscillated, so the claim was untestable.
ENSO oscillates on 2-7 years against a population whose memory
runs to centuries.

Two things here are non-stationary and both matter. The
ENSO-Dipole correlation fell from 0.72 over 1979-2001 to 0.21
over 2002-2020, so the teleconnection itself is weakening. And
most CMIP6 models project increasing ENSO frequency under
warming, while the ensemble-mean amplitude change is near zero
with a spread from -0.4 to +0.6 C in standard deviation.

Literature: Docs/literature.md section 10, Docs/structure.md.
"""

import numpy as np

# Spectral peak and bandwidth of the ENSO oscillator. The 2-7 year
# band is the standard description; 4 years is its centre. Damping
# sets how broad the band is -- near 1 gives a narrow, almost
# periodic signal, which ENSO is not.
ENSO_PERIOD_YEARS = 4.0
ENSO_DAMPING = 0.72

# Observed standard deviation of the Nino3.4 index, used to convert
# reported event magnitudes in degrees into sigma.
NINO34_SD_C = 0.775

# Reported peak Nino3.4 anomalies, in degrees. The 1877-78 and
# 2015-16 values are a statistical dead heat and sit well within the
# uncertainty of nineteenth-century ship data, so the ranking
# between them means little. The 2026-27 figure is a forecast, not
# an observation: the consensus peak is ~3.6 C with the middle 80%
# of the ensemble at or above the previous record and the low end of
# the plume at 2.8 C.
HISTORICAL_EVENTS = {
    '1877-78': 2.73,
    '2015-16': 2.75,
    '2026-27 (forecast)': 3.6,
    '2026-27 (plume low)': 2.8,
}

# Quadratic rectification strength. ENSO is positively skewed with a
# one-sided fat tail produced by extreme El Nino events, attributed
# to dynamical nonlinearity -- nonlinear zonal advection in
# particular. Nino3.4 shows only slight positive skew, less than
# Nino3.
#
# The default reproduces that mild observed skew. It does NOT
# reproduce the observed rate of extreme events, and no value of it
# reliably could: see `return_period()`.
ENSO_SKEWNESS = 0.08

# Strength of the composite El Nino to which the 0.5 C subsurface
# figure corresponds. Composites are dominated by ordinary events,
# so this is taken as a moderate one. It is an assumption, and every
# scaled event magnitude below is proportional to it.
COMPOSITE_EVENT_SIGMA = 1.5

# Peak subsurface warming on the West Antarctic shelf during El
# Nino, between 150 m and the shelf bottom. Empirical.
ENSO_SUBSURFACE_AMPLITUDE_C = 0.5

# Depth above which the response reverses sign. Empirical: the
# reported subsurface warming band starts at 150 m.
SUBSURFACE_DEPTH_M = 150.0

# Depth range spanned by the lattice rows. The model's reference
# organism was observed near 490 m, so the domain runs past it.
LATTICE_MAX_DEPTH_M = 1000.0

# Antarctic Dipole. Sea ice and surface habitat response to El
# Nino: positive means more ice.
#
# SIGNS are grounded -- El Nino raises sea ice in the northern Ross
# sector and lowers it in the Bellingshausen and northern Weddell.
# RELATIVE MAGNITUDES are shaped to put the dipole extremes at the
# Ross and Bellingshausen poles. They are not regression
# coefficients.
SECTOR_ICE_RESPONSE = {
    'Weddell': -0.6,
    'Indian': -0.1,
    'West Pacific': 0.5,
    'Ross': 1.0,
    'Amundsen': 0.6,
    'Bellingshausen': -0.9,
}

# Subsurface CDW response to El Nino: positive means more warm
# water on the shelf. This is a different field from the ice
# response, concentrated in West Antarctica where the shelf
# warming was measured.
SECTOR_SUBSURFACE_RESPONSE = {
    'Weddell': 0.2,
    'Indian': 0.1,
    'West Pacific': 0.4,
    'Ross': 0.6,
    'Amundsen': 1.0,
    'Bellingshausen': 1.0,
}

# Decadal weakening of the ENSO-Dipole teleconnection, expressed as
# the reported correlation in each era.
TELECONNECTION_EARLY = 0.72        # 1979-2001
TELECONNECTION_LATE = 0.21         # 2002-2020
TELECONNECTION_SHIFT_YEAR = 2002

# SAM varies faster than ENSO and carries a strong positive trend --
# it is reported as being in its most positive state in roughly a
# thousand years, attributed to greenhouse forcing and ozone
# depletion.
SAM_PERIOD_YEARS = 1.6
SAM_DAMPING = 0.45


# ---------------------------------------------------------
# Mode generators
# ---------------------------------------------------------


def _damped_oscillator(n, period, damping, rng, burn_in=200):
    """Noise-driven damped oscillator, normalised to unit variance.

    A second-order autoregression whose spectrum peaks at `period`
    with a width set by `damping`:

        x[t] = 2 r cos(w) x[t-1] - r^2 x[t-2] + noise

    Deliberately not a sine wave. ENSO is quasi-periodic -- it has
    a broad spectral peak, not a frequency -- and a clean sinusoid
    would make the variance-transfer results look sharper than the
    real forcing warrants.
    """
    omega = 2.0 * np.pi / float(period)
    a1 = 2.0 * damping * np.cos(omega)
    a2 = -damping ** 2

    total = int(n) + burn_in
    noise = rng.standard_normal(total)
    x = np.zeros(total)
    for t in range(2, total):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + noise[t]

    x = x[burn_in:]
    spread = x.std()
    return x / spread if spread > 0 else x


def _rectify(x, strength):
    """Quadratic rectification, renormalised to unit variance.

    y = x + a(x^2 - 1) introduces positive skew while leaving the
    mean at zero. Chosen over an arbitrary skewing transform
    because it is the shape dynamical nonlinearity actually
    produces -- nonlinear zonal advection rectifies the anomaly
    quadratically, which is the standard explanation for ENSO's
    one-sided fat tail.
    """
    if strength == 0:
        return x
    y = x + strength * (x ** 2 - 1.0)
    spread = y.std()
    return y / spread if spread > 0 else y


def enso_index(n_years, rng=None, period=ENSO_PERIOD_YEARS,
               damping=ENSO_DAMPING, skewness=0.0):
    """Standardised ENSO index, one value per year.

    Positive is El Nino. Unit variance by construction, so
    amplitude is applied downstream rather than being baked in.

    `skewness` defaults to 0 so existing results are unchanged.
    Pass ENSO_SKEWNESS for the mildly skewed observed shape, or a
    larger value to fatten the tail deliberately -- but read
    `return_period()` before treating any value as calibrated.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    base = _damped_oscillator(n_years, period, damping, rng)
    return _rectify(base, skewness)


def sam_index(n_years, rng=None, period=SAM_PERIOD_YEARS,
              damping=SAM_DAMPING, trend_per_year=0.0):
    """Standardised SAM index with an optional positive trend.

    The trend defaults to zero so it has to be asked for. The
    observed positive trend is real but its magnitude in index
    units per year is a modelling choice, not a measurement.
    """
    rng = np.random.default_rng(1) if rng is None else rng
    base = _damped_oscillator(n_years, period, damping, rng)
    return base + trend_per_year * np.arange(n_years)


# ---------------------------------------------------------
# Teleconnection
# ---------------------------------------------------------


def teleconnection_strength(year, early=TELECONNECTION_EARLY,
                            late=TELECONNECTION_LATE,
                            shift_year=TELECONNECTION_SHIFT_YEAR):
    """Coupling between ENSO and the Antarctic Dipole, by calendar year.

    A step, because that is how the shift is reported: correlation
    0.72 across 1979-2001 against 0.21 across 2002-2020. The real
    transition was presumably not instantaneous, and treating it as
    one is the crude part of this function.

    The consequence is worth stating plainly: a weakening
    teleconnection means the same ENSO delivers less Antarctic
    signal, so the ecosystem's environmental cue degrades even
    where ENSO itself is unchanged.
    """
    year = np.asarray(year, dtype=float)
    return np.where(year < shift_year, early, late)


def row_depths(rows, max_depth=LATTICE_MAX_DEPTH_M):
    """Mid-depth of each lattice row, surface first."""
    edges = np.linspace(0.0, max_depth, int(rows) + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def depth_weight(depth_m, transition=SUBSURFACE_DEPTH_M, sharpness=80.0):
    """Sign and strength of the ENSO temperature response with depth.

    Runs from -1 at the surface to +1 well below the transition
    depth. During El Nino the subsurface warms while the surface
    response goes the other way, so the two must not share a sign.

    The transition depth is empirical. Its sharpness is not -- the
    real profile is set by Ekman dynamics and shelf bathymetry, and
    a tanh is a stand-in for it.
    """
    depth_m = np.asarray(depth_m, dtype=float)
    return np.tanh((depth_m - transition) / sharpness)


def temperature_anomaly(index, sector_names, rows,
                        amplitude=ENSO_SUBSURFACE_AMPLITUDE_C,
                        strength=1.0, max_depth=LATTICE_MAX_DEPTH_M,
                        sector_response=None):
    """ENSO temperature anomaly over the (depth, sector) grid, in C.

    Returns shape (rows, n_sectors) for a scalar index, or
    (n_times, rows, n_sectors) for a series.

    The signal is separable by construction -- a depth profile
    times a sector pattern -- which the real teleconnection is not.
    It is the least structure that can carry the two facts that
    matter: the sign flips with depth, and the magnitude varies by
    sector.
    """
    sector_response = (SECTOR_SUBSURFACE_RESPONSE if sector_response is None
                       else sector_response)
    index = np.atleast_1d(np.asarray(index, dtype=float))

    profile = depth_weight(row_depths(rows, max_depth))
    pattern = np.array([sector_response[name] for name in sector_names])

    field = np.einsum('t,r,s->trs', index, profile, pattern)
    field *= amplitude * strength
    return field[0] if field.shape[0] == 1 else field


def habitat_anomaly(index, sector_names, strength=1.0,
                    sector_response=None):
    """Antarctic Dipole anomaly in surface habitat, per sector.

    Dimensionless, positive meaning more sea ice and better surface
    habitat. This is the seesaw: the same El Nino that improves the
    Ross sector degrades the Bellingshausen.
    """
    sector_response = (SECTOR_ICE_RESPONSE if sector_response is None
                       else sector_response)
    index = np.atleast_1d(np.asarray(index, dtype=float))
    pattern = np.array([sector_response[name] for name in sector_names])
    field = strength * np.outer(index, pattern)
    return field[0] if field.shape[0] == 1 else field


# ---------------------------------------------------------
# ENSO under warming
# ---------------------------------------------------------


def warmed_period(delta_T, base_period=ENSO_PERIOD_YEARS,
                  frequency_sensitivity=0.02):
    """ENSO period shortening under warming.

    Most CMIP6 models project increasing ENSO frequency under
    anthropogenic warming, driven by a shift toward Eastern Pacific
    events and higher frequency in both flavours.

    The direction is supported. The coefficient is not -- 2% per
    degree is a placeholder that makes the effect visible without
    claiming to quantify it.
    """
    return base_period / (1.0 + frequency_sensitivity * float(delta_T))


def warmed_amplitude(delta_T, base_amplitude=ENSO_SUBSURFACE_AMPLITUDE_C,
                     amplitude_sensitivity=0.0):
    """ENSO amplitude under warming. Defaults to no change.

    This default is the honest one. CMIP6 amplitude projections
    span an increase of up to 0.6 C in standard deviation to a
    decrease of up to 0.4 C, with an ensemble mean near zero. The
    sign is genuinely unknown, so the model does not pick one.

    Extreme Eastern Pacific events are separately projected to
    become more likely, which is a change in the tail rather than
    in the variance. This function does not represent that.
    """
    return base_amplitude * (1.0 + amplitude_sensitivity * float(delta_T))


def event_sigma(nino34_anomaly_C, sd=NINO34_SD_C):
    """Convert a reported Nino3.4 anomaly in degrees into sigma."""
    return np.asarray(nino34_anomaly_C, dtype=float) / sd


def subsurface_anomaly_C(sigma, amplitude=ENSO_SUBSURFACE_AMPLITUDE_C,
                         composite_sigma=COMPOSITE_EVENT_SIGMA):
    """Antarctic subsurface warming for an event of a given size.

    Scales the measured composite response linearly in event
    magnitude. Two caveats, both material:

    The composite strength it is scaled from is assumed, not
    reported, so every number out of this is proportional to a
    guess.

    Linearity almost certainly fails at the top. The mechanism is
    Ekman transport shutting down and letting Circumpolar Deep
    Water onto the shelf, and once the on-shelf transport is fully
    suppressed a stronger event cannot suppress it further. Real
    response should saturate. This function does not, so it is an
    upper bound at large sigma.
    """
    return amplitude * np.asarray(sigma, dtype=float) / composite_sigma


def return_period(sigma, skewness=0.0, n_years=2_000_000, rng=None,
                  period=ENSO_PERIOD_YEARS, damping=ENSO_DAMPING):
    """Modelled years between events at or above `sigma`.

    Read this as a diagnostic of the generator, not as a statement
    about the world.

    With 149 years of record containing perhaps two events near
    3.5 sigma, the empirical rate has a Poisson 95% interval
    spanning roughly one event per 20 years to one per 600. The
    tail is not identifiable from the data, so no skewness value
    here is calibrated -- it can only be shown to be more or less
    absurd.

    What the function is good for is exposing when a generator is
    assigning a return period to an event that is currently
    happening.
    """
    rng = np.random.default_rng(7) if rng is None else rng
    series = enso_index(int(n_years), rng, period, damping, skewness)
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))

    counts = np.array([(series >= s).sum() for s in sigma], dtype=float)
    with np.errstate(divide='ignore'):
        periods = np.where(counts > 0, n_years / np.maximum(counts, 1e-12),
                           np.inf)
    return periods[0] if periods.size == 1 else periods


def event_pulse(n_years, peak_sigma, onset_years=1.0, decay_years=1.5,
                start_year=None):
    """A single named event as a forcing pulse, in sigma.

    Extreme El Nino events are projected to show faster onset and
    slower decay than the historical average, so onset and decay
    are separate arguments rather than one width. The default 1
    year up and 1.5 years down is a reasonable shape for a strong
    event and is not fitted to any particular one.
    """
    n_years = int(n_years)
    start_year = n_years // 3 if start_year is None else int(start_year)
    t = np.arange(n_years, dtype=float) - start_year

    pulse = np.where(
        t < 0,
        np.exp(-(t / max(onset_years, 1e-9)) ** 2),
        np.exp(-(t / max(decay_years, 1e-9)) ** 2))
    return peak_sigma * pulse


def spectral_peak(series, dt=1.0):
    """Dominant period of a series, in the same units as dt.

    Used to check that a generated index actually oscillates where
    it was asked to.
    """
    series = np.asarray(series, dtype=float)
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series * np.hanning(len(series)))) ** 2
    freqs = np.fft.rfftfreq(len(series), d=dt)

    power[0] = 0.0
    peak = freqs[int(np.argmax(power))]
    return np.inf if peak == 0 else 1.0 / peak
