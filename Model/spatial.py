"""Explicit spatial representation.

Replaces scalar `patch_autocorrelation` with a circumpolar
habitat lattice. Implements additions.md item 7.

Two things a scalar cannot express, both of which the
observations show:

Redistribution is not decline. Antarctic krill are declining
in the Atlantic-Bellingshausen sector while increasing in the
Ross-Pacific sector. A circumpolar scalar averages those into
a small net change and reports stability where the actual
signal is spatial reorganisation.

Connectivity collapse is emergent, not imposed. core.py's
`percolation_connectivity()` produces a threshold by writing
a sigmoid, which is assuming the answer. Here bond occupation
declines smoothly and the sharp transition falls out of the
lattice. That is the same claim made honestly.

Geometry: a rows x cols lattice, periodic in longitude,
partitioned into contiguous circumpolar sectors. Rows are
depth or latitude bands (shelf -> slope -> deep). Square
lattice bond percolation has a known critical probability of
0.5 in the infinite limit; finite lattices smear it. That
known value is what makes the emergent threshold checkable
rather than decorative.

Pure numpy, no graph library, consistent with the rest of the
repo.

Literature: Docs/literature.md sections 6 and 7,
Docs/structure.md.
"""

import numpy as np

# Circumpolar ordering, closing the ring through the
# Antarctic Peninsula (Bellingshausen adjoins Weddell).
SECTOR_NAMES = (
    'Weddell',
    'Indian',
    'West Pacific',
    'Ross',
    'Amundsen',
    'Bellingshausen',
)

# Per-sector mid-trophic supply trend, fraction of baseline
# per year.
#
# The SIGNS are grounded: krill decline in the
# Atlantic-Bellingshausen sector contrasts with increase in
# the Ross-Pacific sector, and in 2023 the Amundsen was the
# only region with substantially above-normal sea ice while
# the outer Weddell, Ross and entire Bellingshausen showed
# extreme lows.
#
# The MAGNITUDES are not grounded. They are illustrative
# values chosen to make sector divergence visible on a
# decadal plot. Do not read them as observed rates.
SECTOR_SUPPLY_TREND = {
    'Weddell': -0.006,
    'Indian': -0.002,
    'West Pacific': 0.002,
    'Ross': 0.004,
    'Amundsen': 0.001,
    'Bellingshausen': -0.008,
}

# Per-sector fragmentation rate, fraction of bonds lost per
# year. Same caveat: ordering is grounded, magnitudes are not.
SECTOR_FRAGMENTATION = {
    'Weddell': 0.010,
    'Indian': 0.006,
    'West Pacific': 0.005,
    'Ross': 0.006,
    'Amundsen': 0.004,
    'Bellingshausen': 0.014,
}


# ---------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------


def sector_of_column(col, cols, n_sectors=len(SECTOR_NAMES)):
    """Map a longitude column to a sector index."""
    return np.minimum((np.asarray(col) * n_sectors) // cols, n_sectors - 1)


def build_lattice(rows=6, cols=60, n_sectors=len(SECTOR_NAMES)):
    """Circumpolar habitat lattice.

    Nodes are indexed row-major: node = row * cols + col.
    Longitude wraps; depth does not.

    Returns (edges, edge_sector, node_sector, shape) where
    edges is an (E, 2) integer array and edge_sector gives the
    sector index of each bond.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            here = r * cols + c
            # longitudinal bond, wrapping at the date line
            edges.append((here, r * cols + (c + 1) % cols))
            # depth bond, no wrap
            if r + 1 < rows:
                edges.append((here, (r + 1) * cols + c))

    edges = np.asarray(edges, dtype=np.int64)
    edge_cols = edges[:, 0] % cols
    edge_sector = sector_of_column(edge_cols, cols, n_sectors)
    node_sector = sector_of_column(np.arange(rows * cols) % cols,
                                   cols, n_sectors)
    return edges, edge_sector, node_sector, (rows, cols)


# ---------------------------------------------------------
# Percolation
# ---------------------------------------------------------


def _components(n_nodes, edges):
    """Union-find connected components. Returns (labels, sizes)."""
    parent = np.arange(n_nodes)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:          # path compression
            parent[i], i = root, parent[i]
        return root

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    labels = np.array([find(i) for i in range(n_nodes)])
    _, labels = np.unique(labels, return_inverse=True)
    sizes = np.bincount(labels)
    return labels, sizes


def occupy(edges, p_by_edge, rng):
    """Keep each bond with its own occupation probability."""
    keep = rng.random(len(edges)) < np.asarray(p_by_edge)
    return edges[keep]


def giant_fraction(n_nodes, edges):
    """Fraction of the domain in the largest connected component.

    This is the operational measure of habitat continuity for a
    wide-ranging species: what matters is the size of the
    largest reachable region, not the mean of a scalar.
    """
    if len(edges) == 0:
        return 1.0 / n_nodes
    _, sizes = _components(n_nodes, edges)
    return float(sizes.max()) / n_nodes


def percolation_sweep(p_values, rows=6, cols=60, reps=8, rng=None):
    """Giant component fraction across uniform bond probability.

    The transition this traces is a property of the lattice, not
    of a chosen functional form. For a square lattice the bond
    percolation threshold is 0.5 in the infinite limit; a finite
    lattice rounds the corner and biases the apparent threshold
    upward.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    edges, _, _, (rows, cols) = build_lattice(rows, cols)
    n_nodes = rows * cols

    out = np.empty(len(p_values))
    for i, p in enumerate(p_values):
        trials = [giant_fraction(n_nodes, occupy(edges, np.full(len(edges), p), rng))
                  for _ in range(reps)]
        out[i] = float(np.mean(trials))
    return out


def critical_probability(p_values, giant, smooth=3):
    """Estimate the threshold as the steepest point of the curve.

    Sampling noise near the transition makes a bare gradient
    estimator jumpy, so the curve is lightly smoothed first.
    Set smooth <= 1 to disable.

    With a finite lattice the transition has real width, so any
    single number summarises a smeared feature rather than
    locating a sharp constant. Treat the returned value as
    accurate to roughly the sample spacing, not better.
    """
    p_values = np.asarray(p_values, dtype=float)
    giant = np.asarray(giant, dtype=float)

    if smooth > 1 and len(giant) > smooth:
        kernel = np.ones(int(smooth)) / float(int(smooth))
        giant = np.convolve(giant, kernel, mode='same')

    slope = np.gradient(giant, p_values)
    return float(p_values[int(np.argmax(slope))])


# ---------------------------------------------------------
# Forcing over time
# ---------------------------------------------------------


def bond_probability(t, initial=0.95, rates=None, sector_names=SECTOR_NAMES):
    """Per-sector bond occupation declining linearly in time.

    Deliberately smooth. If a sharp connectivity collapse
    appears downstream of this, the lattice produced it.
    """
    rates = SECTOR_FRAGMENTATION if rates is None else rates
    t = np.asarray(t, dtype=float)
    per_sector = np.array([rates[name] for name in sector_names])
    p = initial - np.multiply.outer(t, per_sector)
    return np.clip(p, 0.0, 1.0)


def sector_supply(t, trends=None, baseline=1.0, sector_names=SECTOR_NAMES):
    """Per-sector mid-trophic supply under divergent regional trends."""
    trends = SECTOR_SUPPLY_TREND if trends is None else trends
    t = np.asarray(t, dtype=float)
    per_sector = np.array([trends[name] for name in sector_names])
    return baseline * (1.0 + np.multiply.outer(t, per_sector))


def redistribution_index(supply, baseline=1.0, eps=1e-9):
    """Gross sector change divided by net circumpolar change.

    1.0 means every sector moved the same way — a scalar model
    loses nothing. Large values mean sectors are moving in
    opposite directions and the circumpolar mean is reporting
    stability that no sector is experiencing.

    This is the number that says whether the spatial layer is
    earning its cost on a given run.
    """
    anomaly = np.asarray(supply, dtype=float) - baseline
    gross = np.abs(anomaly).sum(axis=-1)
    net = np.abs(anomaly.sum(axis=-1))
    return gross / np.maximum(net, eps)


def connectivity_trajectory(years, rows=6, cols=60, initial=0.95,
                            rates=None, reps=4, rng=None):
    """Giant component fraction over time under sector fragmentation.

    Bonds decline linearly (`bond_probability`). Any threshold in
    the output is emergent.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    edges, edge_sector, _, (rows, cols) = build_lattice(rows, cols)
    n_nodes = rows * cols
    p_t = bond_probability(years, initial, rates)

    out = np.empty(len(np.atleast_1d(years)))
    for i, p_sectors in enumerate(np.atleast_2d(p_t)):
        p_edges = p_sectors[edge_sector]
        trials = [giant_fraction(n_nodes, occupy(edges, p_edges, rng))
                  for _ in range(reps)]
        out[i] = float(np.mean(trials))
    return out


def supports_home_range(giant_frac, required_fraction):
    """Whether the largest component can hold a wide-ranging animal.

    required_fraction is home range as a fraction of the domain.
    Replaces the scalar `patch_viability()` test in core.py with
    one that asks about a real connected region.
    """
    return np.asarray(giant_frac) >= required_fraction
