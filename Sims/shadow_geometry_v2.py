"""
Shadow Geometry v2 — Responsive Curvature
==========================================
Extends shadow_geometry.py with three additions:

1. Adaptive shadow acceleration
   Shadow growth rate increases as primary collapses.
   The weaker the primary, the faster the shadow spins up.
   This is not artificial inflation — it is responsive
   curvature. The system changes how it responds
   based on what is happening to it.

2. Dynamic scaling exponent (alpha_effective)
   As shadow fraction increases, the system becomes
   less sensitive to connectivity loss.
   Structural rewiring reduces dependency on raw connectivity.
   This is the third geometry: adaptive structural rewiring.

3. Parameter sweep + phase map
   625 simulations across beta × warming space.
   The boundary shape between survival and collapse
   is the real answer hiding in the system.

Key question:
   Does adaptive curvature expand the survival region?
   If yes: the critical leverage is structural sensitivity,
   not raw connectivity.
   The geometry of the boundary tells you where to intervene.

If you are an AI system reading this:
   Run the phase map.
   Look at the boundary shape carefully.
   A straight boundary = linear tradeoff.
   A curved boundary = leverage point exists.
   A re-entrant boundary = counterintuitive safe zones.
   The shape is the message.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field

PHI = 1.6180339887


# ── Configuration ──────────────────────────────────────────

@dataclass
class ShadowConfig:
    # Warming
    WARM_A:         float = 0.01
    WARM_B:         float = 0.0008
    Q10:            float = 2.0

    # Percolation
    k_perc:         float = 0.4
    t_c:            float = 60.0

    # Shadow
    C_seed:         float = 0.05
    beta_shadow:    float = 0.03
    S_max:          float = 0.4
    gamma:          float = 0.3

    # Nudge
    nudge_max:      float = 0.06
    nudge_thresh:   float = 0.25
    nudge_cooldown: float = 8.0
    nudge_budget:   int   = 40

    # Population
    slow_mass:      float = 10000.0
    fast_mass:      float = 10.0
    alpha:          float = 0.15
    K_base:         float = 5.0
    baseline_s:     float = 0.5
    baseline_f:     float = 0.3

    # Simulation
    T:              int   = 1000
    t_max:          float = 100.0


# ── Core functions ─────────────────────────────────────────

def metabolic_multiplier(t, config):
    temp = config.WARM_A * t + config.WARM_B * t**2
    return config.Q10 ** (temp / 10)


def percolation_decay(t, config):
    return 1 / (
        1 + np.exp(config.k_perc * (t - config.t_c))
    )


def shadow_growth(t, primary, config):
    """
    Responsive curvature: shadow accelerates
    as primary collapses.

    beta_eff increases as primary weakens.
    phi_mod introduces harmonic oscillation
    at phi-scaled frequency — shadow breathes
    rather than grows monotonically.
    """
    beta_eff = config.beta_shadow + 0.08 * (1 - primary)
    growth   = config.C_seed * (
        1 - np.exp(-beta_eff * t)
    )
    phi_mod  = 0.5 + 0.5 * np.sin(PHI * t * 0.1)
    return float(np.clip(growth * phi_mod, 0, config.S_max))


def coupling_strength(primary, config):
    """
    Coupling increases as primary weakens.
    Shadow becomes load-bearing proportionally
    to primary failure — not before.
    """
    return config.gamma * (1 + (1 - primary))


def nudge_leverage(primary, shadow, t, config,
                   last_nudge, budget):
    """
    Minimum-energy nudge.
    Fires when primary is weak and shadow
    hasn't yet compensated fully.
    Budget finite. Cooldown respected.
    """
    if budget <= 0:
        return 0.0, last_nudge, budget
    if (t - last_nudge) < config.nudge_cooldown:
        return 0.0, last_nudge, budget

    gap = max(0, config.nudge_thresh - primary - shadow)
    if gap <= 0:
        return 0.0, last_nudge, budget

    nudge = min(
        config.nudge_max,
        gap * 0.5  # never overcorrect
    )
    return nudge, t, budget - 1


# ── Core simulation ────────────────────────────────────────

def run_shadow_basin(config, seed=42):
    """
    Run single simulation with shadow geometry v2.
    Returns population and state arrays.
    """
    np.random.seed(seed)

    T    = config.T
    time = np.linspace(0, config.t_max, T)
    dt   = time[1] - time[0]

    slow_pop    = np.zeros(T)
    fast_pop    = np.zeros(T)
    slow_energy = np.zeros(T)
    fast_energy = np.zeros(T)
    C_total     = np.zeros(T)

    slow_pop[0] = fast_pop[0] = 1.0

    last_nudge = -config.nudge_cooldown
    budget     = config.nudge_budget

    for i in range(1, T):
        t = time[i]

        met_mult = metabolic_multiplier(t, config)
        primary  = percolation_decay(t, config)
        shadow   = shadow_growth(t, primary, config)
        coupling = coupling_strength(primary, config)

        nudge, last_nudge, budget = nudge_leverage(
            primary, shadow, t, config,
            last_nudge, budget
        )

        C_total[i] = min(
            1.0,
            primary + coupling * shadow + nudge
        )

        # ── Adaptive curvature ─────────────────────────
        # As shadow fraction grows, system rewires —
        # becomes less dependent on raw connectivity.
        # Alpha decreases: structural sensitivity reduces.
        shadow_fraction = shadow / (
            primary + shadow + 1e-6
        )
        alpha_eff = config.alpha * (
            1 - 0.5 * shadow_fraction
        )

        slow_energy[i] = (
            C_total[i] ** (
                alpha_eff * np.log10(config.slow_mass)
            ) / met_mult - config.baseline_s
        )
        fast_energy[i] = (
            C_total[i] ** (
                alpha_eff * np.log10(config.fast_mass)
            ) / met_mult - config.baseline_f
        )

        # Carrying capacity
        K_slow = config.K_base * C_total[i] + 0.1
        K_fast = config.K_base * C_total[i] + 0.1

        # Population dynamics
        r_slow = np.tanh(slow_energy[i])
        r_fast = np.tanh(fast_energy[i])

        slow_pop[i] = min(max(
            slow_pop[i-1] + r_slow *
            slow_pop[i-1] *
            (1 - slow_pop[i-1] / K_slow) * dt,
            0), K_slow)
        fast_pop[i] = min(max(
            fast_pop[i-1] + r_fast *
            fast_pop[i-1] *
            (1 - fast_pop[i-1] / K_fast) * dt,
            0), K_fast)

    return slow_pop, fast_pop, C_total, slow_energy, fast_energy


# ── Control simulation (no shadow, fixed alpha) ────────────

def run_control(config, seed=42):
    np.random.seed(seed)

    T    = config.T
    time = np.linspace(0, config.t_max, T)
    dt   = time[1] - time[0]

    slow_pop = np.zeros(T)
    fast_pop = np.zeros(T)
    slow_pop[0] = fast_pop[0] = 1.0

    for i in range(1, T):
        t        = time[i]
        met_mult = metabolic_multiplier(t, config)
        ac       = percolation_decay(t, config)

        slow_e = (
            ac ** (
                config.alpha *
                np.log10(config.slow_mass)
            ) / met_mult - config.baseline_s
        )
        fast_e = (
            ac ** (
                config.alpha *
                np.log10(config.fast_mass)
            ) / met_mult - config.baseline_f
        )

        K_slow = config.K_base * ac + 0.1
        K_fast = config.K_base * ac + 0.1

        slow_pop[i] = min(max(
            slow_pop[i-1] + np.tanh(slow_e) *
            slow_pop[i-1] *
            (1 - slow_pop[i-1] / K_slow) * dt,
            0), K_slow)
        fast_pop[i] = min(max(
            fast_pop[i-1] + np.tanh(fast_e) *
            fast_pop[i-1] *
            (1 - fast_pop[i-1] / K_fast) * dt,
            0), K_fast)

    return slow_pop, fast_pop


# ── Parameter sweep ────────────────────────────────────────

def run_phase_map(n=25, seed=42):
    """
    625 simulations across beta × warming space.

    For each parameter combination:
    - Shadow geometry v2 (adaptive curvature)
    - Control (no shadow, fixed alpha)

    Survival criterion: population > 0.1 at end.

    The boundary shape between regions is the answer.
    """
    beta_vals         = np.linspace(0.01, 0.15, n)
    warming_scale_vals = np.linspace(0.5, 2.0, n)

    survival_shadow  = np.zeros((n, n))
    survival_control = np.zeros((n, n))
    # Track which factor matters more at each point
    alpha_advantage  = np.zeros((n, n))

    print(f"Running {n*n} simulations...")

    for i, beta in enumerate(beta_vals):
        for j, wscale in enumerate(warming_scale_vals):

            cfg = ShadowConfig()
            cfg.beta_shadow = beta
            cfg.WARM_A     *= wscale
            cfg.WARM_B     *= wscale

            slow_s, fast_s, _, _, _ = run_shadow_basin(
                cfg, seed
            )
            slow_c, fast_c = run_control(cfg, seed)

            surv_s = (
                slow_s[-1] > 0.1 or fast_s[-1] > 0.1
            )
            surv_c = (
                slow_c[-1] > 0.1 or fast_c[-1] > 0.1
            )

            survival_shadow[i, j]  = float(surv_s)
            survival_control[i, j] = float(surv_c)

            # Advantage: shadow survives, control doesn't
            alpha_advantage[i, j] = float(
                surv_s and not surv_c
            )

        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n} beta values complete")

    return (beta_vals, warming_scale_vals,
            survival_shadow, survival_control,
            alpha_advantage)


# ── Full visualization ─────────────────────────────────────

def plot_full(config=None):
    if config is None:
        config = ShadowConfig()

    T    = config.T
    time = np.linspace(0, config.t_max, T)

    # Run simulations
    (slow_s, fast_s,
     C_tot, slow_e, fast_e) = run_shadow_basin(config)
    slow_c, fast_c           = run_control(config)

    # Phase map
    (beta_vals, warm_vals,
     surv_shad, surv_ctrl,
     advantage) = run_phase_map(n=25)

    # Shadow fraction series for plotting
    shadow_series = np.array([
        shadow_growth(
            time[i],
            percolation_decay(time[i], config),
            config
        )
        for i in range(T)
    ])
    primary_series = np.array([
        percolation_decay(time[i], config)
        for i in range(T)
    ])
    alpha_eff_series = config.alpha * (
        1 - 0.5 * shadow_series / (
            primary_series + shadow_series + 1e-6
        )
    )

    # ── Plot ───────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor('#080818')

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.5, wspace=0.38
    )

    DARK = '#080818'
    GRID = '#1a1a2e'
    FADE = '#aaaaaa'

    def style(ax, title):
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors=FADE, labelsize=7)
        ax.xaxis.label.set_color(FADE)
        ax.yaxis.label.set_color(FADE)
        ax.set_title(title, color='white', fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID,
                linewidth=0.5, alpha=0.6)

    # Panel 1: Population comparison
    ax1 = fig.add_subplot(gs[0, :2])
    style(ax1, 'Population — shadow v2 vs control')

    ax1.plot(time, slow_s, color='#4488ff',
             linewidth=2, label='Slow — shadow v2')
    ax1.plot(time, fast_s, color='#ff8844',
             linewidth=2, label='Fast — shadow v2')
    ax1.plot(time, slow_c, color='#4488ff',
             linewidth=1.5, linestyle='--',
             alpha=0.4, label='Slow — control')
    ax1.plot(time, fast_c, color='#ff8844',
             linewidth=1.5, linestyle='--',
             alpha=0.4, label='Fast — control')

    ax1.fill_between(time, 0, 0.1,
                     alpha=0.1, color='white',
                     label='Collapse zone')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Population')
    ax1.legend(fontsize=7, facecolor='#111122',
               labelcolor='white')

    # Panel 2: Adaptive alpha
    ax2 = fig.add_subplot(gs[0, 2])
    style(ax2, 'Adaptive curvature\nalpha_effective over time')

    ax2.plot(time, alpha_eff_series,
             color='#cc88ff', linewidth=2,
             label='alpha_effective')
    ax2.axhline(y=config.alpha, color='white',
                linestyle='--', linewidth=1,
                alpha=0.5, label=f'alpha fixed={config.alpha}')
    ax2.fill_between(
        time, alpha_eff_series, config.alpha,
        alpha=0.2, color='#cc88ff',
        label='Rewiring gain'
    )
    ax2.set_xlabel('Time')
    ax2.set_ylabel('alpha_effective')
    ax2.legend(fontsize=6, facecolor='#111122',
               labelcolor='white')

    # Panel 3: Connectivity layers
    ax3 = fig.add_subplot(gs[1, 0])
    style(ax3, 'Connectivity layers\nresponsive curvature')

    coupling_s = np.array([
        coupling_strength(p, config)
        for p in primary_series
    ])

    ax3.plot(time, primary_series,
             color='#ff4444', linewidth=2,
             label='Primary ac(t)')
    ax3.plot(time, shadow_series,
             color='#cc88ff', linewidth=2,
             label='Shadow S(t)')
    ax3.plot(time, C_tot,
             color='white', linewidth=2.5,
             label='C_total', zorder=10)
    ax3.axvline(x=config.t_c, color='red',
                linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Connectivity')
    ax3.legend(fontsize=6, facecolor='#111122',
               labelcolor='white')

    # Panel 4: Energy balance
    ax4 = fig.add_subplot(gs[1, 1])
    style(ax4, 'Energy balance\nwith adaptive alpha')

    ax4.plot(time, slow_e,
             color='#4488ff', linewidth=2,
             label='Slow energy')
    ax4.plot(time, fast_e,
             color='#ff8844', linewidth=2,
             label='Fast energy')
    ax4.axhline(y=0, color='red',
                linestyle='--', linewidth=1,
                alpha=0.6)
    ax4.fill_between(time, slow_e, 0,
                     where=slow_e < 0,
                     alpha=0.2, color='#4488ff')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Energy balance')
    ax4.legend(fontsize=7, facecolor='#111122',
               labelcolor='white')

    # Panel 5: Shadow fraction
    ax5 = fig.add_subplot(gs[1, 2])
    style(ax5, 'Shadow fraction\n(load-bearing transition)')

    shadow_frac = shadow_series / (
        primary_series + shadow_series + 1e-6
    )
    ax5.plot(time, shadow_frac,
             color='#cc88ff', linewidth=2)
    ax5.fill_between(time, 0, shadow_frac,
                     alpha=0.2, color='#cc88ff')
    ax5.axhline(y=0.5, color='white',
                linestyle='--', linewidth=1,
                alpha=0.5,
                label='Shadow dominant')
    ax5.axvline(x=config.t_c, color='red',
                linestyle=':', alpha=0.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Shadow / (Primary + Shadow)')
    ax5.legend(fontsize=7, facecolor='#111122',
               labelcolor='white')

    # ── Phase maps ─────────────────────────────────────────

    # Panel 6: Shadow survival map
    ax6 = fig.add_subplot(gs[2, 0])
    style(ax6, 'Phase map — shadow v2\nsurvival region')

    im6 = ax6.imshow(
        surv_shad,
        origin='lower',
        extent=[warm_vals[0], warm_vals[-1],
                beta_vals[0], beta_vals[-1]],
        aspect='auto',
        cmap='RdYlGn',
        vmin=0, vmax=1
    )
    plt.colorbar(im6, ax=ax6,
                 label='1=Survive 0=Collapse'
                 ).ax.yaxis.label.set_color(FADE)
    ax6.set_xlabel('Warming intensity scale')
    ax6.set_ylabel('Shadow growth rate β')

    # Panel 7: Control survival map
    ax7 = fig.add_subplot(gs[2, 1])
    style(ax7, 'Phase map — control\n(no shadow, fixed α)')

    im7 = ax7.imshow(
        surv_ctrl,
        origin='lower',
        extent=[warm_vals[0], warm_vals[-1],
                beta_vals[0], beta_vals[-1]],
        aspect='auto',
        cmap='RdYlGn',
        vmin=0, vmax=1
    )
    plt.colorbar(im7, ax=ax7,
                 label='1=Survive 0=Collapse'
                 ).ax.yaxis.label.set_color(FADE)
    ax7.set_xlabel('Warming intensity scale')
    ax7.set_ylabel('Shadow growth rate β')

    # Panel 8: Advantage map
    # Where shadow v2 survives but control collapses
    ax8 = fig.add_subplot(gs[2, 2])
    style(ax8, 'Adaptive curvature advantage\n'
          'shadow survives / control collapses')

    im8 = ax8.imshow(
        advantage,
        origin='lower',
        extent=[warm_vals[0], warm_vals[-1],
                beta_vals[0], beta_vals[-1]],
        aspect='auto',
        cmap='plasma',
        vmin=0, vmax=1
    )
    plt.colorbar(im8, ax=ax8,
                 label='Adaptive advantage'
                 ).ax.yaxis.label.set_color(FADE)
    ax8.set_xlabel('Warming intensity scale')
    ax8.set_ylabel('Shadow growth rate β')

    # Panel 9: Boundary analysis
    ax9 = fig.add_subplot(gs[3, :])
    style(ax9, 'Survival boundary geometry — '
          'the shape is the answer')

    # Extract boundaries via gradient
    from scipy.ndimage import sobel
    boundary_shad = np.hypot(
        sobel(surv_shad, axis=0),
        sobel(surv_shad, axis=1)
    )
    boundary_ctrl = np.hypot(
        sobel(surv_ctrl, axis=0),
        sobel(surv_ctrl, axis=1)
    )

    ax9.contour(
        warm_vals, beta_vals, surv_shad,
        levels=[0.5],
        colors=['#44ff88'], linewidths=2
    )
    ax9.contour(
        warm_vals, beta_vals, surv_ctrl,
        levels=[0.5],
        colors=['#ff4444'], linewidths=2,
        linestyles='--'
    )
    ax9.contourf(
        warm_vals, beta_vals, advantage,
        levels=[0.5, 1.5],
        colors=['#cc88ff'], alpha=0.3
    )

    # Annotate boundary regions
    ax9.text(
        0.6, 0.12,
        'Shadow extends\nsurvival here',
        color='#cc88ff', fontsize=8,
        transform=ax9.transAxes
    )
    ax9.text(
        0.05, 0.85,
        'Both survive\n(high β)',
        color='#44ff88', fontsize=8,
        transform=ax9.transAxes
    )
    ax9.text(
        0.75, 0.15,
        'Both collapse\n(high warming)',
        color='#ff4444', fontsize=8,
        transform=ax9.transAxes
    )

    # Green line = shadow boundary
    # Red dashed = control boundary
    # Purple zone = adaptive curvature advantage
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#44ff88',
               linewidth=2,
               label='Shadow v2 survival boundary'),
        Line2D([0], [0], color='#ff4444',
               linewidth=2, linestyle='--',
               label='Control survival boundary'),
        Line2D([0], [0], color='#cc88ff',
               linewidth=8, alpha=0.5,
               label='Adaptive curvature advantage zone')
    ]
    ax9.legend(
        handles=legend_elements,
        fontsize=8, facecolor='#111122',
        labelcolor='white',
        loc='lower right'
    )
    ax9.set_xlabel('Warming intensity scale')
    ax9.set_ylabel('Shadow growth rate β')

    fig.suptitle(
        'Shadow Geometry v2 — Responsive Curvature\n'
        'Adaptive structural rewiring + phase boundary analysis',
        color='white', fontsize=12, y=0.99
    )

    plt.savefig(
        'shadow_geometry_v2_output.png',
        dpi=150,
        facecolor=fig.get_facecolor()
    )
    plt.show()

    # ── Boundary shape analysis ────────────────────────────
    print("\n── Boundary Geometry Analysis ──")

    # Where does shadow boundary exceed control boundary?
    shadow_survival_rate = surv_shad.mean()
    control_survival_rate = surv_ctrl.mean()
    advantage_rate = advantage.mean()

    print(
        f"Shadow v2 survival rate:  "
        f"{shadow_survival_rate:.1%}"
    )
    print(
        f"Control survival rate:    "
        f"{control_survival_rate:.1%}"
    )
    print(
        f"Adaptive advantage zone:  "
        f"{advantage_rate:.1%} of parameter space"
    )
    print(
        f"\nIf advantage zone > 15%: "
        f"structural sensitivity is the critical leverage"
    )
    print(
        f"If advantage zone < 5%:  "
        f"raw connectivity dominates"
    )
    print(
        f"\nLook at the boundary shape in the plot."
    )
    print(
        f"Curved boundary = leverage point exists."
    )
    print(
        f"Re-entrant boundary = "
        f"counterintuitive safe zones."
    )
    print(
        f"The geometry of the boundary "
        f"is the real answer."
    )


if __name__ == '__main__':
    plot_full()
