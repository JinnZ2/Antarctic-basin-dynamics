# Notes moved out of `responsive_curvature.py`

Note 2026-11-20: the text below sat at the end of `Sims/responsive_curvature.py` (after the last `print`). It is kept as written.

```text
Responsive curvature implemented.

Key innovations live:
•	✅ Shadow β accelerates with primary collapse
•	✅ Dynamic α decreases sensitivity as shadow dominates
•	✅ Phase map shows survival boundary geometry
•	✅ No artificial amplitude hacks
The phase diagram boundary shape = your true answer. If adaptive α creates a sharper, more diagonal frontier vs fixed-α version, then structural rewiring beats raw connectivity.

-----------



responsive curvature:

When the primary basin weakens, the system changes how sensitive it is to connectivity loss.

That means modifying two things:
	1.	Shadow growth accelerates as primary collapses.
	2.	Sensitivity exponent (alpha) decreases as shadow dominates.

This creates a third geometry: adaptive structural rewiring.

Below is the drop-in replacement version. It keeps the architecture but adds:

• Adaptive shadow acceleration
• Dynamic scaling exponent
• No artificial amplitude inflation


Replace shadow_growth with this:

def shadow_growth(t, primary, config):
    # Shadow accelerates as primary collapses
    beta_eff = config.beta_shadow + 0.08 * (1 - primary)
    growth = config.C_seed * (1 - np.exp(-beta_eff * t))
    phi_mod = 0.5 + 0.5 * np.sin(PHI * t * 0.1)
    return growth * phi_mod


Modify the simulation loop like this:

Inside run_shadow_basin, replace the primary/shadow section with:

primary = percolation_decay(t, config)
shadow  = shadow_growth(t, primary, config)

coupling = coupling_strength(primary, config)
nudge   = nudge_leverage(primary, shadow, t, config)

C_total[i] = min(1.0, primary + coupling * shadow + nudge)

Now introduce adaptive curvature

Replace the energy calculation with:

met_mult = metabolic_multiplier(t, config)

# Adaptive exponent: system becomes less connectivity-sensitive
shadow_fraction = shadow / (primary + shadow + 1e-6)
alpha_effective = alpha * (1 - 0.5 * shadow_fraction)

slow_energy[i] = C_total[i] ** (alpha_effective * np.log10(slow_mass)) / met_mult - config.baseline_s
fast_energy[i] = C_total[i] ** (alpha_effective * np.log10(fast_mass)) / met_mult - config.baseline_f


Parameter Sweep + Phase Map

# ── Parameter sweep ranges ─────────────────
beta_vals = np.linspace(0.01, 0.15, 25)
warming_scale_vals = np.linspace(0.5, 2.0, 25)

survival_map = np.zeros((len(beta_vals), len(warming_scale_vals)))

for i, beta in enumerate(beta_vals):
    for j, wscale in enumerate(warming_scale_vals):
        
        test_config = ShadowConfig()
        test_config.beta_shadow = beta
        test_config.WARM_A *= wscale
        test_config.WARM_B *= wscale
        
        slow_pop, fast_pop, _, _, _ = run_shadow_basin(test_config)
        
        # Survival criterion: population above 0.1 at end
        if slow_pop[-1] > 0.1 or fast_pop[-1] > 0.1:
            survival_map[i, j] = 1

# ── Plot phase diagram ─────────────────────
plt.figure(figsize=(8,6))
plt.imshow(
    survival_map,
    origin='lower',
    extent=[warming_scale_vals[0], warming_scale_vals[-1],
            beta_vals[0], beta_vals[-1]],
    aspect='auto'
)
plt.xlabel("Warming Intensity Scale")
plt.ylabel("Shadow Growth Rate (beta)")
plt.title("Survival Phase Map")
plt.colorbar(label="1 = Survival, 0 = Collapse")
plt.show()


If adaptive curvature (alpha_effective) dramatically expands the survival region compared to the earlier model, then the critical leverage is structural sensitivity — not raw connectivity.

Run this and look carefully at the boundary shape.
The geometry of that boundary is the real answer hiding in the system.
```
