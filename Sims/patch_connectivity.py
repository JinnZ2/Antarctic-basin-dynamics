import numpy as np
import matplotlib.pyplot as plt
from core import patch_viability
import json

with open('../parameters.json') as f:
    P = json.load(f)

# Body mass range spanning small to megafaunal
body_masses = np.logspace(1, 5, 200)  # 10kg to 100,000kg

# Autocorrelation range
autocorrelation_values = np.linspace(0.1, 1.0, 200)

# Build viability surface
viability_surface = np.zeros(
    (len(autocorrelation_values), len(body_masses))
)

for i, ac in enumerate(autocorrelation_values):
    for j, bm in enumerate(body_masses):
        viability_surface[i, j] = float(patch_viability(
            ac, bm, 
            P['body_mass_scaling_exponent']
        ))

# Plot 1: viability surface as heatmap
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

im = ax1.contourf(
    np.log10(body_masses), 
    autocorrelation_values,
    viability_surface, 
    levels=20, 
    cmap='RdYlGn'
)
ax1.set_xlabel('Log body mass (kg)')
ax1.set_ylabel('Spatial autocorrelation')
ax1.set_title('Viability surface\n(green=viable, red=nonviable)')
plt.colorbar(im, ax=ax1)

# Mark approximate positions of reference species
reference_species = {
    'Sleeper shark': (2.3, 0.8),
    'Elephant': (4.0, 0.7),
    'Blue whale': (4.9, 0.6),
    'Large tuna': (2.0, 0.65)
}
for name, (logmass, ac) in reference_species.items():
    ax1.plot(logmass, ac, 'w^', markersize=8)
    ax1.annotate(name, (logmass, ac), 
                textcoords='offset points',
                xytext=(5, 5), color='white', fontsize=7)

# Plot 2: threshold curve for three body sizes
small_mass  = 100      # 100 kg
medium_mass = 5000     # 5,000 kg  
large_mass  = 50000    # 50,000 kg

for mass, label, color in [
    (small_mass,  '100 kg',    'green'),
    (medium_mass, '5,000 kg',  'orange'),
    (large_mass,  '50,000 kg', 'red')
]:
    viability_curve = [
        patch_viability(ac, mass, P['body_mass_scaling_exponent'])
        for ac in autocorrelation_values
    ]
    ax2.plot(autocorrelation_values, viability_curve, 
             color=color, label=label, linewidth=2)

ax2.set_xlabel('Spatial autocorrelation')
ax2.set_ylabel('Viable (1) / Nonviable (0)')
ax2.set_title('Connectivity threshold by body size')
ax2.legend()

# Plot 3: fragmentation over time with warming interaction
time_steps = np.linspace(0, 100, 200)  # years
base_autocorrelation = 0.80
fragmentation_rate = P['fragmentation_rate']

# Baseline fragmentation
ac_baseline = base_autocorrelation * np.exp(
    -fragmentation_rate * 0.005 * time_steps
)

# Fragmentation accelerated by warming
# Warming reduces ice extent, changes current patterns,
# introduces new boundaries
warming_fragmentation_multiplier = 1.8
ac_warmed = base_autocorrelation * np.exp(
    -fragmentation_rate * 0.005 * 
    warming_fragmentation_multiplier * time_steps
)

# Track viability for large-bodied species
viability_baseline = [
    patch_viability(ac, large_mass, P['body_mass_scaling_exponent'])
    for ac in ac_baseline
]
viability_warmed = [
    patch_viability(ac, large_mass, P['body_mass_scaling_exponent'])
    for ac in ac_warmed
]

ax3.plot(time_steps, ac_baseline, 
         'blue', label='Baseline fragmentation', linewidth=2)
ax3.plot(time_steps, ac_warmed, 
         'red', label='Warming-accelerated', linewidth=2, 
         linestyle='--')

# Find and mark threshold crossings
threshold = 0.5
for ac_series, color, label in [
    (ac_baseline, 'blue', 'baseline'),
    (ac_warmed,   'red',  'warmed')
]:
    crossings = np.where(np.diff(
        (np.array(ac_series) < threshold).astype(int)
    ))[0]
    if len(crossings) > 0:
        t_cross = time_steps[crossings[0]]
        ax3.axvline(x=t_cross, color=color, 
                   linestyle=':', alpha=0.7)
        ax3.annotate(f'Threshold\n~{t_cross:.0f} yrs',
                    (t_cross, 0.55), color=color, fontsize=8)

ax3.axhline(y=threshold, color='black', 
            linestyle='--', alpha=0.5, label='Viability threshold')
ax3.set_xlabel('Years')
ax3.set_ylabel('Spatial autocorrelation')
ax3.set_title('Threshold crossing:\nbaseline vs warming-accelerated')
ax3.legend()

plt.tight_layout()
plt.savefig('patch_connectivity_output.png', dpi=150)
plt.show()
