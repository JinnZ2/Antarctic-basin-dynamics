import numpy as np
import matplotlib.pyplot as plt
from core import metabolic_multiplier, adjusted_lifespan
import json

with open('../parameters.json') as f:
    P = json.load(f)

warming_range = np.linspace(0, 6, 100)

apex_multipliers = [metabolic_multiplier(P['Q10_apex'], dT) 
                    for dT in warming_range]
mid_multipliers  = [metabolic_multiplier(P['Q10_midtrophic'], dT) 
                    for dT in warming_range]
micro_multipliers = [metabolic_multiplier(P['Q10_microbial'], dT) 
                     for dT in warming_range]

lifespans = [adjusted_lifespan(P['baseline_lifespan_years'], m) 
             for m in apex_multipliers]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(warming_range, apex_multipliers, label='Apex (Q10=2.5)')
ax1.plot(warming_range, mid_multipliers, label='Mid-trophic (Q10=3.0)')
ax1.plot(warming_range, micro_multipliers, label='Microbial (Q10=3.5)')
ax1.set_xlabel('Temperature increase (°C)')
ax1.set_ylabel('Metabolic rate multiplier')
ax1.set_title('Differential metabolic acceleration by trophic level')
ax1.legend()

ax2.plot(warming_range, lifespans, color='darkblue')
ax2.axhline(y=200, color='red', linestyle='--', 
            label='Approximate generation compression threshold')
ax2.set_xlabel('Temperature increase (°C)')
ax2.set_ylabel('Theoretical lifespan (years)')
ax2.set_title('Lifespan compression under warming')
ax2.legend()

plt.tight_layout()
plt.savefig('energy_basin_output.png', dpi=150)
plt.show()
