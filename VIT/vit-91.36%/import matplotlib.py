import matplotlib.pyplot as plt
import numpy as np

# Data for accuracy across models
models = ['Proposed Hybrid Model', 'Baseline CNN', 'Transformer Model', 'MobileNet V2']
accuracy = [91.6, 88.3, 89.5, 84.0]

x = np.arange(len(models))

plt.figure(figsize=(10,6))
bars = plt.bar(x, accuracy, color='#4E79A7', edgecolor='black')

plt.xticks(x, models, rotation=15, ha='right', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Figure 3: Accuracy Gains Across Models', fontsize=15, fontweight='bold')
plt.ylim(0,100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
