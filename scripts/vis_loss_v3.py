import json
import matplotlib.pyplot as plt

# Load data
with open('data/losses_lri.json') as f:
    losses = json.load(f)

# Filter out None values
cleaned_losses = [(x, y) for x, y in zip(losses['losses'], losses['lri'])
                  if x is not None and y is not None]

# Unpack into X and Y
loss_vals, lri_vals = zip(*cleaned_losses)

# Plot
plt.figure()
plt.plot(lri_vals, loss_vals)
plt.xlabel('LRI')
plt.ylabel('LOSS')
plt.tight_layout()
plt.savefig('data/losses_lri.png', format='png', dpi=150)
plt.close()
