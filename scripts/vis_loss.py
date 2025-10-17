import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.animation import FuncAnimation
from sklearn.neighbors import KNeighborsClassifier
import os

# ---------- 1. Existing loss‑curve plot ----------
with open('data/losses.json') as f:
    losses = json.load(f)
cleaned_losses = [d for d in losses if d != "null"]

plt.figure()
plt.plot(cleaned_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('data/losses.png', format='png', dpi=150)
plt.close()

# ---------- 2. Train / Val scatter + line plots ----------
with open('data/params.json') as f:
    fit = json.load(f)


def make_plot(X_key, Y_key, png_name):
    """Scatter + regression line for a single split (train or val)."""
    X = np.asarray(fit[X_key], dtype=float).ravel()
    Y = np.asarray(fit[Y_key], dtype=float)

    slope, intercept = fit['params']  # params = [slope, intercept]

    # Span only this split’s X‑range to keep the visual tight
    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = slope * x_line + intercept

    plt.figure()
    plt.scatter(X, Y, label=f'{X_key[:-2]} data')  # drop "_X" suffix for label
    plt.plot(x_line, y_line, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'data/{png_name}', format='png', dpi=150)
    plt.close()


# Training set
# make_plot('X_Train', 'Y_Train', 'fit_train.png')

# Validation set
# make_plot('X_Val', 'Y_Val', 'fit_val.png')


# Load data
with open('data/xor_val.json') as f:
    data = json.load(f)

x = np.asarray(data['x'], dtype=float)
y_across_epochs = np.asarray(data['y_across_epochs'], dtype=float)

# === MP4 Animation ===

# Prepare plot
fig, ax = plt.subplots(figsize=(6, 6))
xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]


def update(epoch):
    ax.clear()
    y = y_across_epochs[epoch]

    # Fit k-NN classifier on current labels
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x, y)
    Z = knn.predict(grid_points).reshape(xx.shape)

    # Draw decision region
    ax.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)

    # Plot data points
    for label, color in zip([0.0, 1.0], ['blue', 'red']):
        mask = y == label
        ax.scatter(x[mask, 0], x[mask, 1], c=color, label=f'Class {int(label)}', edgecolors='k')

    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Epoch {epoch}')
    ax.grid(True)
    ax.legend()


# Create animation
ani = FuncAnimation(fig, update, frames=len(y_across_epochs), repeat=True)

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Save MP4 video (1 second per epoch)
FPS = 1
ani.save('data/xor_val.mp4', fps=FPS, dpi=150)

print("✅ MP4 saved to: data/xor_val.mp4")

# === Final Static Plot ===

y_final = y_across_epochs[-1]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y_final)
Z = knn.predict(grid_points).reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)

for label, color in zip([0.0, 1.0], ['blue', 'red']):
    mask = y_final == label
    plt.scatter(x[mask, 0], x[mask, 1], c=color, label=f'Class {int(label)}', edgecolors='k')

plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Final Classification (1: Red, 0: Blue)')
plt.legend()
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal')
plt.savefig('data/xor_val.png', format='png', dpi=150)

print("✅ Final plot saved to: data/xor_val.png")
