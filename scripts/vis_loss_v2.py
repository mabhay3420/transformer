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
with open('data/losses_lri.json') as f:
    losses = json.load(f)
cleaned_losses = [d for d in losses["losses"] if d != "null"]

plt.figure()
plt.plot(cleaned_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('data/losses.png', format='png', dpi=150)
plt.close()