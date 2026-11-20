import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-6, 6, 400)
y = -x**2 + 0.045 * x**4

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, y, color="#5b7ea6", linewidth=1.8)

ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

spine_color = "#b0b0b0"
ax.spines["left"].set_color(spine_color)
ax.spines["bottom"].set_color(spine_color)
ax.tick_params(axis="both", colors="#666666", labelsize=9, length=3)

ax.minorticks_on()
ax.tick_params(axis="x", which="minor", bottom=True, top=False, length=2, color=spine_color)
ax.tick_params(axis="y", which="minor", left=True, right=False, length=2, color=spine_color)

plt.tight_layout()
plt.show()