import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ──────────────────────────────────────────────────────────────────────
d = {"MNIST": 784, "Fashion MNIST": 784, "CIFAR-10": 3072}

data = {
    "MNIST": {
        "95": [21, 11, 40, 37, 35, 41, 27, 28, 37, 27],
        "99": [116, 74, 159, 148, 151, 156, 122, 132, 143, 120],
    },
    "Fashion MNIST": {
        "95": [3, 3, 2, 5, 2, 74, 5, 9, 10, 5],
        "99": [87, 34, 59, 93, 55, 254, 113, 102, 155, 92],
    },
    "CIFAR-10": {
        "95": [1, 9, 2, 6, 2, 6, 4, 6, 1, 7],
        "99": [50, 139, 78, 96, 86, 96, 132, 111, 54, 127],
    },
}

# Convert to % of ambient dimension
pct = {}
for ds, vals in data.items():
    dim = d[ds]
    p95 = np.array(vals["95"]) / dim * 100
    p99 = np.array(vals["99"]) / dim * 100
    pct[ds] = {"95": p95, "gap": p99 - p95}

# ── Layout ────────────────────────────────────────────────────────────────────
n_classes = 10
n_datasets = 3
class_labels = [f"Class {i}" for i in range(n_classes)]

bar_w = 0.22          # width of each dataset's bar
group_gap = 0.08      # extra gap between class groups
x = np.arange(n_classes) * (n_datasets * bar_w + group_gap)
offsets = np.array([-1, 0, 1]) * bar_w   # center the three bars per group

colors     = ["#378ADD", "#1D9E75", "#D85A30"]
alpha_gap  = 0.30

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

dataset_names = list(pct.keys())

for i, (ds, color) in enumerate(zip(dataset_names, colors)):
    xpos = x + offsets[i]
    p95  = pct[ds]["95"]
    gap  = pct[ds]["gap"]

    ax.bar(xpos, p95, width=bar_w, color=color,       label=f"{ds} — 95%")
    ax.bar(xpos, gap, width=bar_w, color=color,
           alpha=alpha_gap, bottom=p95, label=f"{ds} — gap to 99%")

# ── Axes & labels ─────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(class_labels, fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel("% of Singular Values", fontsize=16)
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

# ── Legend ────────────────────────────────────────────────────────────────────
solid_patches = [
    mpatches.Patch(color=c, label=rf"{ds} ($d={d[ds]})$")
    for ds, c in zip(dataset_names, colors)
]
gap_patch = mpatches.Patch(
    facecolor="gray", alpha=alpha_gap, label="Gap: 95% → 99%"
)
ax.legend(handles=solid_patches + [gap_patch],
          fontsize=14, frameon=False, loc="upper right")

ax.set_title(
    "Percentage of Singular Values Needed to Capture 95% / 99% of Frobenius Norm",
    fontsize=20, pad=12
)

plt.tight_layout()
plt.savefig("image_data_class_singular_value_plot.pdf", dpi=150, bbox_inches="tight")
plt.show()
