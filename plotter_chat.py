import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "LIP": {"pred": "zmp_pred_lip.npy", "meas": "zmp_meas_lip.npy"},
    "Two-mass\n(Best)": {"pred": "zmp_pred_twomass.npy", "meas": "zmp_meas_tm.npy"},
    "Two-mass\n(Filter)": {"pred": "zmp_pred_twomass_filt.npy", "meas": "zmp_meas_filt.npy"},
    "Two-mass\n(ZMP-tot)": {"pred": "zmp_pred_twomass_newzmp.npy", "meas": "zmp_meas_newzmp.npy"},
}

DT = 0.01
T_EVAL_START = 2.0
T_EVAL_END = 20.0

MEDFILT_MEAS = 51
MEDFILT_PRED = 15

OUTPUT_PNG = "zmp_comparison.png"
OUTPUT_PDF = "zmp_comparison.pdf"

DISPLAY_DPI = 120
SAVE_DPI = 300

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "LIP": "#2563EB",
    "Two-mass\n(Best)": "#16A34A",
    "Two-mass\n(Filter)": "#D97706",
    "Two-mass\n(ZMP-tot)": "#DC2626",
}

MEAS_COLOR = "#6B7280"

TICK_FS = 9
LABEL_FS = 10
TITLE_FS = 11
ANNOT_FS = 9

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_mode(pred_path, meas_path):
    p, m = Path(pred_path), Path(meas_path)
    if not p.exists() or not m.exists():
        print(f"[WARN] Missing {pred_path} or {meas_path}")
        return None
    return np.load(p), np.load(m)


def compute_stats(pred, meas, axis):
    n = min(len(pred), len(meas))
    i0 = int(T_EVAL_START / DT)
    i1 = int(T_EVAL_END / DT)

    p = medfilt(pred[:n, axis], MEDFILT_PRED)
    m = medfilt(meas[:n, axis], MEDFILT_MEAS)

    err = p[i0:i1] - m[i0:i1]
    return {
        "rmse": np.sqrt(np.mean(err**2)),
        "mae": np.mean(np.abs(err)),
        "p": p, "m": m,
        "t": np.arange(n) * DT,
        "i0": i0, "i1": i1
    }


def improvement(base, val):
    return (base - val) / base * 100


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

data = {}
stats = {}

for name, paths in MODES.items():
    res = load_mode(paths["pred"], paths["meas"])
    if res is None:
        continue
    pred, meas = res
    stats[name] = {
        "x": compute_stats(pred, meas, 0),
        "y": compute_stats(pred, meas, 1)
    }
    data[name] = True

if not data:
    sys.exit("No valid data.")

modes = list(data.keys())
lip = modes[0]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE (layout FIXATO)
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(5 * len(modes), 17), dpi=DISPLAY_DPI)

gs = gridspec.GridSpec(
    3, len(modes),
    height_ratios=[0.75, 1, 1],
    hspace=0.55,   # ← FIX spacing verticale
    wspace=0.35
)

# ─────────────────────────────────────────────────────────────────────────────
# TABLE (fix overlap)
# ─────────────────────────────────────────────────────────────────────────────

ax_table = fig.add_subplot(gs[0, :])
ax_table.axis("off")

rows = []
for m in modes:
    sx, sy = stats[m]["x"], stats[m]["y"]
    rows.append([
        m.replace("\n", " "),
        f"{sx['rmse']*100:.2f}",
        f"{sx['mae']*100:.2f}",
        f"{improvement(stats[lip]['x']['rmse'], sx['rmse']):+.1f}",
        f"{sy['rmse']*100:.2f}",
        f"{sy['mae']*100:.2f}",
        f"{improvement(stats[lip]['y']['rmse'], sy['rmse']):+.1f}",
    ])

table = ax_table.table(
    cellText=rows,
    colLabels=["Mode","RMSE_x","MAE_x","Δx","RMSE_y","MAE_y","Δy"],
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(ANNOT_FS)

# 🔥 FIX PRINCIPALE: altezza + padding celle
table.scale(1, 1.9)

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(0.5)
    cell.set_edgecolor("#CBD5E1")
    cell.PAD = 0.08   # ← spazio interno testo

    if row == 0:
        cell.set_facecolor("#1E293B")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#F1F5F9")

ax_table.set_title(
    f"ZMP Tracking Error ({T_EVAL_START}-{T_EVAL_END}s)",
    fontsize=TITLE_FS+2,
    fontweight="bold",
    pad=10
)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot(ax, m, axis):
    s = stats[m][axis]
    t = s["t"]
    sl = slice(s["i0"], s["i1"])
    c = MODE_COLORS[m]

    ax.plot(t[sl], s["m"][sl], '--', color=MEAS_COLOR, alpha=0.6)
    ax.plot(t[sl], s["p"][sl], color=c, linewidth=1.6)

    rmse = s["rmse"] * 100
    imp = improvement(stats[lip][axis]["rmse"], s["rmse"])

    ax.set_xlim(T_EVAL_START, T_EVAL_END)
    ax.set_xlabel("t [s]", fontsize=LABEL_FS)

    ax.annotate(
        f"{rmse:.2f} cm\n{imp:+.1f}%",
        xy=(0.97,0.93), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=ANNOT_FS,
        bbox=dict(fc="white", alpha=0.85, pad=0.3)
    )

for i, m in enumerate(modes):
    # X
    ax1 = fig.add_subplot(gs[1, i])
    plot(ax1, m, "x")
    ax1.set_title(m.replace("\n"," "), fontsize=TITLE_FS, color=MODE_COLORS[m])
    ax1.set_ylabel("ZMP x [m]", fontsize=LABEL_FS)

    # Y
    ax2 = fig.add_subplot(gs[2, i])
    plot(ax2, m, "y")
    ax2.set_ylabel("ZMP y [m]", fontsize=LABEL_FS)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

plt.savefig(OUTPUT_PNG, dpi=SAVE_DPI, bbox_inches="tight")
plt.savefig(OUTPUT_PDF, bbox_inches="tight")

print("Saved PNG and PDF")
plt.show()