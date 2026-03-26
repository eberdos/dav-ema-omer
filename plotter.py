"""
plot_zmp_comparison.py
======================
Robust comparison plot for LIP vs Two-Mass MPC variants.

Usage
-----
1. Edit MODES dict below with the actual file paths for each mode.
2. Run:  python plot_zmp_comparison.py

Output
------
- zmp_comparison.png   : side-by-side subplots (X and Y), one column per mode
- zmp_comparison_table.txt : plain-text statistics table
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import medfilt
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit paths here
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "LIP":             {"pred": "zmp_pred_lip.npy",       "meas": "zmp_meas_lip.npy"},
    "Two-mass\n(Best)":{"pred": "zmp_pred_twomass.npy",   "meas": "zmp_meas_tm.npy"},
    "Two-mass\n(Filter)":{"pred":"zmp_pred_twomass_filt.npy",  "meas": "zmp_meas_filt.npy"},
    "Two-mass\n(ZMP-tot)":{"pred":"zmp_pred_twomass_newzmp.npy","meas":"zmp_meas_newzmp.npy"},
}

DT            = 0.01          # simulation time step [s]
T_EVAL_START  = 2.0           # evaluation window start [s]  (skip transient)
T_EVAL_END    = 20.0          # evaluation window end   [s]
MEDFILT_MEAS  = 51            # median-filter kernel for raw measured ZMP
MEDFILT_PRED  = 15            # median-filter kernel for predicted ZMP
OUTPUT_PNG    = "zmp_comparison.png"
OUTPUT_TABLE  = "zmp_comparison_table.txt"

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

MODE_COLORS = {
    "LIP":              "#2563EB",   # blue
    "Two-mass\n(Best)": "#16A34A",   # green
    "Two-mass\n(Filter)":"#D97706",  # amber
    "Two-mass\n(ZMP-tot)":"#DC2626", # red
}
MEAS_COLOR  = "#6B7280"  # grey for measured ZMP
PRED_ALPHA  = 0.92
MEAS_ALPHA  = 0.55
SHADE_ALPHA = 0.12


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_mode(pred_path: str, meas_path: str):
    """Load pred and meas arrays; return (pred, meas) or None if files missing."""
    p, m = Path(pred_path), Path(meas_path)
    if not p.exists():
        print(f"  [WARN] missing: {pred_path}")
        return None
    if not m.exists():
        print(f"  [WARN] missing: {meas_path}")
        return None
    pred = np.load(str(p))
    meas = np.load(str(m))
    return pred, meas


def compute_stats(pred: np.ndarray, meas: np.ndarray, dt: float,
                  t_start: float, t_end: float,
                  axis: int = 0) -> dict:
    """
    Compute RMSE and MAE between filtered pred and meas on the evaluation window.

    Parameters
    ----------
    pred, meas : (N, 3) arrays, columns = [x, y, z]
    axis       : 0=x, 1=y
    """
    n      = min(len(pred), len(meas))
    i0     = int(t_start / dt)
    i1     = min(int(t_end / dt), n)

    p_raw  = pred[:n, axis]
    m_raw  = meas[:n, axis]

    p_filt = medfilt(p_raw, MEDFILT_PRED)
    m_filt = medfilt(m_raw, MEDFILT_MEAS)

    err    = p_filt[i0:i1] - m_filt[i0:i1]
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    mae    = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "mae": mae,
            "pred_filt": p_filt, "meas_filt": m_filt,
            "n": n, "t": np.arange(n) * dt,
            "i0": i0, "i1": i1}


def improvement_pct(base_rmse: float, other_rmse: float) -> float:
    """Positive = better than base, negative = worse."""
    return (base_rmse - other_rmse) / base_rmse * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data...")
data   = {}
stats  = {}

for mode_name, paths in MODES.items():
    result = load_mode(paths["pred"], paths["meas"])
    if result is None:
        continue
    pred, meas = result
    sx = compute_stats(pred, meas, DT, T_EVAL_START, T_EVAL_END, axis=0)
    sy = compute_stats(pred, meas, DT, T_EVAL_START, T_EVAL_END, axis=1)
    data[mode_name]  = {"pred": pred, "meas": meas}
    stats[mode_name] = {"x": sx, "y": sy}
    print(f"  {mode_name.replace(chr(10),' '):<22s}  "
          f"RMSE_x={sx['rmse']*100:.2f}cm  MAE_x={sx['mae']*100:.2f}cm  |  "
          f"RMSE_y={sy['rmse']*100:.2f}cm  MAE_y={sy['mae']*100:.2f}cm")

if not data:
    sys.exit("No valid data files found. Edit MODES paths and retry.")

mode_names = list(data.keys())
n_modes    = len(mode_names)
lip_key    = mode_names[0]           # first mode is the reference baseline


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

base_rmse_x = stats[lip_key]["x"]["rmse"]
base_rmse_y = stats[lip_key]["y"]["rmse"]

col_w = 14
header = (f"{'Mode':<22s}  "
          f"{'RMSE_x[cm]':>{col_w}}  {'MAE_x[cm]':>{col_w}}  "
          f"{'Δ_x[%]':>{col_w}}  "
          f"{'RMSE_y[cm]':>{col_w}}  {'MAE_y[cm]':>{col_w}}  "
          f"{'Δ_y[%]':>{col_w}}")
sep = "─" * len(header)

table_lines = [sep, header, sep]
for mn in mode_names:
    sx  = stats[mn]["x"]
    sy  = stats[mn]["y"]
    dx  = improvement_pct(base_rmse_x, sx["rmse"])
    dy  = improvement_pct(base_rmse_y, sy["rmse"])
    label = mn.replace("\n", " ")
    table_lines.append(
        f"{label:<22s}  "
        f"{sx['rmse']*100:>{col_w}.3f}  {sx['mae']*100:>{col_w}.3f}  "
        f"{dx:>{col_w}.1f}  "
        f"{sy['rmse']*100:>{col_w}.3f}  {sy['mae']*100:>{col_w}.3f}  "
        f"{dy:>{col_w}.1f}"
    )
table_lines.append(sep)
table_text = "\n".join(table_lines)
print("\n" + table_text)

with open(OUTPUT_TABLE, "w") as f:
    f.write("\n")
print(f"Table saved → {OUTPUT_TABLE}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

# Layout: 3 rows × n_modes columns
#   row 0 : statistics table (rendered as text)
#   row 1 : X-axis ZMP plots
#   row 2 : Y-axis ZMP plots

FIG_W      = max(5 * n_modes, 14)
FIG_H      = 13
TICK_FS    = 8
LABEL_FS   = 9
TITLE_FS   = 10
ANNOT_FS   = 8

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
})

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)
fig.patch.set_facecolor("#F8FAFC")

# GridSpec: row 0 = table, row 1-2 = plots
gs = gridspec.GridSpec(
    3, n_modes,
    figure=fig,
    height_ratios=[0.38, 1, 1],
    hspace=0.55,
    wspace=0.30,
    top=0.94, bottom=0.07, left=0.06, right=0.97,
)

# ── Row 0: statistics table (merged cells) ───────────────────────────────────
ax_table = fig.add_subplot(gs[0, :])
ax_table.axis("off")

col_labels = ["Mode", "RMSE_x\n[cm]", "MAE_x\n[cm]", "Δ_x\n[%]",
                       "RMSE_y\n[cm]", "MAE_y\n[cm]", "Δ_y\n[%]"]
table_rows = []
for mn in mode_names:
    sx  = stats[mn]["x"]
    sy  = stats[mn]["y"]
    dx  = improvement_pct(base_rmse_x, sx["rmse"])
    dy  = improvement_pct(base_rmse_y, sy["rmse"])
    label = mn.replace("\n", " ")
    table_rows.append([
        label,
        f"{sx['rmse']*100:.3f}",
        f"{sx['mae']*100:.3f}",
        f"{dx:+.1f}",
        f"{sy['rmse']*100:.3f}",
        f"{sy['mae']*100:.3f}",
        f"{dy:+.1f}",
    ])

tbl = ax_table.table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(ANNOT_FS + 0.5)
tbl.scale(1, 1.55)

# colour rows by mode, header dark
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#CBD5E1")
    cell.set_linewidth(0.5)
    if row == 0:
        cell.set_facecolor("#1E293B")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        mn     = mode_names[row - 1]
        color  = MODE_COLORS.get(mn, "#94A3B8")
        cell.set_facecolor(color + "22")  # very light tint
        if col == 3 or col == 6:
            # improvement column: green if positive, red if negative
            val = float(table_rows[row - 1][col].replace("+",""))
            cell.set_text_props(
                color="#16A34A" if val > 0 else ("#DC2626" if val < 0 else "#374151"),
                fontweight="bold",
            )

ax_table.set_title(
    f"ZMP Tracking Error Summary  (eval window {T_EVAL_START}–{T_EVAL_END}s | swing mass = 10%)",
    fontsize=TITLE_FS + 1, fontweight="bold", color="#1E293B", pad=6,
)


# ── Helper: single subplot ───────────────────────────────────────────────────

def plot_single(ax, mn, axis_idx, row_label):
    """Fill one subplot for mode `mn`, axis X or Y."""
    sx    = stats[mn][axis_idx]
    color = MODE_COLORS.get(mn, "#6B7280")
    t     = sx["t"]
    sl    = slice(sx["i0"], sx["i1"])

    # shaded error band
    ax.fill_between(
        t[sl],
        sx["meas_filt"][sl],
        sx["pred_filt"][sl],
        color=color, alpha=SHADE_ALPHA, linewidth=0,
        label="_nolegend_",
    )

    # measured ZMP (grey, dashed)
    ax.plot(
        t[sl], sx["meas_filt"][sl],
        color=MEAS_COLOR, lw=1.0, ls="--", alpha=MEAS_ALPHA,
        label="measured",
    )

    # predicted ZMP (mode colour, solid)
    ax.plot(
        t[sl], sx["pred_filt"][sl],
        color=color, lw=1.6, alpha=PRED_ALPHA,
        label="predicted",
    )

    # RMSE annotation box
    rmse_cm = sx["rmse"] * 100
    improv  = improvement_pct(
        stats[lip_key][axis_idx]["rmse"], sx["rmse"]
    )
    sign    = "+" if improv > 0 else ""
    ann_col = "#16A34A" if improv > 0 else ("#DC2626" if improv < 0 else "#374151")
    ax.annotate(
        f"RMSE = {rmse_cm:.2f} cm\n{sign}{improv:.1f}% vs LIP",
        xy=(0.97, 0.96), xycoords="axes fraction",
        ha="right", va="top",
        fontsize=ANNOT_FS,
        color=ann_col,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ann_col,
                  alpha=0.85, lw=0.8),
    )

    ax.set_xlim(T_EVAL_START, T_EVAL_END)
    ax.set_xlabel("t [s]", fontsize=LABEL_FS)
    ax.set_ylabel(f"$z_{{{'x' if axis_idx=='x' else 'y'},tot}}$ [m]",
                  fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)

    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(
            f"{'Sagittal' if axis_idx=='x' else 'Coronal'}  "
            f"$z_{{{'x' if axis_idx=='x' else 'y'},tot}}$ [m]",
            fontsize=LABEL_FS,
        )
    else:
        ax.set_ylabel("")

    # light legend only on first row / last column
    if ax.get_subplotspec().is_last_col():
        ax.legend(fontsize=ANNOT_FS - 0.5, loc="upper left",
                  framealpha=0.8, edgecolor="#CBD5E1")


# ── Row 1 & 2: X and Y plots ─────────────────────────────────────────────────

for col_idx, mn in enumerate(mode_names):
    color = MODE_COLORS.get(mn, "#6B7280")
    label = mn.replace("\n", " ")

    # ---- X (sagittal) ----
    ax_x = fig.add_subplot(gs[1, col_idx])
    ax_x.set_facecolor("#FFFFFF")
    plot_single(ax_x, mn, "x", "Sagittal")

    # column title
    ax_x.set_title(
        label,
        fontsize=TITLE_FS, fontweight="bold", color=color, pad=5,
    )

    # ---- Y (coronal) ----
    ax_y = fig.add_subplot(gs[2, col_idx])
    ax_y.set_facecolor("#FFFFFF")
    plot_single(ax_y, mn, "y", "Coronal")


# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "IS-MPC Two-Mass Model — ZMP Tracking: Predicted vs Measured",
    fontsize=TITLE_FS + 3, fontweight="bold", color="#0F172A", y=0.99,
)

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Figure saved → {OUTPUT_PNG}")
plt.show()