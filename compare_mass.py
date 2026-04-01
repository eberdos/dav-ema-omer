'''
Comparison between LIP and 2-mass model with increasing swing mass %.
Data is persisted across runs via JSON.
'''

import json
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE  = 'compare_mass_data.json'
MIN_POINTS = 5   # soglia minima per plottare

KEY_MAP = {'LIP': 'LIP', 'TM': 'TM', 'TM_ZMP': 'TM_ZMP', 'TM_NoY': 'TM_NoY'}

COLORS = {
    'LIP':    '#2563EB',
    'TM':     '#16A34A',
    'TM_ZMP': '#D97706',
    'TM_NoY': '#DC2626',
}
LABELS = {
    'LIP':    'LIP',
    'TM':     '2-Mass (TM-MPC cost only)',
    'TM_ZMP': '2-Mass (TM-all)',
    'TM_NoY': '2-Mass (TM on X only)',
}


def _load() -> dict:
    empty = {k: {'mass_pct': [], 'rmse': []} for k in KEY_MAP}
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        for k in KEY_MAP:
            if k not in data:
                data[k] = {'mass_pct': [], 'rmse': []}
        return data
    return empty


def _save(data: dict):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def reset_data(mode: str = None):
    data = _load()
    if mode is None:
        data = {k: {'mass_pct': [], 'rmse': []} for k in KEY_MAP}
    else:
        key = KEY_MAP.get(mode, 'TM')
        data[key] = {'mass_pct': [], 'rmse': []}
    _save(data)
    print(f"[compare_mass] Reset: {'tutto' if mode is None else key}.")


def collect_data(mass_pct: float, rmse: float, mode: str):
    key  = KEY_MAP.get(mode, 'TM')
    data = _load()

    if len(data[key]['mass_pct']) >= MIN_POINTS:
        print(f"[compare_mass] {key}: ciclo già completo ({MIN_POINTS} punti). "
              f"Resetta con compare_mass.reset_data('{mode}') per ricominciare.")
        return len(data[key]['mass_pct'])

    data[key]['mass_pct'].append(round(mass_pct, 1))
    data[key]['rmse'].append(round(rmse * 100, 4))
    _save(data)
    n = len(data[key]['mass_pct'])
    print(f"[compare_mass] {key} — punto #{n}/{MIN_POINTS}: "
          f"mass={mass_pct:.1f}%, rmse={rmse*100:.3f} cm")
    return n


def has_enough_data(mode: str, min_points: int = MIN_POINTS) -> bool:
    key  = KEY_MAP.get(mode, 'TM')
    data = _load()
    return len(data[key]['mass_pct']) >= min_points


def is_empty(mode: str = None) -> bool:
    data = _load()
    if mode is None:
        return all(len(data[k]['mass_pct']) == 0 for k in KEY_MAP)
    key = KEY_MAP.get(mode, 'TM')
    return len(data[key]['mass_pct']) == 0


def plot_comparison():
    """
    Plotta RMSE vs massa% per tutte le modalità con almeno MIN_POINTS punti.
    La LIP è una linea orizzontale (non dipende dalla massa).
    """
    data    = _load()
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for key in ['LIP', 'TM', 'TM_ZMP', 'TM_NoY']:
        d = data[key]
        if len(d['mass_pct']) < MIN_POINTS:
            print(f"[compare_mass] {key}: dati insufficienti "
                  f"({len(d['mass_pct'])}/{MIN_POINTS}), skip.")
            continue

        pairs       = sorted(zip(d['mass_pct'], d['rmse']))
        mass_pcts   = [p[0] for p in pairs]
        rmses       = [p[1] for p in pairs]

        if key == 'LIP':
            # LIP non dipende dalla massa: linea tratteggiata orizzontale
            # usa la media dei suoi valori (tutti dovrebbero essere uguali
            # se la velocità di riferimento è fissa)
            avg_rmse = sum(rmses) / len(rmses)
            ax.axhline(avg_rmse, color=COLORS[key], linestyle='--',
                       linewidth=1.5, label=f'{LABELS[key]} — {avg_rmse:.2f} cm')
        else:
            ax.plot(mass_pcts, rmses,
                    color=COLORS[key], marker='o', linewidth=1.8,
                    markersize=5, label=LABELS[key])

        plotted += 1

    if plotted == 0:
        print("[compare_mass] Nessuna serie ha abbastanza dati per plottare.")
        plt.close()
        return

    ax.set_xlabel('Swing mass (% total mass)', fontsize=12)
    ax.set_ylabel('RMSE (cm)', fontsize=12)
    ax.set_title('ZMP Tracking RMSE vs. Swing Mass %', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig('compare_mass_rmse_vs_mass.pdf', dpi=300)
    plt.show()
    print("[compare_mass] Plot salvato → compare_mass_rmse_vs_mass.png")