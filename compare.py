'''
Comparison between LIP and 2-mass model with increasing velocity.
Data is persisted across runs via JSON.
'''

import json
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE   = 'compare_data.json'
MIN_POINTS  = 5   # soglia minima per plottare

KEY_MAP = {'LIP': 'LIP', 'TM': 'TM', 'TM_ZMP': 'TM_ZMP'}

def _load() -> dict:
    empty = {k: {'vel': [], 'rmse': []} for k in KEY_MAP}
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        # aggiunge chiavi mancanti senza perdere i dati esistenti
        for k in KEY_MAP:
            if k not in data:
                data[k] = {'vel': [], 'rmse': []}
        return data
    return empty

def _save(data: dict):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def reset_data(mode: str = None):
    data = _load()
    if mode is None:
        data = {k: {'vel': [], 'rmse': []} for k in KEY_MAP}
    else:
        key = KEY_MAP.get(mode, 'TM')
        data[key] = {'vel': [], 'rmse': []}
    _save(data)
    print(f"[compare] Reset: {'tutto' if mode is None else key}.")


def collect_data(vel: float, rmse: float, mode: str):
    key  = KEY_MAP.get(mode, 'TM')
    data = _load()

    if len(data[key]['vel']) >= MIN_POINTS:
        print(f"[compare] {key}: ciclo già completo ({MIN_POINTS} punti). Resetta manualmente con compare.reset_data('{mode}') per ricominciare.")
        return len(data[key]['vel'])

    data[key]['vel'].append(round(vel, 4))
    data[key]['rmse'].append(round(rmse * 100, 4))
    _save(data)
    n = len(data[key]['vel'])
    print(f"[compare] {key} — punto #{n}/{MIN_POINTS}: vel={vel:.3f} m/s, rmse={rmse*100:.3f} cm")
    return n


def has_enough_data(mode: str, min_points: int = MIN_POINTS) -> bool:
    key  = KEY_MAP.get(mode, 'TM')
    data = _load()
    return len(data[key]['vel']) >= min_points


def is_empty(mode: str = None) -> bool:
    """Ritorna True se non ci sono dati per la modalità (o per entrambe se mode è None)."""
    data = _load()
    if mode is None:
        return all(len(data[k]['vel']) == 0 for k in ['LIP', 'TM_ZMP','TM'])
    key  = KEY_MAP.get(mode, 'TM')
    return len(data[key]['vel']) == 0



def plot_comparison():
    """
    Plotta RMSE vs velocità per tutte le modalità disponibili nello stesso grafico.
    Una serie viene plottata solo se ha raggiunto MIN_POINTS.
    """
    data = _load()
    plt.figure(figsize=(10, 6))
    plotted = 0

    for key, label in [('LIP', 'LIP'), ('TM_ZMP', '2-Mass, filter and total ZMP'),('TM', '2-Mass, No filter, total ZMP')]:
        d = data[key]
        if len(d['vel']) < MIN_POINTS:
            print(f"[compare] {key}: dati insufficienti ({len(d['vel'])}/{MIN_POINTS}), skip.")
            continue
        pairs = sorted(zip(d['vel'], d['rmse']))
        vels, rmses = zip(*pairs)
        plt.plot(vels, rmses, label=f'{label} RMSE', marker='o')
        plotted += 1

    if plotted == 0:
        print("[compare] Nessuna serie ha abbastanza dati per plottare.")
        plt.close()
        return

    plt.xlabel('Velocity (m/s)')
    plt.ylabel('RMSE (cm)')
    plt.title('ZMP Tracking RMSE vs. Average Velocity')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('compare_rmse_vs_vel.png', dpi=150)
    plt.show()