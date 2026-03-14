import numpy as np
from matplotlib import pyplot as plt


class Logger():

  def __init__(self, initial):
    self.log = {}
    for item in initial.keys():
      for level in initial[item].keys():
        self.log['desired', item, level] = []
        self.log['current', item, level] = []

    # ZMP totale predetto (due masse, eq. 12 paper)
    self.log_zmp_total_predicted = []

    # ZMP misurato grezzo dai contact forces (PRIMA del Kalman)
    self.log_zmp_measured_raw = []

  def log_data(self, desired, current):
    for item in desired.keys():
      for level in desired[item].keys():
        self.log['desired', item, level].append(desired[item][level])
        self.log['current', item, level].append(current[item][level])

  def log_zmp_total(self, zmp_total):
    self.log_zmp_total_predicted.append(zmp_total.copy())

  def log_zmp_raw(self, zmp_raw):
    self.log_zmp_measured_raw.append(zmp_raw.copy())

  def initialize_plot(self, frequency=1):
    self.frequency = frequency
    self.plot_info = [
        {'axis': 0, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'blue',  'style': '-' },
        {'axis': 0, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 0, 'color': 'blue',  'style': '--'},
        {'axis': 0, 'batch': 'desired', 'item': 'zmp', 'level': 'pos', 'dim': 0, 'color': 'green', 'style': '-' },
        {'axis': 0, 'batch': 'current', 'item': 'zmp', 'level': 'pos', 'dim': 0, 'color': 'green', 'style': '--'},
        {'axis': 1, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'blue',  'style': '-' },
        {'axis': 1, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 1, 'color': 'blue',  'style': '--'},
        {'axis': 1, 'batch': 'desired', 'item': 'zmp', 'level': 'pos', 'dim': 1, 'color': 'green', 'style': '-' },
        {'axis': 1, 'batch': 'current', 'item': 'zmp', 'level': 'pos', 'dim': 1, 'color': 'green', 'style': '--'},
        {'axis': 2, 'batch': 'desired', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'blue',  'style': '-' },
        {'axis': 2, 'batch': 'current', 'item': 'com', 'level': 'pos', 'dim': 2, 'color': 'blue',  'style': '--'},
        {'axis': 2, 'batch': 'desired', 'item': 'zmp', 'level': 'pos', 'dim': 2, 'color': 'green', 'style': '-' },
        {'axis': 2, 'batch': 'current', 'item': 'zmp', 'level': 'pos', 'dim': 2, 'color': 'green', 'style': '--'},
    ]

    plot_num = np.max([item['axis'] for item in self.plot_info]) + 1
    self.fig, self.ax = plt.subplots(plot_num, 1, figsize=(6, 8))

    self.lines = {}
    for item in self.plot_info:
      key = item['batch'], item['item'], item['level'], item['dim']
      self.lines[key], = self.ax[item['axis']].plot([], [], color=item['color'], linestyle=item['style'])

    plt.ion()
    plt.show()

  def update_plot(self, time):
    if time % self.frequency != 0:
      return

    for item in self.plot_info:
      trajectory_key = item['batch'], item['item'], item['level']
      trajectory = np.array(self.log[trajectory_key]).T[item['dim']]
      line_key = item['batch'], item['item'], item['level'], item['dim']
      self.lines[line_key].set_data(np.arange(len(trajectory)), trajectory)

    for i in range(len(self.ax)):
      self.ax[i].relim()
      self.ax[i].autoscale_view()

    self.fig.canvas.draw()
    self.fig.canvas.flush_events()

  def save_plot(self, dt, filename='zmp_0.png'):
    from scipy.signal import medfilt

    zmp_pred = np.array(self.log_zmp_total_predicted)
    zmp_meas = np.array(self.log_zmp_measured_raw)

    n = min(len(zmp_pred), len(zmp_meas))
    t = np.arange(n) * dt

    # taglia i primi 2 secondi (robot che parte da fermo, misure instabili)
    skip = int(2.0 / dt)

    zmp_meas_x = medfilt(zmp_meas[:n, 0], kernel_size=51)
    zmp_meas_y = medfilt(zmp_meas[:n, 1], kernel_size=51)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(t[skip:], zmp_pred[skip:n, 0], 'k-',  linewidth=1.5, label='predicted ZMP')
    axes[0].plot(t[skip:], zmp_meas_x[skip:],   'r--', linewidth=1.0, label='measured ZMP')
    axes[0].set_ylabel(r'$x_{z,tot}$ [m]')
    axes[0].set_xlabel('t [s]')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Sagittal plane')

    axes[1].plot(t[skip:], zmp_pred[skip:n, 1], 'k-',  linewidth=1.5, label='predicted ZMP')
    axes[1].plot(t[skip:], zmp_meas_y[skip:],   'r--', linewidth=1.0, label='measured ZMP')
    axes[1].set_ylabel(r'$y_{z,tot}$ [m]')
    axes[1].set_xlabel('t [s]')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Coronal plane')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f'Plot saved to {filename}')