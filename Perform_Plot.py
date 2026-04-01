import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

dt = 0.01
t_start = 0.0
t_end   = 20.0
i_start = int(t_start / dt)
i_end   = int(t_end   / dt)

pred_lip     = np.load('zmp_pred_lip.npy')
pred_twomass = np.load('zmp_pred_twomass_NoY.npy')
meas_lip     = np.load('zmp_meas_lip.npy')
meas_twomass = np.load('zmp_meas_NoY.npy')

n = min(len(pred_lip), len(pred_twomass), len(meas_lip), len(meas_twomass))
t = np.arange(n) * dt
meas_lip_x     = medfilt(meas_lip[:n, 0],     51)
meas_twomass_x = medfilt(meas_twomass[:n, 0], 51)
pred_lip_x     = medfilt(pred_lip[:n, 0],     15)
pred_twomass_x = medfilt(pred_twomass[:n, 0], 15)

sl = slice(i_start, i_end)

# errore di stima nella finestra 5-10s
# RMSE/MAE
rmse_lip     = np.sqrt(np.mean((pred_lip_x[sl]     - meas_lip_x[sl]    )**2))
rmse_twomass = np.sqrt(np.mean((pred_twomass_x[sl] - meas_twomass_x[sl])**2))
mae_lip      = np.mean(np.abs(pred_lip_x[sl]     - meas_lip_x[sl]    ))
mae_twomass  = np.mean(np.abs(pred_twomass_x[sl] - meas_twomass_x[sl]))

print(f"{'':30s} {'LIP':>10s} {'Two-mass':>10s}")
print(f"{'RMSE [m]':30s} {rmse_lip:10.4f} {rmse_twomass:10.4f}")
print(f"{'MAE  [m]':30s} {mae_lip:10.4f} {mae_twomass:10.4f}")
print(f"{'Improvement RMSE':30s} {(rmse_lip - rmse_twomass)/rmse_lip*100:+.1f}%")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=300)

axes[0].plot(t[sl], meas_lip_x[sl],      'r--', lw=1.0, label='measured ZMP')
axes[0].plot(t[sl], pred_lip[sl, 0],     'b-',  lw=1.5, label='LIP predicted')
axes[0].set_ylabel(r'$x_{z,tot}$ [m]')
axes[0].set_xlabel('t [s]')
axes[0].set_xlim(t_start, t_end)
axes[0].set_title(f'LIP (0% swing mass) — RMSE={rmse_lip*100:.2f} cm, MAE={mae_lip*100:.2f} cm')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t[sl], meas_twomass_x[sl],    'r--', lw=1.0, label='measured ZMP')
axes[1].plot(t[sl], pred_twomass[sl, 0],   'k-',  lw=1.5, label='Two-mass predicted')
axes[1].set_ylabel(r'$x_{z,tot}$ [m]')
axes[1].set_xlabel('t [s]')
axes[1].set_xlim(t_start, t_end)
axes[1].set_title(f'Two-mass (30% swing mass) — RMSE={rmse_twomass*100:.2f} cm, MAE={mae_twomass*100:.2f} cm')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('comparison.png', dpi=700)