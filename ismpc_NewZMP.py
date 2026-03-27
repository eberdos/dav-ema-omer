#ismpc_NewZMP
import numpy as np
import casadi as cs


class Ismpc:

  def __init__(self, initial, footstep_planner, params):

    self.params = params
    self.N = params['N']
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']

    # two-mass model parameters
    self.M = params['body_mass']
    self.m = params['swing_mass']
    self.zm_max = params['swing_height']

    self.initial = initial
    self.footstep_planner = footstep_planner

    self.sigma_fun = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1)

    # -------------------------
    # LIP dynamics
    # -------------------------

    self.A_lip = np.array([
        [0, 1, 0],
        [self.eta**2, 0, -self.eta**2],
        [0, 0, 0]
    ])

    self.B_lip = np.array([
        [0],
        [0],
        [1]
    ])

    self.f = lambda x, u: cs.vertcat(
        self.A_lip @ x[0:3] + self.B_lip @ u[0],
        self.A_lip @ x[3:6] + self.B_lip @ u[1],
        self.A_lip @ x[6:9] + self.B_lip @ u[2] + np.array([0, -params['g'], 0]),
    )

    # -------------------------
    # OPTIMIZATION
    # -------------------------

    self.opt = cs.Opti('conic')

    p_opts = {"expand": True}
    s_opts = {"max_iter": 1000, "verbose": False}

    self.opt.solver("osqp", p_opts, s_opts)

    self.U = self.opt.variable(3, self.N)
    self.X = self.opt.variable(9, self.N + 1)

    self.x0_param        = self.opt.parameter(9)
    self.zmp_x_mid_param = self.opt.parameter(self.N)
    self.zmp_y_mid_param = self.opt.parameter(self.N)
    self.zmp_z_mid_param = self.opt.parameter(self.N)

    self.zmp_x_swing_param = self.opt.parameter(self.N)
    self.zmp_y_swing_param = self.opt.parameter(self.N)
    self.sigma_param        = self.opt.parameter(self.N)
    self.sigma_y_param = self.opt.parameter(self.N)

    # -------------------------
    # DYNAMICS CONSTRAINTS
    # -------------------------

    for i in range(self.N):
      self.opt.subject_to(
          self.X[:, i + 1] ==
          self.X[:, i] + self.delta * self.f(self.X[:, i], self.U[:, i])
      )

    # -------------------------
    # TWO-MASS TOTAL ZMP  (eq. 12, sagittal only)
    # -------------------------

    zmp_x_total = (1.0 / (1.0 + self.sigma_param)) * self.X[2, 1:].T \
                + (self.sigma_param / (1.0 + self.sigma_param)) * self.zmp_x_swing_param
    

    zmp_y_total = (1.0 / (1.0 + self.sigma_y_param)) * self.X[5, 1:].T \
                + (self.sigma_y_param / (1.0 + self.sigma_y_param)) * self.zmp_y_swing_param

    # -------------------------
    # COST FUNCTION
    #
    # BUG 3 FIX: original code had cost on x_z,M but constraint on x_z,tot.
    # These fight each other: cost pulls x_z,M to mc_x, constraint pushes
    # x_z,M away from mc_x to compensate swing. Larger sigma = larger conflict.
    #
    # Fix: cost on x_z,tot (sagittal). Now cost and constraint are consistent:
    # both want x_z,tot near mc_x. x_z,M is shifted by the optimizer to
    # (1+σ)*mc_x - σ*swing_x, which is ~1-2cm from mc_x for typical sigma —
    # well inside the ±5cm foot constraint → constraint never active → smooth
    # CoM reference identical in shape to LIP.
    #
    # y: kept as pure LIP. Applying sigma in y forces x_z,M,y far outside the
    # foot (≈ +16cm for stance at +10cm) → infeasible or coronal instability.
    # -------------------------

    cost = cs.sumsqr(self.U)
    cost += 200 * cs.sumsqr(zmp_x_total        - self.zmp_x_mid_param)  # two-mass (x)
    cost += 200 * cs.sumsqr(zmp_y_total    - self.zmp_y_mid_param)  # two-mass (y)
    cost += 200 * cs.sumsqr(self.X[8, 1:].T    - self.zmp_z_mid_param)

    self.opt.minimize(cost)

    # -------------------------
    # ZMP CONSTRAINTS
    #
    # BUG 3 FIX (continued): constraint on PRIMARY x_z,M (pure LIP), not x_z,tot.
    # This guarantees feasibility regardless of swing foot position.
    # The soft cost above handles the two-mass correction without hard constraints.
    # -------------------------

    '''
    self.opt.subject_to(self.X[2, 1:].T <= self.zmp_x_mid_param + self.foot_size / 2)
    self.opt.subject_to(self.X[2, 1:].T >= self.zmp_x_mid_param - self.foot_size / 2)
  
    self.opt.subject_to(self.X[5, 1:].T <= self.zmp_y_mid_param + self.foot_size / 2)
    self.opt.subject_to(self.X[5, 1:].T >= self.zmp_y_mid_param - self.foot_size / 2)
    '''

    # zmp_x constraint
    self.opt.subject_to(zmp_x_total <= self.zmp_x_mid_param + self.foot_size / 2)
    self.opt.subject_to(zmp_x_total >= self.zmp_x_mid_param - self.foot_size / 2)

    # zmp_y constraint
    self.opt.subject_to(zmp_y_total <= self.zmp_y_mid_param + self.foot_size / 2)
    self.opt.subject_to(zmp_y_total >= self.zmp_y_mid_param - self.foot_size / 2)

    # -------------------------
    # INITIAL CONDITION
    # -------------------------

    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    # -------------------------
    # STABILITY CONSTRAINT
    # -------------------------

    self.opt.subject_to(
        self.X[1, 0] + self.eta * (self.X[0, 0] - self.X[2, 0]) ==
        self.X[1, self.N] + self.eta * (self.X[0, self.N] - self.X[2, self.N])
    )
    self.opt.subject_to(
        self.X[4, 0] + self.eta * (self.X[3, 0] - self.X[5, 0]) ==
        self.X[4, self.N] + self.eta * (self.X[3, self.N] - self.X[5, self.N])
    )
    self.opt.subject_to(
        self.X[7, 0] + self.eta * (self.X[6, 0] - self.X[8, 0]) ==
        self.X[7, self.N] + self.eta * (self.X[6, self.N] - self.X[8, self.N])
    )

    # -------------------------
    # STATE
    # -------------------------

    self.x = np.zeros(9)

    self.lip_state = {
        'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
        'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3), 'pos_total': np.zeros(3)}
    }

  # -------------------------------------------------
  # SWING FOOT MODEL  (parabolic, sagittal, paper eq.18)
  #
  # Returns xm (linear position), zm (parabolic height), ddzm (constant).
  # phase_time_s MUST be in SECONDS.
  # -------------------------------------------------

  def swing_foot_model(self, phase_time_s, step_length):

    T = self.params['ss_duration'] * self.delta   # seconds

    # clamp to valid domain
    t = np.clip(phase_time_s, 0.0, T)

    # linear horizontal position (ẍ_m=0 → inertia correction vanishes, x_z,m = x_m)
    xm   = step_length * (t / T)

    # parabolic height: zm(0)=zm(T)=0, peak at T/2
    zm   = -4.0 * self.zm_max * t * (t - T) / T**2

    # constant vertical acceleration (second derivative of parabola)
    ddzm = -8.0 * self.zm_max / T**2

    return xm, zm, ddzm

  # -------------------------------------------------
  # SOLVE MPC
  # -------------------------------------------------

  def solve(self, current, t):

    self.x = np.array([
        current['com']['pos'][0], current['com']['vel'][0], current['zmp']['pos'][0],
        current['com']['pos'][1], current['com']['vel'][1], current['zmp']['pos'][1],
        current['com']['pos'][2], current['com']['vel'][2], current['zmp']['pos'][2]
    ])

    mc_x, mc_y, mc_z = self.generate_moving_constraint(t)

    swing_x = np.zeros(self.N)
    swing_y = np.zeros(self.N)
    sigma   = np.zeros(self.N)

    plan = self.footstep_planner.plan
    n    = len(plan)

    for i in range(self.N):

      phase = self.footstep_planner.get_phase_at_time(t + i)
      
      step_idx = self.footstep_planner.get_step_index_at_time(t + i)
      fs_start_time = self.footstep_planner.get_start_time(step_idx)
      ds_start_time = fs_start_time + plan[step_idx]['ss_duration']
      if phase == 'ss':

        step_idx = self.footstep_planner.get_step_index_at_time(t + i)

        # BUG 1 FIX: phase_time must be in SECONDS.
        # Original: (t+i) - start_time  → discrete steps (0…69)
        # Correct:  multiply by delta to convert to seconds.
        phase_time_s = ((t + i) - self.footstep_planner.get_start_time(step_idx)) \
                       * self.delta

        # BUG 2 FIX: swing travels from liftoff (step_idx-1) to landing (step_idx+1).
        # Original:  stance (step_idx) → landing (step_idx+1), wrong origin and length.
        # Correct:   liftoff (step_idx-1) → landing (step_idx+1).
        if step_idx == 0:
          swing_start_x = mc_x[0]              # first step: use initial midfoot
          swing_start_y = mc_y[0]
        else:
          swing_start_x = plan[step_idx - 1]['pos'][0]
          swing_start_y = plan[step_idx - 1]['pos'][1]

        if step_idx + 1 < n:
          swing_end_x = plan[step_idx + 1]['pos'][0]
          swing_end_y = plan[step_idx + 1]['pos'][1]
        else:
          # last step: mirror step length forward
          swing_end_x = plan[step_idx]['pos'][0] \
                       + (plan[step_idx]['pos'][0] - swing_start_x)
          swing_end_y = plan[step_idx]['pos'][1] \
                 + (plan[step_idx]['pos'][1] - swing_start_y)

        step_length_x = swing_end_x - swing_start_x
        step_length_y = swing_end_y - swing_start_y

        xm, zm, ddzm = self.swing_foot_model(phase_time_s, step_length_x)

        # swing foot ZMP in world frame (= x_m since ẍ_m=0 for linear motion)
        swing_x[i] = swing_start_x + xm

        ## swing foot Y (lateral) is linear, no parabolic height or inertia correction
        T = self.params['ss_duration'] * self.delta
        tau = np.clip(phase_time_s / T, 0.0, 1.0)
        swing_y[i] = swing_start_y + tau * step_length_y

        # constant sigma for parabolic trajectory
        sigma[i] = np.clip(
            (self.m / self.M) * (ddzm + self.params['g']) / self.params['g'],
            0.0, 1.0
        )

      else:
        sigma[i]   = 0.0
        swing_x[i] = mc_x[i]
        swing_y[i] = mc_y[i]
      
      # sigma regularization, to avoid jump from 0 to anithing removes spikes in plot)
      alpha = self.sigma_fun(
      t + i,
      ds_start_time - 10,
      ds_start_time + 10    
      )

      sigma[i]   = alpha * sigma[i]

    sigma_y = 1.0 * sigma
    
    # set parameters
    self.opt.set_value(self.x0_param,          self.x)
    self.opt.set_value(self.zmp_x_mid_param,   mc_x)
    self.opt.set_value(self.zmp_y_mid_param,   mc_y)
    self.opt.set_value(self.zmp_z_mid_param,   mc_z)
    self.opt.set_value(self.zmp_x_swing_param, swing_x)
    self.opt.set_value(self.zmp_y_swing_param, swing_y)
    self.opt.set_value(self.sigma_param,       sigma)
    self.opt.set_value(self.sigma_y_param,    sigma_y)

    # solve
    sol = self.opt.solve()

    self.x = sol.value(self.X[:, 1])
    self.u = sol.value(self.U[:, 0])

    self.opt.set_initial(self.U, sol.value(self.U))
    self.opt.set_initial(self.X, sol.value(self.X))

    # output
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.x[6]])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], self.x[7]])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], self.x[8]])
    self.lip_state['zmp']['vel'] = self.u

    self.lip_state['com']['acc'] = self.eta**2 * (
        self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']
    ) + np.array([0, 0, -self.params['g']])

    # BUG 4 FIX: pos_total must use the OPTIMIZED next state (self.x, time t+1),
    # not x_current (pre-solve state at time t).
    # Also use sigma[1]/swing_x[1] for time alignment with the next state.
    sigma1   = sigma[0]   if self.N > 1 else sigma[0]
    sigma_y1 = sigma_y[0] if self.N > 1 else sigma_y[0]
    swing_x1 = swing_x[0] if self.N > 1 else swing_x[0]
    swing_y1 = swing_y[0] if self.N > 1 else swing_y[0]

    zmp_x_pred = (1.0 / (1.0 + sigma1)) * self.x[2] \
               + (sigma1 / (1.0 + sigma1)) * swing_x1 #x_z_tot
    
    zmp_y_pred = (1.0 / (1.0 + sigma_y1)) * self.x[5] \
           + (sigma_y1 / (1.0 + sigma_y1)) * swing_y1 #y_z_tot
    
    self.lip_state['zmp']['pos_total'] = np.array([zmp_x_pred, zmp_y_pred, self.x[8]])

    contact = self.footstep_planner.get_phase_at_time(t)

    if contact == 'ss':
      contact = plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

    return self.lip_state, contact

  # -------------------------------------------------
  # GENERATE MOVING CONSTRAINT
  # -------------------------------------------------

  def generate_moving_constraint(self, t):

    mc_x = np.full(self.N, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
    mc_y = np.full(self.N, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)

    plan      = self.footstep_planner.plan
    last_time = self.footstep_planner.get_start_time(len(plan) - 1)
    time_array = np.clip(np.array(range(t, t + self.N)), 0, last_time)

    for j in range(len(plan) - 1):
      fs_start_time  = self.footstep_planner.get_start_time(j)
      ds_start_time  = fs_start_time + plan[j]['ss_duration']
      fs_end_time    = ds_start_time + plan[j]['ds_duration']
      fs_current_pos = plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
      fs_target_pos  = plan[j + 1]['pos']
      mc_x += self.sigma_fun(time_array, ds_start_time, fs_end_time) * (fs_target_pos[0] - fs_current_pos[0])
      mc_y += self.sigma_fun(time_array, ds_start_time, fs_end_time) * (fs_target_pos[1] - fs_current_pos[1])

    return mc_x, mc_y, np.zeros(self.N)