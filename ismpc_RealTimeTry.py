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
        self.M = params['body_mass']
        self.m = params['swing_mass']
        self.zm_max = params['swing_height']

        self.initial = initial
        self.footstep_planner = footstep_planner
        self.sigma_fun = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0), 0, 1)

        # LIP dynamics (x, y, z) – same as ismpc2
        self.A_lip = np.array([[0, 1, 0],
                               [self.eta**2, 0, -self.eta**2],
                               [0, 0, 0]])
        self.B_lip = np.array([[0], [0], [1]])

        self.f = lambda x, u: cs.vertcat(
            self.A_lip @ x[0:3] + self.B_lip @ u[0],
            self.A_lip @ x[3:6] + self.B_lip @ u[1],
            self.A_lip @ x[6:9] + self.B_lip @ u[2] + np.array([0, -params['g'], 0]),
        )

        # Optimization
        self.opt = cs.Opti('conic')
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000, "verbose": False}
        self.opt.solver("osqp", p_opts, s_opts)

        self.U = self.opt.variable(3, self.N)
        self.X = self.opt.variable(9, self.N + 1)

        # Parameters
        self.x0_param          = self.opt.parameter(9)
        self.zmp_x_mid_param   = self.opt.parameter(self.N)
        self.zmp_y_mid_param   = self.opt.parameter(self.N)
        self.zmp_z_mid_param   = self.opt.parameter(self.N)   # vertical ZMP reference (0 for constant height)
        self.zmp_x_swing_param = self.opt.parameter(self.N)
        self.zmp_y_swing_param = self.opt.parameter(self.N)
        self.sigma_param       = self.opt.parameter(self.N)

        # Velocity reference parameters (size N)
        self.vx_param = self.opt.parameter(self.N)
        self.vy_param = self.opt.parameter(self.N)
        self.k_vx = params.get('k_vx', 50.0)
        self.k_vy = params.get('k_vy', 50.0)

        # Dynamics constraints
        for i in range(self.N):
            self.opt.subject_to(
                self.X[:, i+1] == self.X[:, i] + self.delta * self.f(self.X[:, i], self.U[:, i])
            )

        # Total ZMP (Eq. 12)
        zmp_x_total = (1.0 / (1.0 + self.sigma_param)) * self.X[2, 1:].T \
                    + (self.sigma_param / (1.0 + self.sigma_param)) * self.zmp_x_swing_param
        zmp_y_total = (1.0 / (1.0 + self.sigma_param)) * self.X[5, 1:].T \
                    + (self.sigma_param / (1.0 + self.sigma_param)) * self.zmp_y_swing_param

        # Cost function (paper Eq. 23) plus vertical ZMP tracking
        cost  = cs.sumsqr(self.U[0, :])                              # (dx_z,M/dt)^2
        cost += self.k_vx * cs.sumsqr(self.X[1, 1:].T - self.vx_param)  # k_vx*(dx_M/dt - vx)^2
        cost += cs.sumsqr(self.U[1, :])                              # (dy_z,M/dt)^2
        cost += self.k_vy * cs.sumsqr(self.X[4, 1:].T - self.vy_param)  # k_vy*(dy_M/dt - vy)^2
        cost += cs.sumsqr(self.U[2, :])                              # vertical control effort
        cost += 200 * cs.sumsqr(self.X[8, 1:].T - self.zmp_z_mid_param)  # vertical ZMP tracking

        # Soft total ZMP constraints with slacks (high penalty)
        slack_x_plus  = self.opt.variable(self.N)
        slack_x_minus = self.opt.variable(self.N)
        slack_y_plus  = self.opt.variable(self.N)
        slack_y_minus = self.opt.variable(self.N)
        self.opt.subject_to(slack_x_plus  >= 0)
        self.opt.subject_to(slack_x_minus >= 0)
        self.opt.subject_to(slack_y_plus  >= 0)
        self.opt.subject_to(slack_y_minus >= 0)
        cost += 1e6 * cs.sumsqr(slack_x_plus  + slack_x_minus)
        cost += 1e6 * cs.sumsqr(slack_y_plus  + slack_y_minus)

        self.opt.subject_to(zmp_x_total <= self.zmp_x_mid_param + self.foot_size/2 + slack_x_plus)
        self.opt.subject_to(zmp_x_total >= self.zmp_x_mid_param - self.foot_size/2 - slack_x_minus)
        self.opt.subject_to(zmp_y_total <= self.zmp_y_mid_param + self.foot_size/2 + slack_y_plus)
        self.opt.subject_to(zmp_y_total >= self.zmp_y_mid_param - self.foot_size/2 - slack_y_minus)

        # Initial condition
        self.opt.subject_to(self.X[:, 0] == self.x0_param)

        # Stability constraints (endpoint equality) for x, y, and z (as in ismpc2)
        omega = self.eta
        self.opt.subject_to(
            self.X[1, 0] + omega * (self.X[0, 0] - self.X[2, 0]) ==
            self.X[1, self.N] + omega * (self.X[0, self.N] - self.X[2, self.N])
        )
        self.opt.subject_to(
            self.X[4, 0] + omega * (self.X[3, 0] - self.X[5, 0]) ==
            self.X[4, self.N] + omega * (self.X[3, self.N] - self.X[5, self.N])
        )
        # Vertical stability constraint – critical for constant height
        self.opt.subject_to(
            self.X[7, 0] + omega * (self.X[6, 0] - self.X[8, 0]) ==
            self.X[7, self.N] + omega * (self.X[6, self.N] - self.X[8, self.N])
        )

        # Internal state
        self.x = np.zeros(9)
        self.lip_state = {
            'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
            'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3), 'pos_total': np.zeros(3)}
        }

    def swing_foot_model(self, phase_time_s, step_length):
        T = self.params['ss_duration'] * self.delta
        t = np.clip(phase_time_s, 0.0, T)
        xm = step_length * (t / T)
        zm = -4.0 * self.zm_max * t * (t - T) / T**2
        ddzm = -8.0 * self.zm_max / T**2
        return xm, zm, ddzm

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
        n = len(plan)

        for i in range(self.N):
            phase = self.footstep_planner.get_phase_at_time(t + i)
            step_idx = self.footstep_planner.get_step_index_at_time(t + i)
            fs_start_time = self.footstep_planner.get_start_time(step_idx)
            ds_start_time = fs_start_time + plan[step_idx]['ss_duration']

            if phase == 'ss':
                phase_time_s = ((t + i) - fs_start_time) * self.delta
                if step_idx == 0:
                    swing_start_x = mc_x[0]
                    swing_start_y = mc_y[0]
                else:
                    swing_start_x = plan[step_idx - 1]['pos'][0]
                    swing_start_y = plan[step_idx - 1]['pos'][1]
                if step_idx + 1 < n:
                    swing_end_x = plan[step_idx + 1]['pos'][0]
                    swing_end_y = plan[step_idx + 1]['pos'][1]
                else:
                    swing_end_x = plan[step_idx]['pos'][0] + (plan[step_idx]['pos'][0] - swing_start_x)
                    swing_end_y = plan[step_idx]['pos'][1] + (plan[step_idx]['pos'][1] - swing_start_y)
                step_length   = swing_end_x - swing_start_x
                step_length_y = swing_end_y - swing_start_y
                T = self.params['ss_duration'] * self.delta
                tau = np.clip(phase_time_s / T, 0.0, 1.0)
                xm, zm, ddzm = self.swing_foot_model(phase_time_s, step_length)
                swing_x[i] = swing_start_x + xm
                swing_y[i] = swing_start_y + tau * step_length_y
                sigma[i] = np.clip((self.m / self.M) * (ddzm + self.params['g']) / self.params['g'], 0.0, 1.0)
            else:
                sigma[i] = 0.0
                swing_x[i] = mc_x[i]
                swing_y[i] = mc_y[i]

            # Smooth sigma transition
            alpha = self.sigma_fun(t + i, ds_start_time - 10, ds_start_time + 10)
            sigma[i] *= alpha

        # Set parameters
        self.opt.set_value(self.x0_param,          self.x)
        self.opt.set_value(self.zmp_x_mid_param,   mc_x)
        self.opt.set_value(self.zmp_y_mid_param,   mc_y)
        self.opt.set_value(self.zmp_z_mid_param,   mc_z)   # mc_z = 0
        self.opt.set_value(self.zmp_x_swing_param, swing_x)
        self.opt.set_value(self.zmp_y_swing_param, swing_y)
        self.opt.set_value(self.sigma_param,       sigma)

        # Velocity reference
        vx = self.params.get('vx', 0.3)
        vy = self.params.get('vy', 0.0)
        self.opt.set_value(self.vx_param, np.full(self.N, vx))
        self.opt.set_value(self.vy_param, np.full(self.N, vy))

        # Solve
        sol = self.opt.solve()
        self.x = sol.value(self.X[:, 1])
        self.u = sol.value(self.U[:, 0])

        self.opt.set_initial(self.U, sol.value(self.U))
        self.opt.set_initial(self.X, sol.value(self.X))

        # Output
        self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.x[6]])
        self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], self.x[7]])
        self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], self.x[8]])
        self.lip_state['zmp']['vel'] = self.u
        self.lip_state['com']['acc'] = self.eta**2 * (
            self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']
        ) + np.array([0, 0, -self.params['g']])

        # Total ZMP for logging
        sigma0 = sigma[0]
        swing_x0 = swing_x[0]
        swing_y0 = swing_y[0]
        zmp_x_pred = (1.0 / (1.0 + sigma0)) * self.x[2] + (sigma0 / (1.0 + sigma0)) * swing_x0
        zmp_y_pred = (1.0 / (1.0 + sigma0)) * self.x[5] + (sigma0 / (1.0 + sigma0)) * swing_y0
        self.lip_state['zmp']['pos_total'] = np.array([zmp_x_pred, zmp_y_pred, self.x[8]])

        contact = self.footstep_planner.get_phase_at_time(t)
        if contact == 'ss':
            contact = plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']

        return self.lip_state, contact

    def generate_moving_constraint(self, t):
        mc_x = np.full(self.N, (self.initial['lfoot']['pos'][3] + self.initial['rfoot']['pos'][3]) / 2.)
        mc_y = np.full(self.N, (self.initial['lfoot']['pos'][4] + self.initial['rfoot']['pos'][4]) / 2.)
        plan = self.footstep_planner.plan
        last_time = self.footstep_planner.get_start_time(len(plan) - 1)
        time_array = np.clip(np.arange(t, t + self.N), 0, last_time)
        for j in range(len(plan) - 1):
            fs_start_time = self.footstep_planner.get_start_time(j)
            ds_start_time = fs_start_time + plan[j]['ss_duration']
            fs_end_time   = ds_start_time + plan[j]['ds_duration']
            current_pos = plan[j]['pos'] if j > 0 else np.array([mc_x[0], mc_y[0]])
            target_pos  = plan[j + 1]['pos']
            weight = self.sigma_fun(time_array, ds_start_time, fs_end_time)
            mc_x += weight * (target_pos[0] - current_pos[0])
            mc_y += weight * (target_pos[1] - current_pos[1])
        return mc_x, mc_y, np.zeros(self.N)