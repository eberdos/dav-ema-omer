import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import compare

mode="TM"   #enter here what you are using, use LIP, TM for two mass withouth filter and total zmp costraint, 
                #TM_NoY for Y-LIP version, TM_ZMP for two mass with total ZMP costraint
if mode=='LIP':
  import ismpc_LIP as ismpc
  R_zval=1e4
elif mode=='TM':
  import ismpc_Best as ismpc
  R_zval=1e2
elif mode=='TM_ZMP':
  import ismpc_NewZMP as ismpc
  R_zval=1e2
else:
  import ismpc_YLIP as ismpc
  R_zval=1e4

import footstep_planner
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from logger import Logger


class Hrp4Controller(dart.gui.osg.WorldNode):

  def __init__(self, world, hrp4):
    super(Hrp4Controller, self).__init__(world)
    print("controller initialized")
    self.world = world
    self.hrp4  = hrp4
    self.time  = 0
    self._last_valid_zmp = np.zeros(3)
    self.params = {
        'g':               9.81,
        'h':               0.72,
        'foot_size':       0.1,
        'step_height':     0.02,
        'ss_duration':     70,
        'ds_duration':     30,
        'world_time_step': world.getTimeStep(),
        'first_swing':     'rfoot',
        'µ':               0.5,
        'N':               100,
        'dof':             self.hrp4.getNumDofs(),
        # two-mass parameters
        'body_mass':       self.hrp4.getMass() * 0.9,
        'swing_mass':      self.hrp4.getMass() * 0.1,
        'swing_height':    0.02,
        # real time variables
        'footstep_x_adapt': 0.05,
        'footstep_y_adapt': 0.03,
        'n_footstep_vars':        3,      # number of upcoming footstep variables
        'footstep_x_adapt':       0.05,   # ±5 cm max footstep adaptation in x
        'two_mass_cost_weight':   50.0,   # weight on two-mass cost (vs LIP cost=100)
        'footstep_reg_weight':    50.0,   # weight pulling Xf back toward nominal plan
    }
    self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])

    # robot links
    self.lsole = hrp4.getBodyNode('l_sole')
    self.rsole = hrp4.getBodyNode('r_sole')
    self.torso = hrp4.getBodyNode('torso')
    self.base  = hrp4.getBodyNode('body')

    for i in range(hrp4.getNumJoints()):
      joint = hrp4.getJoint(i)
      dim   = joint.getNumDofs()
      if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
      elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

    # initial configuration
    initial_configuration = {
        'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0.,
        'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3.,
        'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3.,
        'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25.,
        'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.
    }
    for joint_name, value in initial_configuration.items():
      self.hrp4.setPosition(
          self.hrp4.getDof(joint_name).getIndexInSkeleton(),
          value * np.pi / 180.
      )

    # position robot on ground
    lsole_pos = self.lsole.getTransform(
        withRespectTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World()
    ).translation()
    rsole_pos = self.rsole.getTransform(
        withRespectTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World()
    ).translation()
    self.hrp4.setPosition(3, -(lsole_pos[0] + rsole_pos[0]) / 2.)
    self.hrp4.setPosition(4, -(lsole_pos[1] + rsole_pos[1]) / 2.)
    self.hrp4.setPosition(5, -(lsole_pos[2] + rsole_pos[2]) / 2.)

    # initialize state
    self.initial = self.retrieve_state()
    self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot'
    self.desired = copy.deepcopy(self.initial)

    # selection matrix for redundant dofs
    redundant_dofs = [
        "NECK_Y", "NECK_P",
        "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P",
        "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"
    ]

    # initialize modules
    self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

    # reference = [(0.2, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10

    # For comparison
    #reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10
    #reference = [(0.12, 0., 0.2)] * 5 + [(0.12, 0., -0.1)] * 10 + [(0.12, 0., 0.)] * 10
    #reference = [(0.15, 0., 0.2)] * 5 + [(0.15, 0., -0.1)] * 10 + [(0.15, 0., 0.)] * 10
    #reference = [(0.18, 0., 0.2)] * 5 + [(0.18, 0., -0.1)] * 10 + [(0.18, 0., 0.)] * 10
    reference = [(0.2, 0., 0.2)] * 5 + [(0.2, 0., -0.1)] * 10 + [(0.2, 0., 0.)] * 10
    


    self.footstep_planner = footstep_planner.FootstepPlanner(
        reference,
        self.initial['lfoot']['pos'],
        self.initial['rfoot']['pos'],
        self.params
    )

    self.mpc = ismpc.Ismpc(self.initial, self.footstep_planner, self.params)

    self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
        self.initial, self.footstep_planner, self.params
    )

    # initialize kalman filter
    A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
    B = self.params['world_time_step'] * self.mpc.B_lip
    d = np.zeros(9)
    d[7] = -self.params['world_time_step'] * self.params['g']
    H = np.identity(3)
    Q = block_diag(1., 1., 1.)
    R = block_diag(1e1, 1e2, R_zval)
    P = np.identity(3)
    x = np.array([
        self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0],
        self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1],
        self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]
    ])
    self.kf = filter.KalmanFilter(
        block_diag(A, A, A),
        block_diag(B, B, B),
        d,
        block_diag(H, H, H),
        block_diag(Q, Q, Q),
        block_diag(R, R, R),
        block_diag(P, P, P),
        x
    )

    self.logger = Logger(self.initial)
    self.logger.initialize_plot(frequency=10)

  def customPreStep(self):
    print(self.time)

    # retrieve raw state from simulator
    self.current = self.retrieve_state()

    # salva ZMP grezzo PRIMA del Kalman (per il plot stile paper)
    raw_zmp = self.current['zmp']['pos'].copy()
    raw_zmp = self._last_valid_zmp.copy()
    
    # kalman filter
    u = np.array([
        self.desired['zmp']['vel'][0],
        self.desired['zmp']['vel'][1],
        self.desired['zmp']['vel'][2]
    ])
    self.kf.predict(u)
    x_flt, _ = self.kf.update(np.array([
        self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0],
        self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1],
        self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]
    ]))

    # aggiorna stato con output Kalman
    self.current['com']['pos'][0] = x_flt[0]
    self.current['com']['vel'][0] = x_flt[1]
    self.current['zmp']['pos'][0] = x_flt[2]
    self.current['com']['pos'][1] = x_flt[3]
    self.current['com']['vel'][1] = x_flt[4]
    self.current['zmp']['pos'][1] = x_flt[5]
    self.current['com']['pos'][2] = x_flt[6]
    self.current['com']['vel'][2] = x_flt[7]
    self.current['zmp']['pos'][2] = x_flt[8]

    # MPC
    lip_state, contact = self.mpc.solve(self.current, self.time)

    self.desired['com']['pos'] = lip_state['com']['pos']
    self.desired['com']['vel'] = lip_state['com']['vel']
    self.desired['com']['acc'] = lip_state['com']['acc']
    self.desired['zmp']['pos'] = lip_state['zmp']['pos']
    self.desired['zmp']['vel'] = lip_state['zmp']['vel']

    # foot trajectories
    feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
    for foot in ['lfoot', 'rfoot']:
      for key in ['pos', 'vel', 'acc']:
        self.desired[foot][key] = feet_trajectories[foot][key]

    # torso and base references
    for link in ['torso', 'base']:
      for key in ['pos', 'vel', 'acc']:
        self.desired[link][key] = (
            self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]
        ) / 2.

    # inverse dynamics
    commands = self.id.get_joint_torques(self.desired, self.current, contact)
    for i in range(self.params['dof'] - 6):
      self.hrp4.setCommand(i + 6, commands[i])

    # logging
    self.logger.log_data(self.current, self.desired)
    self.logger.log_zmp_total(self.mpc.lip_state['zmp']['pos_total'])  # predicted (2-mass)
    self.logger.log_zmp_raw(raw_zmp)                                    # measured (contact forces)
    # self.logger.update_plot(self.time)

    self.time += 1

  def retrieve_state(self):
    com_position      = self.hrp4.getCOM()
    torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(
        withRespectTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
    base_orientation  = get_rotvec(self.hrp4.getBodyNode('body').getTransform(
        withRespectTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

    l_foot_transform    = self.lsole.getTransform(
        withRespectTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())
    l_foot_orientation  = get_rotvec(l_foot_transform.rotation())
    l_foot_position     = l_foot_transform.translation()
    left_foot_pose      = np.hstack((l_foot_orientation, l_foot_position))

    r_foot_transform    = self.rsole.getTransform(
        withRespectTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())
    r_foot_orientation  = get_rotvec(r_foot_transform.rotation())
    r_foot_position     = r_foot_transform.translation()
    right_foot_pose     = np.hstack((r_foot_orientation, r_foot_position))

    com_velocity            = self.hrp4.getCOMLinearVelocity(
        relativeTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())
    torso_angular_velocity  = self.hrp4.getBodyNode('torso').getAngularVelocity(
        relativeTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())
    base_angular_velocity   = self.hrp4.getBodyNode('body').getAngularVelocity(
        relativeTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())
    l_foot_spatial_velocity = self.lsole.getSpatialVelocity(
        relativeTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())
    r_foot_spatial_velocity = self.rsole.getSpatialVelocity(
        relativeTo=dart.dynamics.Frame.World(),
        inCoordinatesOf=dart.dynamics.Frame.World())

    # contact forces
    force = np.zeros(3)
    for contact in self.world.getLastCollisionResult().getContacts():
      force += contact.force

    # ZMP from contact forces
    zmp    = np.zeros(3)
    zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / self.params['h'])
    for contact in self.world.getLastCollisionResult().getContacts():
      if contact.force[2] <= 0.1:
        continue
      zmp[0] += (contact.point[0] * contact.force[2] / force[2]
                 + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
      zmp[1] += (contact.point[1] * contact.force[2] / force[2]
                 + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])

    if force[2] <= 0.1:
    # durante SS senza contatto affidabile: usa piede di stance come ZMP
      midpoint = (l_foot_position + r_foot_position) / 2.
      zmp[0] = midpoint[0]
      zmp[1] = midpoint[1]
      zmp[2] = 0.0
    else:
      midpoint = (l_foot_position + r_foot_position) / 2.
      zmp[0] = np.clip(zmp[0], midpoint[0] - 0.3, midpoint[0] + 0.3)
      zmp[1] = np.clip(zmp[1], midpoint[1] - 0.3, midpoint[1] + 0.3)
      zmp[2] = np.clip(zmp[2], midpoint[2] - 0.3, midpoint[2] + 0.3)
      self._last_valid_zmp = zmp.copy()

    return {
        'lfoot': {'pos': left_foot_pose,       'vel': l_foot_spatial_velocity, 'acc': np.zeros(6)},
        'rfoot': {'pos': right_foot_pose,      'vel': r_foot_spatial_velocity, 'acc': np.zeros(6)},
        'com':   {'pos': com_position,         'vel': com_velocity,            'acc': np.zeros(3)},
        'torso': {'pos': torso_orientation,    'vel': torso_angular_velocity,  'acc': np.zeros(3)},
        'base':  {'pos': base_orientation,     'vel': base_angular_velocity,   'acc': np.zeros(3)},
        'joint': {'pos': self.hrp4.getPositions(), 'vel': self.hrp4.getVelocities(), 'acc': np.zeros(self.params['dof'])},
        'zmp':   {'pos': zmp,                  'vel': np.zeros(3),             'acc': np.zeros(3)}
    }


if __name__ == "__main__":
  world = dart.simulation.World()
  print(dart.__version__)
  urdfParser  = dart.utils.DartLoader()
  current_dir = os.path.dirname(os.path.abspath(__file__))
  hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
  ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
  world.addSkeleton(hrp4)
  world.addSkeleton(ground)
  world.setGravity([0, 0, -9.81])
  world.setTimeStep(0.01)

  default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
  for body in hrp4.getBodyNodes():
    if body.getMass() == 0.0:
      body.setMass(1e-8)
      body.setInertia(default_inertia)

  viewer = dart.gui.osg.Viewer()
  node   = Hrp4Controller(world, hrp4)

  viewer.addWorldNode(node)
  viewer.setUpViewInWindow(0, 0, 1280, 720)
  viewer.setCameraHomePosition([5., -1., 1.5], [1., 0., 0.5], [0., 0., 1.])

  i = 0
  while i < 2500:
    node.customPreStep()
    world.step()
    viewer.frame()
    i += 1

  print("yahu")
  if mode=="LIP":
    np.save('zmp_meas_lip.npy', np.array(node.logger.log_zmp_measured_raw))
    np.save('zmp_pred_lip.npy', np.array(node.logger.log_zmp_total_predicted))
    node.logger.save_plot(dt=world.getTimeStep(), filename='zmp_lip.png')
  elif mode=='TM_ZMP':
    np.save('zmp_meas_newzmp.npy',         np.array(node.logger.log_zmp_measured_raw))
    np.save('zmp_pred_twomass_newzmp.npy', np.array(node.logger.log_zmp_total_predicted))
    swing_ratio = node.params['swing_mass'] / node.hrp4.getMass()
    name = f'zmp_{int(swing_ratio * 100)}.png'
    node.logger.save_plot(dt=world.getTimeStep(), filename=name)
  elif mode=='TM':
    np.save('zmp_meas_tm.npy',         np.array(node.logger.log_zmp_measured_raw))
    np.save('zmp_pred_twomass.npy', np.array(node.logger.log_zmp_total_predicted))
    swing_ratio = node.params['swing_mass'] / node.hrp4.getMass()
    name = f'zmp_{int(swing_ratio * 100)}.png'
    node.logger.save_plot(dt=world.getTimeStep(), filename=name)
  elif mode=='TM_NoY':
    np.save('zmp_meas_NoY.npy',         np.array(node.logger.log_zmp_measured_raw))
    np.save('zmp_pred_twomass_NoY.npy', np.array(node.logger.log_zmp_total_predicted))
    swing_ratio = node.params['swing_mass'] / node.hrp4.getMass()
    name = f'zmp_{int(swing_ratio * 100)}.png'
    node.logger.save_plot(dt=world.getTimeStep(), filename=name)


  # ====================== #
  #    COMPARISON PLOTS    #
  # ====================== #
  comparison_mode = True   # ← True per abilitare, False per disabilitare

  if not comparison_mode and not compare.is_empty(mode):
    compare.reset_data()

  if comparison_mode:
      dt             = world.getTimeStep()
      com_log        = np.array(node.logger.log['current', 'com', 'pos'])
      total_distance = np.linalg.norm(com_log[-1, :2] - com_log[0, :2])
      total_time     = len(com_log) * dt
      velocity       = total_distance / total_time

      rmse = node.logger.compute_rmse(dt=dt)
      n    = compare.collect_data(velocity, rmse, mode)

      if compare.has_enough_data(mode):
        compare.plot_comparison()
      else:
        print(f"[compare] Ancora {compare.MIN_POINTS - n} run per completare il ciclo.")