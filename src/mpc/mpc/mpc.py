import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
from .QP import QP

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleControlMode

class MPC(Node):
  def __init__(self):
    super().__init__('mpc')

    self.offboard_enabled = False

    self.declare_parameter("target_east", 0.0) #NEU frame
    self.east_ref = float(self.get_parameter("target_east").value) #NEU frame

    self.declare_parameter("target_north", 0.0) #NEU frame
    self.north_ref = float(self.get_parameter("target_north").value) #NEU frame

    self.declare_parameter("target_up", 10.0) #NEU frame
    self.down_ref = -float(self.get_parameter("target_up").value) #NED frame

    #current position/velocity, NEU frame
    self.north, self.east, self.down = 0.0, 0.0, 0.0
    self.v_north, self.v_east, self.v_down = 0.0, 0.0, 0.0

    self.position = np.array([self.north, self.east, self.down])
    self.velocity = np.array([self.v_north, self.v_east, self.v_down])

    #current pitch, roll, yaw
    self.pitch, self.roll, self.yaw = 0.0, 0.0, 0.0
    self.g = 9.81
    self.mpc_hover_thrust = 0.6 #hover thrust, set in ROMFS, NEU frame

    qos_profile_pub = QoSProfile(
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
      durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=0
    )

    qos_profile_sub = QoSProfile(
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
      durability=QoSDurabilityPolicy.VOLATILE,
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=0
    )

    self.subscriber_odometry = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.callback_odometry, qos_profile_sub)
    self.subscriber_vehicle_status = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.callback_vehicle_status, qos_profile_sub)
    self.subscriber_vehicle_control_mode = self.create_subscription(VehicleControlMode, '/fmu/out/vehicle_control_mode', self.callback_vehicle_control_mode, qos_profile_sub)

    self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
    self.publisher_trajectory_setpoint = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_pub)
    self.publisher_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile_pub)

    timer_period = 0.05 #sec, 50Hz
    self.dt = timer_period
    self.timer = self.create_timer(timer_period, self.callback_cmdloop)
    self.armed = VehicleStatus.ARMING_STATE_DISARMED #1, armed == 2, msg.arming_state
    self.N = 15 #horizon

    self.x = np.array([
      [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    ])

    self.x_ref = np.array([
      [self.north_ref], [self.east_ref], [self.down_ref],
      [0.0], [0.0], [0.0]
    ])

    self.x_max = np.array([
      [1000.0], [1000.0], [1000.0], [0.0], [0.0], [0.0]
    ])

    '''
    self.x_min = ([
      [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    ])
    '''
    self.x_min = -self.x_max

    self.qp_solver = QP(self.dt, self.N, self.x, self.x_ref, self.x_max, self.x_min)

  def callback_odometry(self, msg):
    qw, qx, qy, qz = msg.q
    self.roll = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx*qx + qy*qy))
    self.pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    self.yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy+qz*qz))

    self.north, self.east, self.down = msg.position
    self.position = np.array([
      self.north, self.east, self.down
    ])

    self.v_north, self.v_east, self.v_down = msg.velocity
    self.velocity = np.array([
      self.v_north, self.v_east, self.v_down
    ])

    self.x = np.hstack([self.position, self.velocity]).reshape(6, 1)

  def callback_vehicle_status(self, msg):
    self.armed = msg.arming_state

  def callback_vehicle_control_mode(self, msg):
    self.offboard_enabled = msg.flag_control_offboard_enabled

  def publish_offboard_mode(self):
    msg = OffboardControlMode()
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    msg.position = True
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = False
    self.publisher_offboard_mode.publish(msg)

  def publish_vehicle_command(self, command, param1, param2):
    msg = VehicleCommand()
    msg.param1 = param1
    msg.param2 = param2
    msg.command = command
    msg.from_external = True
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    self.publisher_vehicle_command.publish(msg)

  def publish_arm_command(self):
    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
    print("Arm command send")

  def publish_disarm_command(self):
    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 0.0)
    print("Disarm command send")

  def publish_trajectory_setpoint(self, pn, pe, pd):
    msg = TrajectorySetpoint()
    msg.position[0] = pn
    msg.position[1] = pe
    msg.position[2] = pd
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    self.publisher_trajectory_setpoint.publish(msg)

  def find_target_state(self):
    qp = QP(self.dt, self.N, self.x, self.x_ref, self.x_max, self.x_min) #set qp
    u_opt = qp.solve()
    u0 = u_opt[:3]
    return u0

  def acc_to_pos(self, u):
    v = self.velocity + u*self.dt
    p = self.position + v*self.dt
    return p

  def callback_cmdloop(self):
    if(self.armed == VehicleStatus.ARMING_STATE_DISARMED):
      self.publish_arm_command()
      self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

    self.publish_offboard_mode()
    u0 = self.find_target_state()
    u0[2] = u0[2] - self.g # account for gravity
    p = self.acc_to_pos(u0)
    self.publish_trajectory_setpoint(p[0], p[1], p[2])
    #TODO: add gravity as a disturbance term; v_dot = u + g

def main(args=None):
  rclpy.init(args=args)

  mpc = MPC()
  rclpy.spin(mpc)

  mpc.destroy_node()
  rclpy.shutdown()

if __name__ == 'main':
  main()
