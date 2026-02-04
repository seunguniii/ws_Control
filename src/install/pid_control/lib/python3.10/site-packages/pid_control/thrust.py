import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleControlMode

class PIDController(Node):
  def __init__(self):
    super().__init__('pid_controller')

    self.offboard_enabled = False

    self.declare_parameter("target_east", 0.0) #NEU frame
    self.east_ref = float(self.get_parameter("target_east").value) #NEU frame

    self.declare_parameter("target_north", 0.0) #NEU frame
    self.north_ref = float(self.get_parameter("target_north").value) #NEU frame

    self.declare_parameter("target_up", 10.0) #NEU frame
    self.up_ref = float(self.get_parameter("target_up").value) #NEU frame

    #current position/velocity, NEU frame
    self.north, self.east, self.up = 0.0, 0.0, 0.0
    self.v_north, self.v_east, self.v_up = 0.0, 0.0, 0.0

    #current pitch, roll, yaw
    self.pitch, self.roll, self.yaw = 0.0, 0.0, 0.0

    self.m = 2.064 #system mass, kg
    self.g = -9.81 #gravitational acc, m/s^2 #NEU frame
    self.mpc_hover_thrust = 0.6 #hover thrust, set in ROMFS, NEU frame

    self.kp_north = 2.0
    self.ki_north = 0.1*self.kp_north
    self.kd_north = 6.0

    self.kp_east = 2.0
    self.ki_east = 0.1*self.kp_east
    self.kd_east = 6.0

    self.kp_up = 9.0
    self.ki_up = 0.1*self.kp_up
    self.kd_up = 5.0

    self.e_north_integral = 0.0
    self.e_east_integral = 0.0
    self.e_up_integral = 0.0

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
    self.publisher_vehicle_attitude_setpoint = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile_pub)
    self.publisher_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile_pub)

    timer_period = 0.02 #sec, 50Hz
    self.dt = timer_period
    self.timer = self.create_timer(timer_period, self.callback_cmdloop)
    self.armed = VehicleStatus.ARMING_STATE_DISARMED #1, armed == 2, msg.arming_state

  def callback_odometry(self, msg):
    qw, qx, qy, qz = msg.q
    self.roll = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx*qx + qy*qy))
    self.pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1.0, 1.0))
    self.yaw = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy*qy+qz*qz))

    self.north, self.east, _ = msg.position
    self.up = -msg.position[2] #NED frame to NEU frame

    self.v_north, self.v_east, _ = msg.velocity
    self.v_up = -msg.velocity[2] #NED frame to NEU frame

  def callback_vehicle_status(self, msg):
    self.armed = msg.arming_state

  def callback_vehicle_control_mode(self, msg):
    self.offboard_enabled = msg.flag_control_offboard_enabled

  def publish_offboard_mode(self):
    msg = OffboardControlMode()
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    msg.position = False
    msg.velocity = False
    msg.acceleration = False
    msg.attitude = True
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

  def publish_vehicle_attitude(self, qw, qx, qy, qz, thrust):
    msg = VehicleAttitudeSetpoint()
    msg.q_d = qw, qx, qy, qz
    msg.thrust_body = 0.0, 0.0, thrust
    msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    self.publisher_vehicle_attitude_setpoint.publish(msg)

  def find_target_state(self):
    e_north, e_east, e_up = self.north_ref - self.north, self.east_ref - self.east, self.up_ref - self.up
    e_north_dot, e_east_dot, e_up_dot = -self.v_north, -self.v_east, -self.v_up

    self.e_north_integral += e_north*self.dt
    self.e_east_integral += e_east*self.dt
    self.e_up_integral += e_up*self.dt

    u_north = self.kp_north*e_north + self.ki_north*self.e_north_integral + self.kd_north*e_north_dot
    u_east = self.kp_east*e_east + self.ki_east*self.e_east_integral + self.kd_east*e_east_dot
    u_up = self.kp_up*e_up + self.ki_up*self.e_up_integral + self.kd_up*e_up_dot

    u_north_rotated = np.clip(np.cos(self.yaw)*u_north + np.sin(self.yaw)*u_east, -2.0, 2.0)
    u_east_rotated = np.clip(-np.sin(self.yaw)*u_east + np.cos(self.yaw)*u_east, -2.0, 2.0)

    pitch = np.clip(u_north_rotated/(self.g-u_up), -0.25, 0.25)
    roll = np.clip(u_east_rotated/(u_up-self.g), -0.25, 0.25)

    if e_north*e_north + e_east*e_east > 1:
      yaw = np.arctan2(e_east, e_north) #look towards the next setpoint when far enough (>1 m)
    else:
       yaw = self.yaw

    thrust = self.mpc_hover_thrust*(1+u_up/self.g)
    thrust = max(thrust, -1.0)/np.cos(self.pitch)/np.cos(self.roll)

    return roll, pitch, yaw, thrust

  def to_quaternion(self, roll, pitch, yaw):
    cosr = np.cos(roll*0.5)
    cosp = np.cos(pitch*0.5)
    cosy = np.cos(yaw*0.5)

    sinr = np.sin(roll*0.5)
    sinp = np.sin(pitch*0.5)
    siny = np.sin(yaw*0.5)

    qw = cosr*cosp*cosy + sinr*sinp*siny
    qx = sinr*cosp*cosy - cosr*sinp*siny
    qy = cosr*sinp*cosy + sinr*cosp*siny
    qz = cosr*cosp*siny - sinr*sinp*cosy

    return qw, qx, qy, qz

  def callback_cmdloop(self):
    if(self.armed == VehicleStatus.ARMING_STATE_DISARMED):
      self.publish_arm_command()
      return

    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
    self.publish_offboard_mode()

    roll, pitch, yaw, thrust = self.find_target_state()
    q = self.to_quaternion(roll, pitch, yaw)
    q /= np.linalg.norm(q) #normalize q
    self.publish_vehicle_attitude(q[0], q[1], q[2], q[3], thrust)

def main(args=None):
  rclpy.init(args=args)

  pid_controller = PIDController()
  rclpy.spin(pid_controller)

  pid_controller.destroy_node()
  rclpy.shutdown()

if __name__ == 'main':
  main()
