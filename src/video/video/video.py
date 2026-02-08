import rclpy
from rclpy.node import Node

import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from px4_msgs.msg import VehicleOdometry

class Video(Node):
  def __init__(self):
    super().__init__('video')

    self.subscriber_prediced_position = self.create_subscription(
      Float64MultiArray, '/predicted_position', self.callback_predicted_position, 10)
    self.subscriber_vehicle_odometry = self.create_subscription(
      VehicleOdometry, '/fmu/out/vehicle_odometry', self.callback_vehicle_odometry,
      qos_profile=rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST,depth=10)
    )
    self.q = [1.0, 0.0, 0.0, 0.0]
    self.position = [0.0, 0.0, 0.0]
    self.N = 15 #horizon of mpc
    self.target_matrix = np.zeros((self.N, 3))

    self._cap = self.open_camera()


    FPS = 30.0
    if self._cap is not None:
      self.timer = self.create_timer(1.0/FPS, self.callback_stream)
      self.get_logger().info("Stream initialized")


  def callback_predicted_position(self, msg):
    rows = msg.layout.dim[0].size
    cols = msg.layout.dim[1].size

    self.target_matrix = np.array(msg.data).reshape((rows, cols))

  def callback_vehicle_odometry(self, msg):
    self.position = msg.position
    self.q = msg.q
    self.euler = self.quaternion_to_euler(msg.q)

  def callback_stream(self):
    ret, frame = self._cap.read()

    if ret:
      self.draw_predicted_positions(frame)
      cv2.imshow("Drone View", frame)
      cv2.waitKey(1)
    else:
      self.get_logger().warn("Frame grab failed.")

  def quaternion_to_euler(self, q):
    w, x, y, z = q

    sinrXcosp = 2*(w*x + y*z)
    cosrXcosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinrXcosp, cosrXcosp)

    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
      pitch = math.copysign(math.pi/2, sinp)
    else:
      pitch = math.asin(sinp)

    sinyXcosp = 2*(w*z + x*y)
    cosyXcosp = 1- 2*(y*y + z*z)
    yaw = math.atan2(sinyXcosp, cosyXcosp)

    euler = [roll, pitch, yaw]
    return euler

  def rotate_coordinate(self, current_position, target_position, q):
    error = np.array(target_position) - np.array(current_position)
    w, x, y, z = q
    q = [x, y, z, w]
    rotation = R.from_quat(q)

    relative_target = rotation.inv().apply(error)

    return relative_target

  def stop(self):
    if self._cap:
      self._cap.release()
    cv2.destroyAllWindows()

  def open_camera(self):
    pipeline = (
      "udpsrc port=5600 ! "
      "application/x-rtp, encoding-name=H264 ! "
      "rtph264depay ! h264parse ! avdec_h264 ! "
      "videoconvert ! "
      "videoscale ! video/x-raw, width=640, height=480 ! "
      "appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
      self.get_logger().error("Could not open GStreamer pipeline")
      return None
    return cap

  def draw_predicted_positions(self, frame):
    size = 120
    length = 10
    origin = (320, 240)
    step = 1

    for i in range(self.N-1-step, 0, -step):
      relative_target_now = self.rotate_coordinate(self.position, self.target_matrix[i], self.q)
      relative_target_next = self.rotate_coordinate(self.position, self.target_matrix[i+step], self.q)
      screen_target_now = (int(origin[0] + relative_target_now[1]*length), int(origin[1] + relative_target_now[2]*length))
      screen_target_next = (int(origin[0] + relative_target_next[1]*length), int(origin[1] + relative_target_next[2]*length))

      #distance_to_target = np.linalg.norm(relative_target_now)

      with np.errstate(divide='ignore', invalid='ignore'): #suppress division by 0 warning/error; handled with min max functions
        try:
          size = 100/i
        except:
          size = 120

      start_x = int(max(screen_target_now[0] - size, 0))
      start_y = int(max(screen_target_now[1] - size, 0))
      end_x = int(min(screen_target_now[0] + size, 640))
      end_y = int(min(screen_target_now[1] + size, 480))
      
      cv2.rectangle(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        (255, 0, 0),
        2
      )
      
      '''
      cv2.putText(frame, str(i), (end_x, end_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

      cv2.drawMarker(frame, origin, (0, 0, 255))
      '''
      cv2.line(frame, screen_target_now, screen_target_next, (0, 0, 255), self.N-1-i)

def main(args=None):
  rclpy.init(args=args)

  video = Video()
  rclpy.spin(video)

  video.destroy_node()
  rclpy_shutdow()

if __name__ == 'main':
  main()

