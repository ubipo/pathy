"""Steering - calculates the throttle and steering inputs needed to stay on path
based on images from /paddy/mask and published the results to /paddy/steering.

"""

import os, sys, json
from datetime import datetime, timedelta

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class Steering(Node):
    def __init__(self, throttle):
        super().__init__('steering')
        self._throttle = throttle
        self.create_subscription(Image, '/paddy/mask', self._mask_handler, 10)
        self._steering_pub = self.create_publisher(String, '/paddy/steering', 10)
        self._last_dms_msg_time = 0.0 # epoch
        self._bridge = CvBridge()

    def run_forever(self):
        self.get_logger().info("Running...")
        rclpy.spin(self)

    def _mask_handler(self, msg):
        mask = self._bridge.imgmsg_to_cv2(msg)
        im_array = np.asarray(mask)
        row_y = int(0.85 * mask.shape[0])
        row = im_array[row_y]
        integraal = np.cumsum(row)
        path_center = np.argmax(integraal > max(integraal) / 2)
        image_center = len(row) / 2
        path_center_offset = path_center - image_center
        steer = path_center_offset / image_center
        # print(f"{row.shape=} {len(row)=} {len(integraal)=} {image_center=}, {path_center=}, {path_center_offset=}, {steer=}")
        out_msg_obj = {
            "throttle": 1,
            "steer": steer
        }
        out_msg = String(data=json.dumps(out_msg_obj))
        self._steering_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    throttle = 1 # full forward
    steering = Steering(throttle)
    steering.run_forever()
    steering.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
