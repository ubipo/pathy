"""Camstream - streams webcam frames to /paddy/rgb on a regular interval.


"""

import os
import sys

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CamStream(Node):
    def __init__(self, video_source):
        super().__init__('camstream')
        self._source = video_source
        self._pub = self.create_publisher(Image, '/paddy/rgb', 10)
        timer_period = 0.5
        self.create_timer(timer_period, self._timer_callback)
        self._bridge = CvBridge()
        
    def init(self):
        self._init_cap()
        self.get_logger().info('Init OK')

    def run_forever(self):
        self.get_logger().info("Running...")
        rclpy.spin(self)

    def _timer_callback(self):
        frame = self._get_frame()
        my_msg = self._bridge.cv2_to_imgmsg(np.array(frame), "bgr8")
        self._pub.publish(my_msg)

    def _init_cap(self):
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise Exception(f"video source: {self._source} could not be opened")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fourcc = cv2.VideoWriter_fourcc(*'YUYV')
        ret = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if not ret:
            raise Exception(ret)

        self.get_logger().info('Capture init OK')
        self.cap = cap

    def _get_frame(self):
        ret = self.cap.grab()
        assert ret, "Capture grab"
        ret, raw_frame = self.cap.retrieve()
        assert ret, "Capture retrieve"
        return raw_frame


def main(args=None):
    rclpy.init(args=args)
    video_source = 0 # /dev/video0
    camstream = CamStream(video_source)
    camstream.init()
    camstream.run_forever()
    camstream.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
