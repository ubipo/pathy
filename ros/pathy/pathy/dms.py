"""Dead man's switch

Passes steering inputs through from /pathy/steering to /pathy/steering_safe, but 
only if a message was recently posted to /pathy/dms.

"""

import os, sys
from datetime import datetime, timedelta

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Empty


class Dms(Node):
    def __init__(self, dms_max_delay):
        super().__init__('dms')
        self._pub = self.create_publisher(String, '/pathy/steering_safe', 10)
        self.create_subscription(String, '/pathy/steering', self._steering_handler, 10)
        self.create_subscription(Empty, '/pathy/dms', self._dms_handler, 10)
        self._dms_max_delay = dms_max_delay
        self._last_dms_msg_time = datetime.min

    def run_forever(self):
        self.get_logger().info("Running...")
        rclpy.spin(self)

    def _steering_handler(self, msg):
        if (datetime.now() - self._last_dms_msg_time) < self._dms_max_delay:
            self._pub.publish(msg)

    def _dms_handler(self, msg):
        self._last_dms_msg_time = datetime.now()


def main(args=None):
    rclpy.init(args=args)
    dms_max_delay = timedelta(seconds=1.0)
    dms = Dms(dms_max_delay)
    dms.run_forever()
    dms.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
