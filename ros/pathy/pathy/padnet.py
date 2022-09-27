"""Padnet

Runs our semantic segmentation CNN on images from /pathy/rgb and 
publishes them on /pathy/mask.

Untested.

Adapted from https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import tensorflow as tf


# TODO: Fix
output_names = ['Logits/Softmax']
input_names = ['input_1']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


class Padnet(Node):
    def __init__(self, model_path):
        super().__init__('padnet')
        self.model_path = model_path
        self.create_subscription(Image, '/pathy/rgb', self._rgb_handler, 10)
        self._mask_pub = self.create_publisher(Image, '/pathy/mask', 10)
        self._bridge = CvBridge()

    def run_forever(self):
        self.get_logger().info("Running...")
        rclpy.spin(self)

    def init(self):
        self._init_graph()
        self.get_logger().info('Init OK')

    def _init_graph(self):
        trt_graph = get_frozen_graph('./model/trt_graph.pb')
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self._tf_sess = tf.Session(config=tf_config)
        tf.import_graph_def(trt_graph, name='')
        self._input_tensor_name = input_names[0] + ":0"
        output_tensor_name = output_names[0] + ":0"
        self._output_tensor = self._tf_sess.graph.get_tensor_by_name(output_tensor_name)

    def _rgb_handler(self, msg):
        rgb = self._bridge.imgmsg_to_cv2(msg)
        rgb_array = np.asarray(rgb)
        feed_dict = {
            self._input_tensor_name: rgb_array
        }
        mask = self._tf_sess.run(self._output_tensor, feed_dict)
        mask_msg = self._bridge.cv2_to_imgmsg(np.array(mask))
        self._mask_pub.publish(mask_msg)


def main(args=None):
    rclpy.init(args=args)
    graph_file = Path("./model.pb")
    padnet = Padnet(graph_file)
    padnet.init()
    padnet.run_forever()
    padnet.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
