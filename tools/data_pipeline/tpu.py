import tensorflow as tf
from tensorflow.python.distribute.tpu_strategy import TPUStrategy


def resolve_tpu_strategy(tpu: str) -> TPUStrategy:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu)
    return tf.distribute.TPUStrategy(resolver)

def get_tpu_devices():
    return tf.config.list_logical_devices('TPU')
