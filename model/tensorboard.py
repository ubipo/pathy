from datetime import datetime
import tensorflow as tf


def create_tensorboard_callback(url: str):
    log_dir = url + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
