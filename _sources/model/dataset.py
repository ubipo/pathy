import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
from typing import List


def read_tfrecord(tfrecord, image_size):
    features = {
        "rgb": tf.io.FixedLenFeature([], tf.string),
        "gt": tf.io.FixedLenFeature([], tf.string),
    }
    data = tf.io.parse_single_example(tfrecord, features)

    rgb = tf.image.decode_jpeg(data['rgb'], channels=3)
    rgb = tf.image.resize(rgb, image_size)
    rgb = tf.cast(rgb, tf.float32) / 255.0

    gt = tf.image.decode_png(data['gt'], channels=1)
    gt = tf.image.resize(gt, image_size)
    gt = tf.cast(gt, tf.float32) / 255.0

    return rgb, gt

def load_dataset(filenames, image_size, parallel_ops) -> Dataset:
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=parallel_ops)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(
        lambda tfrecord: read_tfrecord(tfrecord, image_size),
        num_parallel_calls=parallel_ops
    )
    return dataset

def split_dataset_paths(ps: List[str], train_ratio: float, validation_ratio: float):
    training_split = train_ratio
    validation_split = train_ratio + validation_ratio
    valid_index = int(len(ps) * training_split)
    test_index = int(len(ps) * validation_split)
    train_ps = ps[:valid_index]
    validate_ps = ps[valid_index:test_index]
    test_ps = ps[test_index:]
    return train_ps, validate_ps, test_ps
