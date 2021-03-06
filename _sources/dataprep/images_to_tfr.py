"""Converts a directory with rgb/ and gt/ image subdirectories
to a single directory with tfr files

Usage: create_label_jsons.py <dir>
    dir: directory with rgb/ and gt/ subdirs

tfr files get written to a new tfr/ directory within <dir>

See: https://www.tensorflow.org/tutorials/load_data/tfrecord
"""

import sys, os
from pathlib import Path

import tensorflow as tf


def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def image_to_tfr(rgb_p: Path, gt_p: Path, tfr_p: Path):
    with open(rgb_p, 'rb') as rgb_f:
        with open(gt_p, 'rb') as gt_f:
            with tf.io.TFRecordWriter(str(tfr_p.resolve())) as out_file:
                feature = {
                    "rgb": _bytestring_feature([rgb_f.read()]),
                    "gt": _bytestring_feature([gt_f.read()])
                }
                tf_record = tf.train.Example(features=tf.train.Features(feature=feature))
                out_file.write(tf_record.SerializeToString())

def images_to_tfr(directory: Path):
    rgb_dir = directory / "rgb"
    gt_dir = directory / "gt"
    tfr_dir = directory / "tfr"
    os.mkdir(tfr_dir)
    
    rgb_ps = sorted(rgb_dir.iterdir())
    gt_ps = sorted(gt_dir.iterdir())
    assert(len(rgb_ps) == len(gt_ps))
    
    for rgb_p, gt_p in zip(rgb_ps, gt_ps):
        stem = rgb_p.stem
        assert(stem == gt_p.stem)
        image_to_tfr(rgb_p, gt_p, tfr_dir / f"{stem}.tfr")
    


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    assert(directory.is_dir())
    images_to_tfr(directory)
