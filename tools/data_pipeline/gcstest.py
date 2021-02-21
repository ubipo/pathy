"""

See: https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/04_Keras_Flowers_transfer_learning_solution.ipynb
"""

import datetime
from functools import partial

import numpy as np
import tensorflow as tf
import segmentation_models as sm
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.tpu_strategy import TPUStrategy
from tensorflow.python.ops.array_ops import repeat


# def batch_dataset(dataset: Dataset) -> Dataset:
#     # dataset = dataset.cache() # This dataset fits in RAM

#     dataset = dataset.batch(BATCH_SIZE)
#     # dataset = dataset.prefetch(AUTO)
#     return dataset

