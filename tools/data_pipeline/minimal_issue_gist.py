import logging

import tensorflow as tf
import numpy as np

logging.getLogger('tensorflow').setLevel(logging.DEBUG)


def get_dataset(dispatcher, batch_size):
  fake_data = np.zeros((128, 256, 256), dtype=np.float32)
  ds = tf.data.Dataset.from_tensor_slices(fake_data)
  ds = ds.repeat()
  ds = ds.batch(batch_size)
  ds = ds.apply(tf.data.experimental.service.distribute('parallel_epochs', dispatcher.target, job_name='data_job'))
  return ds

def run(strategy, batch_size):
  data_dispatcher = tf.data.experimental.service.DispatchServer()
  dispatcher_address = data_dispatcher.target.split("://")[1]
  worker = tf.data.experimental.service.WorkerServer(dispatcher_address=dispatcher_address)

  per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
  dataset = strategy.experimental_distribute_datasets_from_function(
    lambda _: get_dataset(
      data_dispatcher,
      per_replica_batch_size,
    )
  )
  ds_iterator = iter(dataset)

  @tf.function(input_signature=[tf.TensorSpec([None, 256, 256], dtype=tf.float32)])
  def step_fn(inputs):
    pass

  @tf.function
  def train_step(iterator):
    strategy.run(step_fn, args=(next(iterator),))

  for _ in range(10):
    train_step(ds_iterator)

  print('Finished, stopping data workers...')
  del ds_iterator
  del dataset
  data_dispatcher._stop()
  worker._stop()
  print('Stopped')


def setup_tpu_strategy():
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='padnet')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return tf.distribute.TPUStrategy(resolver)


if __name__ == '__main__':
  tpu = False
  if tpu:
    strategy = setup_tpu_strategy()
  else:
    strategy = tf.distribute.get_strategy()
  batch_size = 64
  run(strategy, batch_size)

