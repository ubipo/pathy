"""

See: https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/04_Keras_Flowers_transfer_learning_solution.ipynb
"""

import tensorflow as tf
import segmentation_models as sm


GCS_PATTERN = 'gs://padnet-data/freiburg/tfr/*.tfr'
IMAGE_SIZE = [640, 448]
BATCH_SIZE = 16 # Using TPU v3-8 device => must be divisible by 8 for sharding
# AUTO = tf.data.experimental.AUTOTUNE
AUTO = None

def read_tfrecord(tfrecord):
    features = {
        "rgb": tf.io.FixedLenFeature([], tf.string),
        "gt": tf.io.FixedLenFeature([], tf.string),
    }
    data = tf.io.parse_single_example(tfrecord, features)

    rgb = tf.image.decode_jpeg(data['rgb'], channels=3)
    rgb = tf.image.resize(rgb, IMAGE_SIZE)
    rgb = tf.cast(rgb, tf.float32) / 255.0
    # rgb = tf.reshape(rgb, [*IMAGE_SIZE, 3])

    gt = tf.image.decode_png(data['gt'], channels=1)
    gt = tf.image.resize(gt, IMAGE_SIZE)
    gt = tf.cast(gt, tf.float32) / 255.0
    # gt = tf.reshape(gt, [*IMAGE_SIZE, 1])

    return rgb, gt

def load_dataset(filenames):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_batched_dataset(filenames, train=False):
    dataset = load_dataset(filenames)
    # dataset = dataset.cache() # This dataset fits in RAM
    if train:
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)
    # dataset = dataset.prefetch(AUTO)
    return dataset

def init_tpu():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://10.240.1.2')
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)

def create_model() -> tf.keras.Model:
    BACKBONE = 'efficientnetb3'
    CLASSES = ['path']
    LR = 0.0001

    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    tpu_strategy = init_tpu()

    # with open("/dev/random"):
    with tpu_strategy.scope():    
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)

        optim = tf.keras.optimizers.Adam(LR)

        # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        print("All devices: ", tf.config.list_logical_devices('TPU'))
        model: tf.keras.Model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
        # model.compile(optim, total_loss, metrics)
        model.compile(optim, total_loss)
        return model



tfr_ps = tf.io.gfile.glob(GCS_PATTERN)

# 70% train, 15 valid, 15 test
VALIDATION_SPLIT = 0.7
TEST_SPLIT = 0.85

valid_index = int(len(tfr_ps) * VALIDATION_SPLIT)
test_index = int(len(tfr_ps) * TEST_SPLIT)
tfr_ps_train = tfr_ps[:valid_index]
tfr_ps_valid = tfr_ps[valid_index:test_index]
tfr_ps_test = tfr_ps[test_index:]

training_dataset = get_batched_dataset(tfr_ps_train, train=True)
validation_dataset = get_batched_dataset(tfr_ps_valid, train=False)

# model = create_model()

EPOCHS = 40
validation_steps = int(3670 // len(tfr_ps) * len(tfr_ps_valid)) // BATCH_SIZE
steps_per_epoch = int(3670 // len(tfr_ps) * len(tfr_ps_train)) // BATCH_SIZE

history = model.fit(
    x=training_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    # callbacks=callbacks, 
    validation_data=validation_dataset,
    validation_steps=validation_steps
)


# dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
# dataset = tf.data.TFRecordDataset(tfr_ps_train)
# filenames = tf.io.gfile.glob(GCS_PATTERN)
# rgb_paths = tf.data.Dataset.list_files(GCS_PATTERN_RGB, shuffle=False)
# rgb_images = rgb_paths.map(jpg_map_fn, num_parallel_calls=8)

# gt_paths = tf.data.Dataset.list_files(GCS_PATTERN_GT, shuffle=False)
# gt_images = gt_paths.map(png_map_fn, num_parallel_calls=8)

# images = tf.data.Dataset.zip((rgb_images, gt_images))

