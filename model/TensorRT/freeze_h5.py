import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.python.keras.backend as K
import numpy as np
import segmentation_models as sm

frozen_out_path = ''
frozen_graph_filename = "frozen_graph"
model_fname = './test_save.h5'

def custom_lossf():
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss

model = load_model(model_fname, custom_objects={
    'dice_loss_plus_1binary_focal_loss': custom_lossf(),
    'iou_score': sm.metrics.IOUScore(threshold=0.5),
    'f1-score': sm.metrics.FScore(threshold=0.5)
    })

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)

# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)