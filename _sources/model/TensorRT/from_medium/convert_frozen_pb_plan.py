import tensorflow as tf
import tensorrt as trt
import frozen_pb_to_plan

BATCH_SIZE = 16
H, W, C = 448, 640, 3

if __name__ == "__main__":
    '''
    generate the inference engine 
    '''
    pb_model_path = "./frozen_graph.pb"
    plan_model_path = "./model.plan"
    input_node_name = "x:0"
    output_node_name = "Identity:0"

    frozen_pb_to_plan(pb_model_path,
                      plan_model_path,
                      input_node_name,
                      output_node_name,
                      [C, H, W],
                      data_type=trt.float32, # change this for different TRT precision
                      max_batch_size=1,
                      max_workspace=1<<30)
