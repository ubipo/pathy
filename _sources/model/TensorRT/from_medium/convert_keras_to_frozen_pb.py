from converter import keras_to_frozen_pb

input_keras_model = "../input/inception_v3.h5"
output_pb_model = "../output/inception_v3.pb"


if __name__ == "__main__":
    node_out_name = keras_to_frozen_pb(input_keras_model, output_pb_model)
    print("the output node name is:", node_out_name)