# Offloaded inference

Due to time constraints we couldn't get inference to work on the Jetson Nano.
As a workaround we simply ran inference on the GCS laptop instead of the 
Jetson's GPU. 

RGB images from the camera are streamed to the laptop over a websocket, similar to the web UI.

The [inference script running on the laptop](../model/run_inference_ws) then returns the inferred path mask back (using the `OFFLOADED_INFERENCE` message type).

This works well and is actually very handy during development while testing different DNN setups.

The [ROS node necessary for onboard inference](pathy/pathy/padnet) is also included (untested) in the pathy ROS package. It *should* work plug-and-play if the Jetson's [Tensorflow](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) and [TensorRT](https://developer.nvidia.com/tensorrt) python bindings work.
