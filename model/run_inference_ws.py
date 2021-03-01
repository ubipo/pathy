"""

"""

import sys, json, base64, logging, asyncio
from queue import Queue, LifoQueue
from threading import Thread

import tensorflow as tf
import segmentation_models as sm
import numpy as np
import cv2
import websockets


IMAGE_SIZE = [448, 640]


def opencv_to_model_img(img: np.dtype):
    resized = tf.image.resize(img, IMAGE_SIZE)
    cast = tf.cast(resized, tf.float32) / 255.0
    return cast

def model_img_to_opencv(model_img: tf.Tensor) -> np.dtype:
    '''
    Keras model output floats in [0,1], while OpenCV represents images as uint8
    (i.e. integers in [0, 255]).
    '''
    opencv_img = np.round(model_img.numpy() * 255).astype('uint8')
    return opencv_img

def b64_to_opencv(b64: str) -> np.dtype:
    img_bytes = base64.b64decode(b64)
    as_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(as_np, cv2.IMREAD_COLOR)
    return img

def opencv_to_b64(img: np.dtype) -> str:
    is_success, mask_buf = cv2.imencode(".png", img)
    im_out = cv2.imdecode(mask_buf, cv2.IMREAD_ANYCOLOR)
    if not is_success:
        raise Exception("Could not encode image")
    b64 = base64.b64encode(mask_buf.tobytes()).decode('utf-8')
    return b64

def predict(model: tf.keras.Model, img: np.dtype) -> np.dtype:
    predictions = model(np.asarray([opencv_to_model_img(img)]))
    mask = model_img_to_opencv(predictions[0])
    return mask

def predict_from_queue(
    model: tf.keras.Model,
    in_queue: Queue,
    socket: websockets.WebSocketClientProtocol,
    loop: asyncio.AbstractEventLoop
):
    while True:
        img = in_queue.get()
        mask = predict(model, img)
        return_msg_data = {
            "type": "OFFLOADED_INFERENCE",
            "data": opencv_to_b64(mask)
        }
        loop.create_task(socket.send(json.dumps(return_msg_data)))

async def run_inference_ws(url: str, model: tf.keras.Model):
    logging.info("Connecting...")
    async with websockets.connect(url) as websocket:
        logging.info("Connected!")
        predict_in = LifoQueue(maxsize=1)
        Thread(
            target=predict_from_queue,
            args=[model, predict_in, websocket, asyncio.get_running_loop()]
        ).start()
        async for msg in websocket:
            msg_obj = json.loads(msg)
            msg_type = msg_obj["type"]
            if msg_type == "RGB":
                msg_data = msg_obj["data"]
                img = b64_to_opencv(msg_data)
                if not predict_in.empty():
                    predict_in.get()
                predict_in.put(img)
                print(".", end='', flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model_path = sys.argv[1]
    url = sys.argv[2]
    model = tf.keras.models.load_model(model_path, compile=False)
    asyncio.get_event_loop().run_until_complete(run_inference_ws(url, model))
