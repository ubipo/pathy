"""GCS WebUI WebSockets server

Listens on several topics to serve WebSocket connections from our GCS WebUI, 
also publishes to /pathy/dms, on request of the WebUI.

See /gcs/webui.html in the repo for the client-side part.
Not to be confused with the Mavlink ground control software (QGroundControl/
Mission Planner).
"""

import sys, base64, json
from enum import Enum
from typing import Any

import cv2
import numpy as np

import rclpy
from rclpy.node import MsgType, Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Empty

import asyncio
import websockets


def opencv_to_b64(img: np.dtype, img_type: str = "png") -> str:
    is_success, mask_buf = cv2.imencode(f".{img_type}", img)
    if not is_success:
        raise Exception("Could not encode image")
    b64 = base64.b64encode(mask_buf.tobytes()).decode('utf-8')
    return b64

def b64_to_opencv(b64: str) -> np.dtype:
    img_bytes = base64.b64decode(b64)
    as_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(as_np, cv2.IMREAD_ANYCOLOR)
    return img

class MessageType(Enum):
    RGB = "RGB"
    MASK = "MASK"
    PING = "PING" # https://stackoverflow.com/questions/10585355/sending-websocket-ping-pong-frame-from-browser
    PONG = "PONG"
    DMS = "DMS"
    STEER = "STEER"
    OFFLOADED_INFERENCE = "OFFLOADED_INFERENCE"

def create_websocket_msg(message_type: MessageType, data: Any):
    return json.dumps({
        "type": message_type.value,
        "data": data
    })

def parse_websocket_msg(msg: str):
    msg_obj = json.loads(msg)
    msg_type = MessageType[str(msg_obj["type"])]
    msg_data = msg_obj["data"]
    return msg_type, msg_data

def async_to_non_blocking(loop: asyncio.AbstractEventLoop, callback):
    def non_blocking_callback(**args): # TODO: args, kargs??
        loop.create_task(callback(**args))
    return non_blocking_callback


class GcsWebuiWsServer(Node):
    def __init__(self):
        super().__init__('gcs_webui_ws_server')
        loop = asyncio.get_event_loop()
        self.create_subscription(
            Image, '/pathy/rgb', async_to_non_blocking(loop, self._on_rgb), 10
        )
        # self.create_subscription(Image, '/pathy/mask', self._on_mask, 10)
        self.create_subscription(
            String, '/pathy/steering', 
            async_to_non_blocking(loop, self._on_steering), 10
        )
        self._mask_pub = self.create_publisher(Image, '/pathy/mask', 10)
        self._dms_pub = self.create_publisher(Empty, '/pathy/dms', 10)
        self._bridge = CvBridge()
        self._clients = set()
        self._loop = asyncio.get_event_loop()

    async def init(self):
        await self._init_ws()
        self.get_logger().info('Init OK')
    
    async def serve_forever(self):
        self.get_logger().info('Running...')
        while not self.executor or self.executor.context.ok():
            rclpy.spin_once(self, timeout_sec=0)
            await asyncio.sleep(0) # yield

    async def _init_ws(self):
        await websockets.serve(self._on_new_client, "0.0.0.0", 5678)
        self.get_logger().info('Websocket init OK')

    async def _on_rgb(self, img_msg: Image):
        b64_img = opencv_to_b64(
            self._bridge.imgmsg_to_cv2(img_msg), img_type="jpg")
        ws_msg = create_websocket_msg(MessageType.RGB, b64_img)
        await self._send_to_all(ws_msg)

    async def _on_mask(self, img_msg: Image):
        img = self._bridge.imgmsg_to_cv2(img_msg)
        await self._on_mask_async_parsed(img)

    async def _on_mask_async_parsed(self, img: np.dtype):
        return_msg_data = {
            "type": MessageType.MASK.value,
            "data": opencv_to_b64(img)
        }
        await self._send_to_all(json.dumps(return_msg_data))

    def _on_steering(self, msg: String):
        self._loop.create_task(self._on_steering_async(msg))

    async def _on_steering_async(self, msg: String):
        msg_obj = json.loads(msg.data)
        return_msg_data = {
            "type": MessageType.STEER.value,
            "data": {
                "steer": msg_obj["steer"],
                "throttle": msg_obj["throttle"]
            }
        }
        await self._send_to_all(json.dumps(return_msg_data))
    
    async def _send_to_all(self, data: str):
        if len(self._clients) > 0:
            await asyncio.gather(*[c.send(data) for c in self._clients], return_exceptions=False)

    async def _on_new_client(self, socket, path):
        self.get_logger().info('New client')
        self._clients.add(socket)
        try:
            async for msg in socket:
                await self._handle_ws_message(socket, msg)
        except websockets.ConnectionClosedOK:
            self.get_logger().info("Goodbye")
            pass
        finally:
            self._clients.remove(socket)

    async def _handle_ws_message(self, socket, msg):
        msg_type, msg_data = parse_websocket_msg(msg)
        if msg_type == MessageType.PING:
            return_msg_data = {
                "type": MessageType.PONG,
                "data": None
            }
            await socket.send(json.dumps(return_msg_data))
        elif msg_type == MessageType.DMS:
            self.get_logger().info("DMS")
            self._dms_pub.publish(Empty())
        elif msg_type == MessageType.OFFLOADED_INFERENCE:
            self.get_logger().info("Offloaded inference")
            mask = b64_to_opencv(msg_data)
            mask_msg = self._bridge.cv2_to_imgmsg(np.array(mask))
            self._mask_pub.publish(mask_msg)
            await self._on_mask_async_parsed(mask)
        else:
            raise Exception(f"Unknown message type: {msg_type}")


async def main_async(args=None):
    rclpy.init(args=args)
    ws_server = GcsWebuiWsServer()
    await ws_server.init()
    await ws_server.serve_forever()
    ws_server.destroy_node()
    rclpy.shutdown()


def main(args=None):
    asyncio.get_event_loop().run_until_complete(main_async(args))


if __name__ == "__main__":
    main(sys.argv)
