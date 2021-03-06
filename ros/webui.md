# Web UI

We created a simple web UI to monitor the rover's vision and to implement a 
dead man's switch. 

The client-side of this system is simply the flat webui.html file in the same
directory as this documentation file (ros/webui.html in the repo).

The server-side is implemented by the 
[GCS WebUI WebSockets server](paddy/paddy/gcs_webui_ws_server) ROS node.

## Protocol

Communication is done over a single WebSocket connection using this simple schema:
```json
{
    type: "<message type>",
    data: "<message data>"
}
```

`type` is one of:
- `RGB`: RGB image from the camera (displayed on the left in the UI)
- `MASK`: inferred path mask (displayed on the right in the UI)
- `PING`: ping ([explanation as to why this is necessary](https://stackoverflow.com/questions/10585355/sending-websocket-ping-pong-frame-from-browser))
- `PONG`: ping
- `DMS`: dead man's switch signal, indicates the GCS connection is still OK
- `STEER`: actual steering data sent to the flight controller
- `OFFLOADED_INFERENCE`: a path mask sent by the run_inference_ws.py script (see [Offloaded inference](offloaded_inference) for more info)

`data` is either a string or an object, depended upon the message type.

In case of an image message (RGB, MASK or OFFLOADED_INFERENCE), `data` is the 
base64 encoded raw png or jpeg image data.
