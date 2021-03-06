# DroidCAM

The rover's main camera is simply an old android phone.
We use [DroidCam](https://www.dev47apps.com/) to connect the phone's camera as a
[V4L2](https://en.wikipedia.org/wiki/Video4Linux) webcam device.

We used the 
[build instructions for the DroidCam linux client](https://github.com/dev47apps/droidcam/#building).

The Jetson Nano and android phone communicate over an IP network created by 
[android USB tethering](https://support.google.com/android/answer/9059108?hl=en#zippy=%2Ctether-by-usb-cable).

To more easily connect the DroidCam client to the DroidCam app running on the 
android phone we wrote two small scripts: [get_android_ip.py](get_android_ip) 
to retrieve the Android phone's dynamic IP address and `start_droidcam.sh` to 
start DroidCam using the former.
