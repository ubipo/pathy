# Rover

Our rover is based on a [tank chassis kit](https://www.seeedstudio.com/TS100-shock-absorber-tank-chassis-with-track-and-DC-geared-motors-Kit-p-4107.html), controlled by a [Pixhawk flight controller](https://docs.px4.io/v1.9.0/en/flight_controller/pixhawk.html), and uses an [NVIDIA Jetson Nano SBC](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) to make autonomous steering decisions. The software logic for this is discussed in the chapter about [ROS](../ros/README).

```{figure} media/rover-cropped.jpg
---
scale: 25%
name: rover
---

The assembled rover.
```

We used a generic USB WiFi-adapter to connect our ground station (laptop) to the SBC and by extent, the flight controller. 

```{figure} media/system-diagram.png
---
scale: 25%
name: system
---

Rover and gcs system overview
```

The rover's components are discussed in [Components](components.md).

All configuration of the Pixhawk, as well as manual control of the rover was done through [QGroundControl](http://qgroundcontrol.com/) GCS software. See the [QGroundControl](qgroundcontrol.md) chapter for more info about this. 

See [Ardupilot Configuration](ardupilot-config.md) for all the necessary ardupilot parameter tweaks.
