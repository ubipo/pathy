# ROS

[ROS (Robot Operating System)](https://www.ros.org/) is a "set of software 
libraries and tools that help you build robot applications". We used the newest
stable ROS release, which at the time of creating this project was 
[ROS 2: Foxy Fitzroy](https://docs.ros.org/en/foxy/Installation.html).

A robot programmed using ROS uses 
[Nodes](https://docs.ros.org/en/foxy/Tutorials/Understanding-ROS2-Nodes.html)
as its functional blocks and 
[Topics](https://docs.ros.org/en/foxy/Tutorials/Topics/Understanding-ROS2-Topics.html)
to exchange data between these nodes.

Nodes are separated into packages. A package is the minimal publishable unit of
ROS 'stuff'.

To control our rover we created one package: "pathy". This package contains all 
nodes to handle camera input, steering, oversight etc.

We discuss the various nodes and their function in our 
[package's readme](pathy/README).

Two auxillary components are needed to use the package: [DroidCam](droidcam/README) for
 the software side of the rover's camera and [web UI](webui) for the live
  monitoring system.

## Advantages of using ROS

Besides encouraging a modular approach, ROS also offers some nice debugging 
functionality.

### Topic inspection

To debug a ROS topic ("/pathy/steering" in this example), use:

```
ros2 topic echo /pathy/steering
```

Something similar can be used for Image topics:

```
ros2 run image_tools showimage --ros-args -r image:=/pathy/rgb
```

```{figure} media/showimage.png
---
scale: 75%
---

Using showimage to inspect an image topic
```

### Rewriting topics

Because topics are just strings, and nodes are thus loosely coupled, it is 
possible to rewrite topics.

I often used:
```
ros2 run pathy mav --ros-args -r /pathy/steering_safe:=/pathy/steering
```

...during development to run the rover with 'unsafe' steering (steering messages 
that aren't protected by the dead man's switch).
