# Pathy

Pathy, the ROS package that runs padnet, consists of 7 nodes:

```{figure} media/ros-diagram.svg
---
scale: 25%
name: nodes-overview
---

Nodes in the Pathy package
```

## Nodes

- [Camstream](pathy/camstream): Webcam video capture
- [Padnet](pathy/padnet): Semantic segmentation CNN
- [Steering](pathy/steering): Steering from path mask
- [DMS](pathy/dms): Dead man's switch
- [Mav](pathy/mav): Mavlink proxy and injection
- [GCS WebUI WebSockets server](pathy/gcs_webui_ws_server): WebSockets server for the WebUI

## Running

During development and testing we manually started all nodes using:
```sh
ros2 run pathy <name-of-node>
```

In [Byobu](https://www.byobu.org/), a terminal multiplexer, this looks something
like this:

```{figure} media/ros-development-byobu.png
---
scale: 50%
name: byobu
---

Running pathy nodes in development
```

This would probably be a bit of a hassle in production. ROS 2 of course has 
[a more elegant way of launching nodes](https://docs.ros.org/en/foxy/Tutorials/Launch-system.html)
. This is a possible improvement.

## Parameters

Currently, all node arguments are hardcoded. This isn't great. ROS 2 again has more 
elegant way of handling this (e.g. using 
[parameter files](https://docs.ros.org/en/foxy/Guides/Parameters-YAML-files-migration-guide.html)
). This is a possible improvement.
