# Components

The rover has two levels. The bottom level is the diy-assembly tank chassis. The top level is a wooden frame to hold the power components.

```{figure} media/rover-lower.jpg
Rover bottom level
```

```{figure} media/rover-upper.jpg
Rover top level
```

(components:chassis)=
## Chassis

The base of the rover is the "[TS100 Shock Absorber Tank Chassis](https://www.seeedstudio.com/TS100-shock-absorber-tank-chassis-with-track-and-DC-geared-motors-Kit-p-4107.html)" by [DOIT](https://www.doit.am) ([manual](https://raw.githubusercontent.com/SeeedDocument/Outsourcing/master/110090267%20TS100%20shock%20absorber%20tank%20chassis%20with%20track%20and%20DC%20geared%20motors%20Kit/InstallationforTS100%20.pdf)). The kit for this rover was kindly provided to us by our mentor at [Airobot](https://airobot.eu/).

The chassis uses [simple DC motors](https://item.taobao.com/item.htm?spm=a1z10.5-c.w4002-7420481794.72.fWWJc1&id=45203541487) with a Hall-effect sensor. It is possible to use these sensors with Ardupilot for better positional accuracy, but we didn't in our setup. The motors are rated for a max of 9V.

(components:h-bridge)=
## H-bridge

To control the speed and direction of the two motors we used a [VMA409 H-bridge breakout board](https://www.velleman.eu/products/view/?id=435576), also provided to us by Airobot.

To ease control of the H-bridge through software we added a simple [NOR-gate](https://web.mit.edu/6.131/www/document/7402.pdf) circuit.

```{figure} media/nor-gates-circuit.svg
---
name: circuit-diagram
---

Circuit to control the H-bridge from the Pixhawk
```

## Flight controller

To control the rover's motors (and potentially other RC components) we used a [Pixhawk 2.4.8 flight controller](https://docs.px4.io/v1.9.0/en/flight_controller/pixhawk.html). 

```{figure} media/pixhawk.png
---
scale: 25%
name: pixhawk
---

Pixhawk flight controller
```

To control the speed of the motors the H-bridge's motor PWN connections (labelled "ENA" and "ENB" on the VMA409) should be connected to the Pixhawk's SERVO1 and SERVO2 pins (mainout rail). 

The Pixhawk's Relay 1 and Relay 2 pins control the motors' direction. See the [circuit diagram above](circuit-diagram) for the necessary connections.

See [Ardupilot Configuration](ardupilot-config.md) for all the necessary ardupilot parameter tweaks.

## SBC

As previously discussed we used an [NVIDIA Jetson Nano SBC](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) to make autonomous steering decisions. The software logic for this is discussed in the chapter about [ROS](../ros/README).

```{figure} media/jetson.png
---
scale: 25%
name: jetson
---

NVIDIA Jetson Nano SBC
```

To relay its eventual steering decisions to the flight controller (and thus the motors) we connected the Jetson Nano to the Pixhawk through serial (Pixhawk TELEM 1 <-> Jetson Nano UART_2). Also see [this guide by MathWorks](https://www.mathworks.com/help/supportpkg/jetsoncpu/ref/jetson-pixhawk-interface-example.html).

## Power

It is possible to power all components though a single LiPo battery but due to time constraints we used three separate batteries:

 - A [14.8V LiPo battery](https://www.genstattu.com/ta-rl3-120c-2000-4s1p.html) for the Jetson Nano
 - Two generic 3.6V 18650 batteries in series
 - A generic USB battery bank

The Jetson Nano is powered by the LiPo battery through a simple [adjustable buck converter](https://www.antratek.be/adjustable-dc-dc-power-converter-1-25v-35v-3a-lm2596) to step the battery's 14.8V down to 5V.

The motors are powered directly by the two 18650 batteries. The motors are rated for a max voltage of 9V and thus don't run at full speed off of the 7V battery output. This isn't a problem.

The Pixhawk is simply powered through its USB-micro connection.
