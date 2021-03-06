# Ardupilot configuration

We only made the absolute necessary changes to Ardupilot's config. You could probably tweak some trim values better.

## General

**FRAME_CLASS**: Rover

## Arming

**ARMING_CHECK**: 9376  
Only check Parameters, Board voltage, Loggig, and system.

**BRD_SAFETYENABLE**: Disabled  
Disable Pixhawk physical safety switch (not connected).

## I/O

**RELAY_PIN**: AUXOUT5  
**RELAY_PIN2**: AUXOUT6  
Track direction output. See 
[Components > Flight Controller](components.md#flight-controller) for hookup 
info.

**SERVO1_FUNCTION**: ThrottleRight  
**SERVO2_FUNCTION**: ThrottleLeft  
Track PWN speed control. See [Components > H-bridge](components.md#h-bridge) for hookup 
info.

**SERVO1_MIN**: 1100 PWN  
**SERVO2_MIN**: 1100 PWN  
Trim.

## Motor

**MOT_SLEWRATE**: 0 %/s  
Disable motor slew. We didn't experience any negative consequences, your experience may differ.

## Steering

**PILOT_STEER_TYPE**: Direction unchanged when backing up  
Personal preference

**SERIAL1_PROTOCOL**: MAVLink1  
You could probably change this to MAVLink2 with only minor changes needed in [ros > mav.py](../ros/paddy/paddy/mav.py).
