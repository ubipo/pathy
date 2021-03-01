"""Mav - proxies mavlink commands from the gcs to the drone's flight controller 
and processes steering input from /paddy/steering_safe.

Note: you can rewrite the /paddy/steering_safe to /paddy/steering to bypass dms
by issuing "--ros-args -r /paddy/steering_safe:=/paddy/steering" on the command
line.
"""

import sys, time, struct, binascii, json, asyncio
from datetime import datetime, timedelta

from pymavlink import mavutil
from pymavlink.dialects.v20.ardupilotmega import MAVLink as ArduMAVLink
from pymavlink.dialects.v10.ardupilotmega import MAVLink_manual_control_message

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


print(sys.version)

def wait_heartbeat(m):
    '''wait for a heartbeat so we know the target system IDs'''
    m.recv_match(type='HEARTBEAT', blocking=True)

def arm_disarm(mf, arm = False):
    mf.mav.command_long_send(
        mf.target_system,
        mf.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        arm, 0, 0, 0, 0, 0, 0
    )

def manual_control(mf, throttle: float, steer: float):
    """
    throttle: (-1 backwards, 0 idle, 1 forwards)
    steer: (-1 left, 0 straight, 1 right)
    """
    # Imitate gcs, pixhawk doesn't respond otherwise
    # Why?
    original_src_system = mf.mav.srcSystem
    original_src_component = mf.mav.srcComponent
    mf.mav.srcSystem = 0xff
    mf.mav.srcComponent = 0xbe
    throttle_mapped = int(throttle * 1000)
    steer_mapped = int(steer * 1000)
    mav: ArduMAVLink = mf.mav
    msg = MAVLink_manual_control_message(
        mf.target_system,
        x = 0,
        y = steer_mapped, 
        z = throttle_mapped,
        r = 0,
        buttons = 0
    )
    mav.send(msg)
    mf.mav.srcSystem = original_src_system
    mf.mav.srcComponent = original_src_component

# REPEAT_DELTA = 0.01
# async def repeated_manual_control(mf, throttle, steer, length_seconds = 1.0):
#     for _ in range(int(float(length_seconds) / REPEAT_DELTA)):
#         manual_control(mf, throttle, steer)
#         asyncio.sleep(REPEAT_DELTA)


class Mav(Node):
    def __init__(self, pixhawk_serial_dev, pixhawk_serial_baud, source_system_me, override_timeout):
        super().__init__('mav')
        self.create_subscription(String, '/paddy/steering_safe', self._steering_handler, 10)
        self._pixhawk_serial_dev = pixhawk_serial_dev
        self._pixhawk_serial_baud = pixhawk_serial_baud
        self._source_system_me = source_system_me
        self._override_timeout = override_timeout
        self._last_manual_control_override = datetime.min

    def init(self):
        self._init_mav()
        self.get_logger().info('Init OK')

    async def run_forever(self):
        self.get_logger().info('Running...')
        self.get_logger().info('Proxying connections between pixhawk and gcs...')
        while not self.executor or self.executor.context.ok():
            rclpy.spin_once(self, timeout_sec=0)
            await asyncio.sleep(0) # yield

            gcs_msg = self._gcs.recv_match(timeout = 0.001)
            if gcs_msg is not None:
                time_since_last_override = datetime.now() - self._last_manual_control_override
                should_pass_msg = (
                    not (
                        gcs_msg.get_type() == "MANUAL_CONTROL"
                        and time_since_last_override < self._override_timeout
                    )
                )
                if should_pass_msg:
                    self._pixhawk.port.write(gcs_msg.get_msgbuf()) # mavwerial.write is broken for python >3.6
                else:
                    print(f"Blocked msg with {gcs_msg.get_type()=}")
        
            pixhawk_msg = self._pixhawk.recv_match(timeout = 0.001)
            if pixhawk_msg is not None:
                self._gcs.write(pixhawk_msg.get_msgbuf())

    def _init_mav(self):
        self._pixhawk = mavutil.mavserial(
            self._pixhawk_serial_dev, 
            baud=self._pixhawk_serial_baud, 
            source_system=self._source_system_me
        )
        self._gcs = mavutil.mavudp(
            "0.0.0.0:14550",
            input=True,
            source_system=self._source_system_me
        )
        self.get_logger().info("Connecting to flight controller...")
        wait_heartbeat(self._pixhawk)
        self.get_logger().info(f"Flight controller sysid: {self._pixhawk.sysid}")
        self.get_logger().info("Waiting for GCS connection...")
        wait_heartbeat(self._gcs)
        self.get_logger().info(f"GCS sysid: {self._gcs.sysid}")

    def _override_manual_control(self, throttle, steer):
        self._last_manual_control_override = datetime.now()
        manual_control(self._pixhawk, throttle, steer)

    def _steering_handler(self, msg: String):
        msg_obj = json.loads(msg.data)
        throttle = msg_obj["throttle"]
        steer = msg_obj["steer"]
        self.get_logger().info(f"Steering: {throttle=} {steer=}")
        self._override_manual_control(throttle, steer)


async def main_async(args=None):
    rclpy.init(args=args)
    # TODO: Move to ROS params
    pixhawk_serial_dev = "/dev/ttyTHS1"
    pixhawk_serial_baud = 57600
    source_system_me = 42
    override_timeout = timedelta(seconds=3)
    mav = Mav(pixhawk_serial_dev, pixhawk_serial_baud, source_system_me, override_timeout)
    mav.init()
    await mav.run_forever()
    mav.destroy_node()
    rclpy.shutdown()


def main(args=None):
    asyncio.get_event_loop().run_until_complete(main_async(args))


if __name__ == "__main__":
    main(sys.argv)
