"""Get Android IP

Gets a USB-tethering-connected Android phone's IP by pinging
all devices on the `192.168.42.0/24` subnet (Android USB 
tethering always uses this subnet).
"""

import subprocess
import socket
import psutil
import sys

ANDROID_NET_PREFIX = "192.168.42"

def get_ip_addresses(family):
    for interface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            if snic.family == family:
                yield (interface, snic.address)

def get_ips_up():
	ps = []
	for i in range(1, 255):
		host = f"{ANDROID_NET_PREFIX}.{i}"
		args = ['ping', '-c', '1', '-w', '2', host]
		ps.append(subprocess.Popen(
			args,
			stdout=subprocess.DEVNULL
		))

	results = {}
	while len(results) != 254:
		for i, p in enumerate(ps):
			result = p.poll()
			if result is not None:
				results[i+1] = result

	ips_up = [f"{ANDROID_NET_PREFIX}.{i}" for i, result in results.items() if result == 0]
	return ips_up

def get_my_android_net_addr():
	for name, addr in get_ip_addresses(socket.AF_INET):
		if addr.startswith(ANDROID_NET_PREFIX):
			return addr

if __name__ == "__main__":
	my_ip = get_my_android_net_addr()
	others_up = [ip for ip in get_ips_up() if ip != my_ip]
	if len(others_up) != 1:
		sys.exit(1)
	print(others_up[0])

