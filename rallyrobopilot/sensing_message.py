import struct

import numpy as np


def iter_unpack(format, data):
    nbr_bytes = struct.calcsize(format)
    return struct.unpack(format, data[:nbr_bytes]), data[nbr_bytes:]

"""
    SensingSnapshot is a packing/unpacking class for diverse car simulation related information
"""
class SensingSnapshot:
    def __init__(self):
        #   Forward - Backward - Left - Right
        self.current_controls = (0,0,0,0)
        self.car_position = (0,0,0)
        self.car_speed = 0
        self.car_angle = 0
        self.rotation_speed = 0
        self.raycast_distances = [0]
        self.image = None

    def pack(self):
        byte_data = b''
        byte_data += struct.pack(">BBBB", *self.current_controls)
        # byte_data += struct.pack(">fffff", self.car_position[0], self.car_position[1], self.car_position[2], self.car_angle, self.car_speed)

        byte_data += struct.pack(">ffffff", self.car_position[0], self.car_position[1],
                                 self.car_position[2], self.car_angle, self.car_speed,
                                 self.rotation_speed)

        nbr_raycasts = len(self.raycast_distances)
        byte_data += struct.pack(">B" + "f" * nbr_raycasts, nbr_raycasts, *self.raycast_distances)

        if self.image is not None:
            byte_data += struct.pack(">ii", self.image.shape[0], self.image.shape[1])
            byte_data +=  self.image.tobytes()
        else:
            byte_data += struct.pack(">ii", 0, 0)

        return byte_data

    def unpack(self, data):
        self.current_controls, data = iter_unpack(">BBBB", data)
        (x,y,z,a,s,rs), data = iter_unpack(">ffffff", data)
        self.car_position = (x,y,z)
        self.car_angle = a
        self.car_speed = s
        self.rotation_speed = rs

        (nbr_raycasts,), data = iter_unpack(">B", data)
        self.raycast_distances, data = iter_unpack(">" + "f" * nbr_raycasts, data)

        (h,w), data = iter_unpack(">ii", data)

        if h*w > 0:
            self.image = np.frombuffer(data, np.uint8).reshape(h,w,3)
        else:
            self.image = None

"""
#   Snapshot formatting and managing class
#       --> pass it the chunks received via socket, the class correctly split and parse the received messages
#       uses a callback to inform higher level code of reception of a complete SensingSnapshot
"""
class SensingSnapshotManager:
    def __init__(self, received_snapshot_callback = None):
        self.pending_data = b''
        self.received_snapshot_callback = received_snapshot_callback

    def pack(self, snapshot):
        data = snapshot.pack()
        data = struct.pack(">i", len(data)) + data

        return data


    def add_message_chunk(self, chunk):
        self.pending_data += chunk

        sizeheader = struct.calcsize(">i")
        message_size = struct.unpack(">i", self.pending_data[:sizeheader])[0]

        if message_size+sizeheader <= len(self.pending_data):
            snapshot = SensingSnapshot()

            snapshot.unpack(self.pending_data[sizeheader:sizeheader+message_size])
            if self.received_snapshot_callback is not None:
                self.received_snapshot_callback(snapshot)

            self.pending_data = self.pending_data[sizeheader+message_size:]


import socket
import imageio
class NetworkDataCmdInterface:
    def __init__(self, callback, address = "127.0.0.1", port = 7654):
        self.data = []

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 125536)
        self.socket.connect((address, port))
        # self.socket.setblocking(False)
        self.socket.settimeout(0.05)

        self.msg_mngr = SensingSnapshotManager(callback)

    def send_cmd(self, cmd):
        self.socket.send(bytes(cmd, "utf-8"))

    def recv_msg(self):
        try:
            while True:
                data = self.socket.recv(2**20)

                if len(data) == 0:
                    break

                self.msg_mngr.add_message_chunk(data)

        except Exception as e:
            pass

    def process_sensing_message(self, sensing_snapshot):
        #   Sample function to use as a callback
        print("sensing_snapshot.car.position =", sensing_snapshot.car_position)

        imageio.imsave("last_image.png", sensing_snapshot.image)

