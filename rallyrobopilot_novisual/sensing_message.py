"""
Minimal SensingSnapshot class for headless GA training.

This module contains only the SensingSnapshot data class needed for:
- Loading recorded lap data from pickle files
- Accessing car state (position, speed, angle, controls)

Network/manager classes removed - not needed for headless training.
Must remain structurally identical to rallyrobopilot.sensing_message.SensingSnapshot
for pickle compatibility.
"""

import struct
import numpy as np


def iter_unpack(format, data):
    """Helper function for unpacking binary data."""
    nbr_bytes = struct.calcsize(format)
    return struct.unpack(format, data[:nbr_bytes]), data[nbr_bytes:]


class SensingSnapshot:
    """
    Data container for car simulation state.

    Used for:
    - Deserializing recorded lap data from .npz files
    - Accessing car state during checkpoint creation

    Attributes:
        current_controls: (forward, backward, left, right) tuple of bools/ints
        car_position: (x, y, z) tuple
        car_speed: float
        car_angle: float (rotation_y in degrees)
        rotation_speed: float
        raycast_distances: list of floats
        image: numpy array (h, w, 3) or None
    """

    def __init__(self):
        # Controls: Forward - Backward - Left - Right
        self.current_controls = (0, 0, 0, 0)
        self.car_position = (0, 0, 0)
        self.car_speed = 0
        self.car_angle = 0
        self.rotation_speed = 0
        self.raycast_distances = [0]
        self.image = None

    def pack(self):
        """
        Serialize snapshot to binary format.

        Returns:
            bytes: Packed binary data
        """
        byte_data = b''
        byte_data += struct.pack(">BBBB", *self.current_controls)
        byte_data += struct.pack(">ffffff",
                                self.car_position[0],
                                self.car_position[1],
                                self.car_position[2],
                                self.car_angle,
                                self.car_speed,
                                self.rotation_speed)

        nbr_raycasts = len(self.raycast_distances)
        byte_data += struct.pack(">B" + "f" * nbr_raycasts,
                                nbr_raycasts,
                                *self.raycast_distances)

        if self.image is not None:
            byte_data += struct.pack(">ii", self.image.shape[0], self.image.shape[1])
            byte_data += self.image.tobytes()
        else:
            byte_data += struct.pack(">ii", 0, 0)

        return byte_data

    def unpack(self, data):
        """
        Deserialize snapshot from binary format.

        Args:
            data: bytes to unpack
        """
        self.current_controls, data = iter_unpack(">BBBB", data)
        (x, y, z, a, s, rs), data = iter_unpack(">ffffff", data)
        self.car_position = (x, y, z)
        self.car_angle = a
        self.car_speed = s
        self.rotation_speed = rs

        (nbr_raycasts,), data = iter_unpack(">B", data)
        self.raycast_distances, data = iter_unpack(">" + "f" * nbr_raycasts, data)

        (h, w), data = iter_unpack(">ii", data)

        if h * w > 0:
            self.image = np.frombuffer(data, np.uint8).reshape(h, w, 3)
        else:
            self.image = None
