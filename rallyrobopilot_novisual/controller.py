"""
Direct controller for physics-only car.
No dependencies on visual game components.
"""


class DirectController:
    """
    Controller interface for programmatic car control.

    Compatible with simulator.py's DirectController interface,
    but works with pure physics car instead of visual Entity.
    """

    def __init__(self, car_physics):
        """
        Args:
            car_physics (CarPhysics): Physics-only car instance
        """
        self.car = car_physics

        # Control state
        self.controls = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False
        }

        # Recording
        self.trajectory = []
        self.frame_count = 0
        self.recording = False

    def apply_controls(self, forward, backward, left, right):
        """
        Set control inputs for next update.

        Args:
            forward (bool): Accelerate forward
            backward (bool): Brake/reverse
            left (bool): Steer left
            right (bool): Steer right
        """
        self.controls['forward'] = forward
        self.controls['backward'] = backward
        self.controls['left'] = left
        self.controls['right'] = right

    def release_all_controls(self):
        """Release all control inputs."""
        self.controls = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False
        }

    def update_car(self):
        """
        Update car physics with current control inputs.

        This should be called once per physics timestep.
        """
        self.car.update(
            forward=self.controls['forward'],
            backward=self.controls['backward'],
            left=self.controls['left'],
            right=self.controls['right']
        )

    def reset_to_state(self, position, angle, speed):
        """
        Reset car to specific state.

        Args:
            position (tuple): (x, y, z) position
            angle (float): rotation_y in degrees
            speed (float): current speed
        """
        self.car.reset(position, angle, speed)

    def start_recording(self):
        """Start recording trajectory."""
        self.trajectory = []
        self.frame_count = 0
        self.recording = True

    def stop_recording(self):
        """Stop recording trajectory."""
        self.recording = False

    def record_state(self, frame):
        """
        Record current car state to trajectory.

        Args:
            frame (int): Frame number
        """
        if self.recording:
            state = {
                'frame': frame,
                'position': (self.car.x, self.car.y, self.car.z),
                'speed': self.car.speed,
                'angle': self.car.rotation_y
            }
            self.trajectory.append(state)
            self.frame_count = frame + 1

    def get_state(self):
        """Get current car state."""
        return self.car.get_state()