"""
DirectController for feeding controls directly to the game without HTTP.
Used for GA training and automated testing.
"""

from ursina import held_keys, invoke
import time
import sys
from pathlib import Path

# Add genetics_algo to path to import checkpoint module
genetics_algo_path = Path(__file__).parent.parent / "genetics_algo"
sys.path.insert(0, str(genetics_algo_path))

from checkpoint import check_line_crossing


class DirectController:
    """
    Controls the car directly by manipulating held_keys.
    Collects trajectory data and detects checkpoint crossing.
    """

    def __init__(self, car):
        """
        Args:
            car: Car instance from the game
        """
        self.car = car

        # Trajectory recording
        self.trajectory = []
        self.recording = False

        # Checkpoint detection
        self.target_checkpoint = None
        self.checkpoint_crossed = False
        self.crossing_frame = None

        # Frame counter
        self.frame_count = 0

    def reset_to_state(self, position, angle, speed=0.0, rotation_speed=0.0):
        """
        Reset car to specific state.

        Args:
            position (tuple): (x, y, z)
            angle (float): Rotation Y in degrees
            speed (float): Initial speed
            rotation_speed (float): Angular velocity (degrees/sec)
        """
        self.car.reset_position = position
        self.car.reset_orientation = (0, angle, 0)
        self.car.reset_car()
        self.car.speed = speed

        print(f"Car reset to position={position}, angle={angle}, speed={speed}")

    def apply_controls(self, forward, backward, left, right):
        """
        Apply control inputs directly to held_keys.

        Args:
            forward (bool): Forward pressed
            backward (bool): Backward pressed
            left (bool): Left pressed
            right (bool): Right pressed
        """
        held_keys['w'] = forward
        held_keys['s'] = backward
        held_keys['a'] = left
        held_keys['d'] = right

    def release_all_controls(self):
        """Release all control keys."""
        held_keys['w'] = False
        held_keys['s'] = False
        held_keys['a'] = False
        held_keys['d'] = False

    def start_recording(self, target_checkpoint=None):
        """
        Start recording trajectory and optionally detect checkpoint crossing.

        Args:
            target_checkpoint (dict, optional): Checkpoint definition to detect crossing
        """
        self.trajectory = []
        self.recording = True
        self.target_checkpoint = target_checkpoint
        self.checkpoint_crossed = False
        self.crossing_frame = None
        self.frame_count = 0

    def stop_recording(self):
        """Stop recording trajectory."""
        self.recording = False

    def record_frame(self):
        """
        Record current car state.
        Called each frame during recording.
        """
        if not self.recording:
            return

        frame_data = {
            'frame': self.frame_count,
            'position': (self.car.world_position.x, self.car.world_position.y, self.car.world_position.z),
            'speed': self.car.speed,
            'angle': self.car.rotation_y
        }

        self.trajectory.append(frame_data)

        # Check checkpoint crossing
        if self.target_checkpoint is not None and not self.checkpoint_crossed:
            if len(self.trajectory) >= 2:
                self._check_checkpoint_crossing()

        self.frame_count += 1

    def _check_checkpoint_crossing(self):
        """Check if checkpoint was crossed between last two frames."""
        prev_frame = self.trajectory[-2]
        curr_frame = self.trajectory[-1]

        pos1 = prev_frame['position']
        pos2 = curr_frame['position']

        if check_line_crossing(pos1, pos2, self.target_checkpoint):
            self.checkpoint_crossed = True
            self.crossing_frame = curr_frame['frame']
            print(f"✓ Checkpoint crossed at frame {self.crossing_frame}")

    def get_simulation_result(self):
        """
        Get simulation result for fitness calculation.

        Returns:
            dict: {
                'checkpoint_crossed': bool,
                'frames_taken': int,
                'trajectory': list,
                'final_position': tuple
            }
        """
        result = {
            'checkpoint_crossed': self.checkpoint_crossed,
            'frames_taken': self.crossing_frame if self.checkpoint_crossed else self.frame_count,
            'trajectory': self.trajectory.copy(),
            'final_position': self.trajectory[-1]['position'] if self.trajectory else None
        }

        return result


# Test function
if __name__ == "__main__":
    print("DirectController module loaded")
    print("Use this in conjunction with the game")