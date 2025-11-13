"""
Simple replay test - validates DirectController matches recorded run exactly.
Can replay from any frame range.
"""

import sys
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent / "genetics_algo"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallyrobopilot import prepare_game_app, DirectController
from panda3d.core import ClockObject
from genetics_algo.lap_data import load_recorded_lap


# ============ CONFIGURATION ============
START_FRAME = 23   # Change this to start from any frame (e.g., 103)
END_FRAME = None   # Set to a number to stop early, or None for full replay


# Debug range - print detailed info for these frames
DEBUG_START = 170
DEBUG_END = 190

# Control frame offset - adjust if controls seem early/late
# 0 = apply controls from current_frame
# +1 = apply controls from current_frame + 1 (1 frame ahead)
# This is needed because controls are recorded AFTER being applied
CONTROL_FRAME_OFFSET = 0
# =======================================


# Global state
controller = None
recorded_data = None
current_frame = 0
simulation_active = False


def setup_simulation():
    """Load recorded lap and setup replay from START_FRAME."""
    global recorded_data, current_frame, simulation_active

    print("\n=== Replaying Recorded Lap ===\n")

    # Load recorded data
    recorded_data = load_recorded_lap()

    # Determine end frame
    actual_end_frame = END_FRAME if END_FRAME is not None else len(recorded_data)

    print(f"Loaded {len(recorded_data)} frames")
    print(f"START_FRAME = {START_FRAME}")
    print(f"Will replay from frame {START_FRAME + 1} to {actual_end_frame}")
    print(f"Control offset: {CONTROL_FRAME_OFFSET}")
    print(f"Debug output for frames {DEBUG_START} to {DEBUG_END}\n")

    # Only teleport if not starting from frame 0
    if START_FRAME > 0:
        # Get the state at START_FRAME
        start_snapshot = recorded_data[START_FRAME]

        print(f"Teleporting to frame {START_FRAME}:")
        print(f"  Position: ({start_snapshot.car_position[0]:.2f}, {start_snapshot.car_position[2]:.2f})")
        print(f"  Speed: {start_snapshot.car_speed:.5f}")
        print(f"  Angle: {start_snapshot.car_angle:.1f}\n")

        # Teleport to exact recorded state
        controller.reset_to_state(
            position=start_snapshot.car_position,
            angle=start_snapshot.car_angle,
            speed=start_snapshot.car_speed
        )

        # Start recording from this point
        controller.start_recording()

        # Manually record the teleport state
        initial_frame = {
            'frame': START_FRAME,
            'position': (controller.car.world_position.x, controller.car.world_position.y, controller.car.world_position.z),
            'speed': controller.car.speed,
            'angle': controller.car.rotation_y
        }
        controller.trajectory.append(initial_frame)
        controller.frame_count = START_FRAME + 1
    else:
        # Starting from frame 0 - car is already at default position
        print("Starting from frame 0 (default position)\n")
        controller.start_recording()

        # Record initial state
        initial_frame = {
            'frame': 0,
            'position': (controller.car.world_position.x, controller.car.world_position.y, controller.car.world_position.z),
            'speed': controller.car.speed,
            'angle': controller.car.rotation_y
        }
        controller.trajectory.append(initial_frame)
        controller.frame_count = 1

    # Start replay
    current_frame = START_FRAME + 1
    simulation_active = True

    print("Starting replay...\n")


def apply_next_control():
    """
    Apply controls BEFORE car update.
    Controls are offset by +1 because they were recorded AFTER being applied.
    """
    global current_frame, simulation_active, recorded_data, controller

    if not simulation_active or not recorded_data:
        return

    # Determine end frame
    actual_end_frame = END_FRAME if END_FRAME is not None else len(recorded_data)

    # Check if done
    if current_frame >= actual_end_frame:
        print(f"\n✓ Replay complete! Frames {START_FRAME + 1} to {current_frame}")
        finish_simulation()
        return

    # Apply controls with offset
    control_frame = current_frame + CONTROL_FRAME_OFFSET

    if 0 <= control_frame < len(recorded_data):
        snapshot = recorded_data[control_frame]
        forward, backward, left, right = snapshot.current_controls
        controller.apply_controls(forward, backward, left, right)

        # Debug output in range
        if DEBUG_START <= current_frame <= DEBUG_END:
            print(f"\n=== FRAME {current_frame} DEBUG (offset={CONTROL_FRAME_OFFSET}) ===")
            print(f"  Applying controls FROM frame {control_frame}: F={forward} B={backward} L={left} R={right}")
            if controller.trajectory:
                last = controller.trajectory[-1]
                print(f"  Current pos: ({last['position'][0]:.2f}, {last['position'][2]:.2f})")
                print(f"  Current speed: {last['speed']:.2f}")
                print(f"  Current angle: {last['angle']:.1f}")
            # Show what the recorded data says for THIS frame
            curr_snapshot = recorded_data[current_frame]
            print(f"  Recorded pos: ({curr_snapshot.car_position[0]:.2f}, {curr_snapshot.car_position[2]:.2f})")
            print(f"  Recorded speed: {curr_snapshot.car_speed:.2f}")
            print(f"  Recorded angle: {curr_snapshot.car_angle:.1f}")
    else:
        controller.release_all_controls()

    current_frame += 1


def finish_simulation():
    """End simulation."""
    global simulation_active

    simulation_active = False
    controller.release_all_controls()
    controller.stop_recording()

    trajectory = controller.trajectory
    print(f"\n=== Complete ===")
    print(f"Recorded frames: {len(trajectory)}")
    if trajectory:
        print(f"Start: frame {trajectory[0]['frame']}, pos=({trajectory[0]['position'][0]:.1f}, {trajectory[0]['position'][2]:.1f})")
        print(f"End: frame {trajectory[-1]['frame']}, pos=({trajectory[-1]['position'][0]:.1f}, {trajectory[-1]['position'][2]:.1f})")
    print("\n✓ Press ESC to quit.")


# Monkey-patch to apply controls BEFORE car processes them
original_car_update = None

def patched_car_update(self):
    """Apply controls first, then run normal update."""
    apply_next_control()
    original_car_update(self)


if __name__ == "__main__":
    print("=== DirectController Replay Test (Cleaned) ===\n")

    app, car = prepare_game_app("SimpleTrack/track_metadata.json")

    clock = ClockObject.getGlobalClock()
    clock.setMode(ClockObject.MLimited)
    clock.setFrameRate(10)

    controller = DirectController(car)
    car.direct_controller = controller

    from rallyrobopilot.car import Car
    original_car_update = Car.update
    Car.update = patched_car_update

    print("✓ Ready\n")

    # Setup and start simulation immediately
    setup_simulation()

    app.run()