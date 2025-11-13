"""
Headless Physics Demo - Pure Python Rally Car Simulation

This demonstrates the physics-only simulation with NO visual components.
Shows car state updates (position, speed, angle) as it drives.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallyrobopilot_novisual import CarPhysics, TrackData, DirectController, SimulationLoop


def print_header():
    """Print demo header."""
    print("=" * 70)
    print("HEADLESS PHYSICS DEMO - Pure Python Rally Car Simulation")
    print("=" * 70)
    print("NO window | NO rendering | NO GPU | Pure CPU calculations")
    print("=" * 70)
    print()


def print_track_info(track):
    """Print track information."""
    print("TRACK INFORMATION")
    print("-" * 70)
    print(f"Track Name: {track.track_name}")
    print(f"Start Position: {track.get_start_position()}")
    print(f"Start Orientation: {track.get_start_orientation()}")
    print(f"Checkpoints Available: {len(track.checkpoints)} checkpoints")
    print()


def print_car_info(car, frame=0):
    """Print current car state."""
    print(f"Frame {frame:4d} | "
          f"Pos: ({car.x:7.2f}, {car.y:7.2f}, {car.z:7.2f}) | "
          f"Speed: {car.speed:6.2f} | "
          f"Angle: {car.rotation_y:6.1f}° | "
          f"RotSpeed: {car.rotation_speed:5.2f}")


def run_demo():
    """Run headless physics demonstration."""
    print_header()

    # Setup track and car
    print("Setting up simulation...")
    track = TrackData("SimpleTrack/track_metadata.json")
    print_track_info(track)
    # Create car at starting position
    start_pos = track.get_start_position()
    start_rot = track.get_start_orientation()
    car = CarPhysics(position=start_pos, rotation_y=start_rot[1])

    # Create controller and simulation loop
    controller = DirectController(car)
    sim_loop = SimulationLoop(controller)

    print("CAR INITIAL STATE")
    print("-" * 70)
    print(f"Position: ({car.x:.2f}, {car.y:.2f}, {car.z:.2f})")
    print(f"Speed: {car.speed:.2f}")
    print(f"Angle: {car.rotation_y:.1f}°")
    print(f"Rotation Speed: {car.rotation_speed:.2f}")
    print()

    # Test 1: Drive forward for 50 frames
    print("TEST 1: Driving forward (50 frames)")
    print("-" * 70)
    controller.apply_controls(forward=True, backward=False, left=False, right=False)

    for frame in range(50):
        sim_loop.step()
        if frame % 10 == 0:  # Print every 10 frames
            print_car_info(car, frame)

    print()

    # Test 2: Drive forward + turn right for 30 frames
    print("TEST 2: Driving forward + turning right (30 frames)")
    print("-" * 70)
    controller.apply_controls(forward=True, backward=False, left=False, right=True)

    for frame in range(50, 80):
        sim_loop.step()
        if frame % 10 == 0:
            print_car_info(car, frame)

    print()

    # Test 3: Coast (no controls) for 30 frames
    print("TEST 3: Coasting (no controls, 30 frames)")
    print("-" * 70)
    controller.release_all_controls()

    for frame in range(80, 110):
        sim_loop.step()
        if frame % 10 == 0:
            print_car_info(car, frame)

    print()

    # Test 4: Brake for 20 frames
    print("TEST 4: Braking (20 frames)")
    print("-" * 70)
    controller.apply_controls(forward=False, backward=True, left=False, right=False)

    for frame in range(110, 130):
        sim_loop.step()
        if frame % 10 == 0:
            print_car_info(car, frame)

    print()

    # Final state
    print("FINAL CAR STATE")
    print("-" * 70)
    print(f"Position: ({car.x:.2f}, {car.y:.2f}, {car.z:.2f})")
    print(f"Speed: {car.speed:.2f}")
    print(f"Angle: {car.rotation_y:.1f}°")
    print(f"Rotation Speed: {car.rotation_speed:.2f}")

    # Calculate distance traveled
    distance = ((car.x - start_pos[0])**2 +
                (car.z - start_pos[2])**2)**0.5
    print(f"Distance from start: {distance:.2f} units")
    print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("✓ Pure Python physics simulation")
    print("✓ No window, no rendering, no GPU")
    print("✓ Same physics as visual game (0.1s timestep)")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()