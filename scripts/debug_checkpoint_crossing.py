"""
Debug script to diagnose why individuals aren't crossing checkpoints.

This script will:
1. Show where checkpoint 0 is located
2. Run the elite individual (exact recorded controls)
3. Print car position every frame
4. Show checkpoint line geometry
5. Test if crossing detection is working
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.lap_data import get_segment_info, get_checkpoints
from genetics_algo.individual import Individual
from genetics_algo.simulator import setup_game_for_simulation
from genetics_algo.checkpoint import check_line_crossing

def main():
    print("="*60)
    print("CHECKPOINT CROSSING DEBUG")
    print("="*60)
    print()

    # Get checkpoint info
    print("Step 1: Loading checkpoint data...")
    checkpoints = get_checkpoints()
    print(f"Total checkpoints: {len(checkpoints)}\n")

    # Focus on checkpoint 0
    checkpoint_0 = checkpoints[0]
    print("Checkpoint 0 details:")
    print(f"  Center: {checkpoint_0['center']}")
    print(f"  Point A: {checkpoint_0['point_a']}")
    print(f"  Point B: {checkpoint_0['point_b']}")
    print(f"  Crossing frame (in recording): {checkpoint_0['crossing_frame']}")
    print()

    # Get segment 0 info
    print("Step 2: Loading segment 0...")
    segment_info = get_segment_info(0)
    print(f"  Start frame: {segment_info['start_frame']}")
    print(f"  End frame: {segment_info['end_frame']}")
    print(f"  Length: {segment_info['length']} frames")
    print(f"  Start position: {segment_info['start_state']['position']}")
    print()

    # Create elite individual
    print("Step 3: Creating elite individual (exact recorded controls)...")
    elite = Individual.from_segment(0)
    print(f"  Controls: {len(elite)} frames")
    print()

    # Setup game
    print("Step 4: Setting up game...")
    app, car, controller, simulator = setup_game_for_simulation()
    print()

    # Manually run simulation with detailed output
    print("Step 5: Running simulation with frame-by-frame output...")
    print("="*60)

    # Reset car to start
    start_state = segment_info['start_state']
    controller.reset_to_state(
        position=start_state['position'],
        angle=start_state['angle'],
        speed=start_state['speed']
    )

    # Wait for game initialization
    print("Waiting for initialization...")
    import time
    start_time = time.time()
    while time.time() - start_time < 2.0:
        app.taskMgr.step()
        time.sleep(0.01)
    print("Ready!\n")

    # Get controls
    controls = elite.get_game_controls()

    print(f"Frame | Position (x, y, z) | Speed | Crossed?")
    print("-" * 60)

    prev_pos = start_state['position']

    # Run simulation frame by frame
    for frame in range(min(len(controls), 40)):  # Max 40 frames for debug
        # Apply control
        if frame < len(controls):
            forward, backward, left, right = controls[frame]
            controller.apply_controls(forward, backward, left, right)

        # Step game
        app.taskMgr.step()

        # Get current position
        current_pos = (
            controller.car.world_position.x,
            controller.car.world_position.y,
            controller.car.world_position.z
        )
        speed = controller.car.speed

        # Check crossing
        crossed = check_line_crossing(prev_pos, current_pos, checkpoint_0)

        marker = " *** CROSSED! ***" if crossed else ""
        print(f"{frame:4d}  | ({current_pos[0]:6.2f}, {current_pos[1]:6.2f}, {current_pos[2]:6.2f}) | {speed:5.2f} | {marker}")

        if crossed:
            print(f"\n✓ Checkpoint crossed at frame {frame}!")
            print(f"  Previous pos: {prev_pos}")
            print(f"  Current pos: {current_pos}")
            break

        prev_pos = current_pos
    else:
        print(f"\n✗ Checkpoint NOT crossed after {frame+1} frames")
        print(f"  Final position: {current_pos}")
        print(f"  Checkpoint center: {checkpoint_0['center']}")

    print("\n" + "="*60)
    print("Keep game window open to inspect. Close manually when done.")


if __name__ == "__main__":
    main()