import sys
from pathlib import Path
import pickle
import lzma
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_checkpoint_line(position, angle, width=50.0):
    """
    Calculate a perpendicular line at the given position and angle.
    This represents a checkpoint line (like finish line).

    Args:
        position: (x, y, z) tuple
        angle: Orientation angle in degrees
        width: Width of the checkpoint line

    Returns:
        Dictionary with line endpoints and metadata
    """
    x, y, z = position

    # Calculate perpendicular angle (90 degrees to car's facing direction)
    perpendicular_angle = angle + 90.0

    # Convert to radians for trigonometry
    angle_rad = np.radians(perpendicular_angle)

    # Calculate half-width offset in X and Z
    half_width = width / 2.0
    dx = half_width * np.sin(angle_rad)
    dz = half_width * np.cos(angle_rad)

    # Calculate the two endpoints of the line
    point_a = (x - dx, y, z - dz)
    point_b = (x + dx, y, z + dz)

    return {
        'center': (x, y, z),
        'point_a': point_a,
        'point_b': point_b,
        'width': width,
        'orientation': perpendicular_angle
    }


def check_line_crossing(pos1, pos2, line):
    """
    Check if trajectory from pos1 to pos2 crosses the given line.
    Uses 2D line intersection in the X-Z plane (top-down view).
    """
    # Extract 2D positions (ignore Y, work in X-Z plane)
    x1, z1 = pos1[0], pos1[2]
    x2, z2 = pos2[0], pos2[2]

    # Extract line endpoints (ignore Y)
    ax, az = line['point_a'][0], line['point_a'][2]
    bx, bz = line['point_b'][0], line['point_b'][2]

    def ccw(A, B, C):
        """Check if three points are counter-clockwise"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = (x1, z1)
    B = (x2, z2)
    C = (ax, az)
    D = (bx, bz)

    # Two segments intersect if endpoints are on opposite sides
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def filter_initial_stationary_frames(snapshots, speed_threshold=0.1):
    """
    Remove initial frames where the car is stationary AND has no input.
    """
    if not snapshots:
        return snapshots

    # Find first frame where car is moving OR has input
    start_idx = 0
    for i, snapshot in enumerate(snapshots):
        forward, backward, left, right = snapshot.current_controls
        has_input = any([forward, backward, left, right])
        is_moving = snapshot.car_speed > speed_threshold

        if is_moving or has_input:
            start_idx = i
            break

    return snapshots[start_idx:]


def calculate_3d_distance(pos1, pos2):
    """Calculate 3D Euclidean distance between two positions."""
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def evaluate_recorded_run(run_path, filter_stationary=True, verbose=True):
    """
    Evaluate a recorded run file, detecting lap completion and measuring performance.
    """
    # Load run data
    with lzma.open(run_path, "rb") as file:
        snapshots = pickle.load(file)

    original_length = len(snapshots)

    # Filter initial stationary frames if requested
    if filter_stationary:
        snapshots = filter_initial_stationary_frames(snapshots)
        frames_removed = original_length - len(snapshots)
        if verbose and frames_removed > 0:
            print(f"Removed {frames_removed} initial stationary frames")

    if verbose:
        print(f"\nLoaded run: {run_path.name}")
        print(f"Total frames: {len(snapshots)}")
        print()

    if len(snapshots) == 0:
        print("Error: No frames after filtering")
        return None

    # Define finish line based on starting position and orientation
    start_snapshot = snapshots[0]
    finish_line = calculate_checkpoint_line(
        start_snapshot.car_position,
        start_snapshot.car_angle,
        width=50.0
    )

    if verbose:
        print("=" * 70)
        print("START POSITION")
        print("=" * 70)
        x, y, z = start_snapshot.car_position
        print(f"Position: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"Speed: {start_snapshot.car_speed:.3f}")
        print(f"Orientation: {start_snapshot.car_angle:.1f}°")
        print(f"Controls: {start_snapshot.current_controls}")
        print()

    # Detect finish line crossings
    crossings = []
    for i in range(len(snapshots) - 1):
        pos1 = snapshots[i].car_position
        pos2 = snapshots[i+1].car_position

        if check_line_crossing(pos1, pos2, finish_line):
            crossings.append(i)

    # Analyze results
    if len(crossings) == 0:
        if verbose:
            print("No finish line crossings detected")
        return {
            'completed_lap': False,
            'crossings': [],
            'lap_frames': None,
            'lap_time': None,
            'total_frames': len(snapshots)
        }

    if verbose:
        print("=" * 70)
        print("FINISH LINE CROSSINGS")
        print("=" * 70)
        for idx, frame in enumerate(crossings):
            snapshot = snapshots[frame]
            x, y, z = snapshot.car_position
            print(f"Crossing {idx+1} at frame {frame}")
            print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f})")
            print(f"  Speed: {snapshot.car_speed:.3f}")
            print()

    # Check if lap was completed (at least 2 crossings: start + finish)
    completed_lap = len(crossings) >= 2

    if completed_lap:
        # Calculate lap statistics
        start_crossing_frame = crossings[0]
        finish_crossing_frame = crossings[1]

        lap_frames = finish_crossing_frame - start_crossing_frame
        lap_time = lap_frames * 0.1  # 10 FPS

        # Get positions at start and finish crossings
        start_pos = snapshots[start_crossing_frame].car_position
        finish_pos = snapshots[finish_crossing_frame].car_position

        # Calculate total distance traveled (sum of segment distances)
        total_distance = 0.0
        for i in range(start_crossing_frame, finish_crossing_frame):
            pos1 = snapshots[i].car_position
            pos2 = snapshots[i+1].car_position
            total_distance += calculate_3d_distance(pos1, pos2)

        # Calculate straight-line distance from start to finish
        straight_distance = calculate_3d_distance(start_pos, finish_pos)

        if verbose:
            print("=" * 70)
            print("LAP COMPLETED")
            print("=" * 70)
            print(f"Lap time: {lap_time:.2f} seconds ({lap_frames} frames)")
            print(f"Start crossing: Frame {start_crossing_frame}")
            print(f"Finish crossing: Frame {finish_crossing_frame}")
            print()
            print(f"Distance traveled: {total_distance:.2f} units")
            print(f"Straight-line distance: {straight_distance:.2f} units")
            print(f"Path efficiency: {(straight_distance / total_distance * 100):.1f}%")
            print()
            print(f"Average speed: {total_distance / lap_time:.2f} units/sec")
            print()

        return {
            'completed_lap': True,
            'crossings': crossings,
            'lap_frames': lap_frames,
            'lap_time': lap_time,
            'total_frames': len(snapshots),
            'distance_traveled': total_distance,
            'straight_distance': straight_distance,
            'start_crossing_frame': start_crossing_frame,
            'finish_crossing_frame': finish_crossing_frame,
            'start_position': start_pos,
            'finish_position': finish_pos
        }
    else:
        if verbose:
            print("=" * 70)
            print("LAP NOT COMPLETED")
            print("=" * 70)
            print(f"Only {len(crossings)} crossing(s) detected")
            print("Need at least 2 crossings to complete a lap")
            print()

        return {
            'completed_lap': False,
            'crossings': crossings,
            'lap_frames': None,
            'lap_time': None,
            'total_frames': len(snapshots)
        }


def evaluate_all_recorded_runs(runs_dir="runs"):
    """Evaluate all recorded runs in the specified directory."""
    runs_path = Path(runs_dir)

    if not runs_path.exists():
        print(f"Directory not found: {runs_dir}")
        return

    recorded_runs = sorted(runs_path.glob("*.npz"))

    if not recorded_runs:
        print(f"No .npz files found in {runs_dir}/")
        return

    print("=" * 70)
    print(f"EVALUATING RECORDED RUNS ({len(recorded_runs)} total)")
    print("=" * 70)
    print()

    results = []

    for run_path in recorded_runs:
        print(f"\n{'='*70}")
        print(f"Evaluating: {run_path.name}")
        print('='*70)

        try:
            result = evaluate_recorded_run(run_path, filter_stationary=True, verbose=True)
            if result:
                result['name'] = run_path.name
                results.append(result)
        except Exception as e:
            print(f"Error evaluating {run_path.name}: {e}")
            print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    completed = [r for r in results if r['completed_lap']]
    incomplete = [r for r in results if not r['completed_lap']]

    if completed:
        completed.sort(key=lambda r: r['lap_time'])
        print(f"\nCompleted laps ({len(completed)}):")
        for i, r in enumerate(completed, 1):
            print(f"  {i}. {r['name']}: {r['lap_time']:.2f}s ({r['lap_frames']} frames, {r['distance_traveled']:.1f} units)")

    if incomplete:
        print(f"\nIncomplete laps ({len(incomplete)}):")
        for r in incomplete:
            print(f"  - {r['name']}: {len(r['crossings'])} crossing(s), {r['total_frames']} frames")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate recorded runs")
    parser.add_argument("--all", action="store_true", help="Evaluate all recorded runs")
    parser.add_argument("--run", type=str, help="Evaluate specific run file")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter initial stationary frames")

    args = parser.parse_args()

    if args.all:
        evaluate_all_recorded_runs()
    elif args.run:
        run_path = Path(args.run)
        if not run_path.exists():
            print(f"Run file not found: {run_path}")
            return

        evaluate_recorded_run(run_path, filter_stationary=not args.no_filter, verbose=True)
    else:
        # Default: evaluate all runs in the runs directory
        evaluate_all_recorded_runs()


if __name__ == "__main__":
    main()