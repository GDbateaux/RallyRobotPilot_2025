import sys
from pathlib import Path
import pickle
import lzma

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallyrobopilot_novisual import CollisionSystem
from genetics_algo.individual import from_game_format
from genetics_algo.simulator import simulate_individual
from genetics_algo import config


class ControlSnapshot:
    def __init__(self, forward, backward, left, right):
        self.current_controls = (forward, backward, left, right)


def check_finish_line_crossing(pos1, pos2):
    x1, y1, z1 = pos1
    x2, y2, z2 = pos2

    # Check if trajectory crosses x=0 plane
    if (x1 <= 0 <= x2) or (x2 <= 0 <= x1):
        # Calculate where the crossing happens in z
        if abs(x2 - x1) > 1e-6:
            t = (0 - x1) / (x2 - x1)
            z_crossing = z1 + t * (z2 - z1)

            # Check if crossing is within finish line bounds (-25 to 25)
            if -25 <= z_crossing <= 25:
                return True

    return False


def filter_initial_stationary_frames(snapshots, speed_threshold=0.1):
    if not snapshots:
        return snapshots

    # Find first frame where car is moving OR has input
    start_idx = 0
    for i, snapshot in enumerate(snapshots):
        forward, backward, left, right = snapshot.current_controls
        has_input = any([forward, backward, left, right])

        # Check if snapshot has speed attribute (SensingSnapshot does, ControlSnapshot doesn't)
        if hasattr(snapshot, 'car_speed'):
            is_moving = snapshot.car_speed > speed_threshold
            if is_moving or has_input:
                start_idx = i
                break
        else:
            # For ControlSnapshot (GA runs), just check input
            if has_input:
                start_idx = i
                break

    filtered = snapshots[start_idx:]
    return filtered


def evaluate_run_from_file(run_path, collision_system, min_frame=50, verbose=True, filter_stationary=False):
    # Load run data (just controls, no positions)
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
        print(f"Loaded run: {run_path.name}")
        print(f"Genome length: {len(snapshots)} frames")
        print(f"Simulating...", end=" ", flush=True)

    # Convert to genome format
    genome = []
    for snapshot in snapshots:
        forward, backward, left, right = snapshot.current_controls
        axis_fb, axis_lr = from_game_format(forward, backward, left, right)
        genome.append((axis_fb, axis_lr))

    # Simulate to get positions
    result = simulate_individual(genome, collision_system)

    if verbose:
        print("✓")

    trajectory = result['trajectory']

    # Check for finish line crossing
    frames_to_finish = None
    for i in range(len(trajectory) - 1):
        frame = i + 1

        # Skip early frames
        if frame < min_frame:
            continue

        pos1 = trajectory[i]['position']
        pos2 = trajectory[i + 1]['position']

        if check_finish_line_crossing(pos1, pos2):
            frames_to_finish = frame
            break

    return {
        'frames_to_finish': frames_to_finish,
        'collision_frame': result['collision_frame'] if result['collision_detected'] else None,
        'total_frames': result['frames_simulated'],
        'finished': frames_to_finish is not None
    }


def evaluate_latest_run(ga_runs_dir="ga_runs", verbose=True):
    runs_path = Path(ga_runs_dir)

    # Find all runs
    runs = list(runs_path.glob("run_*.npz"))

    if not runs:
        print(f"✗ No runs found in {ga_runs_dir}/")
        return None

    # Get latest run
    latest_run = max(runs, key=lambda p: p.stat().st_mtime)

    if verbose:
        print("=" * 70)
        print("EVALUATING LATEST RUN")
        print("=" * 70)
        print()

    # Initialize collision system (suppress output)
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    collision_system = CollisionSystem("SimpleTrack/track_metadata.json")
    sys.stdout = old_stdout

    # Evaluate
    result = evaluate_run_from_file(latest_run, collision_system, min_frame=50, verbose=verbose)

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if result['finished']:
        print(f"✓ Lap completed in {result['frames_to_finish']} frames")
        print(f"  Time: {result['frames_to_finish'] * 0.1:.1f} seconds (at 10 FPS)")
    else:
        print(f"✗ Did not finish lap")
        if result['collision_frame']:
            print(f"  Collision at frame {result['collision_frame']}")
        else:
            print(f"  Ran for {result['total_frames']} frames without finishing")

    print()

    return result


def evaluate_recorded_lap(verbose=True):
    """Evaluate the recorded lap specified in config.RECORDED_LAP_PATH"""
    recorded_path = Path(config.RECORDED_LAP_PATH)

    if not recorded_path.exists():
        print(f"✗ Recorded lap not found: {recorded_path}")
        return None

    if verbose:
        print("=" * 70)
        print("EVALUATING RECORDED LAP FROM CONFIG")
        print(f"Path: {config.RECORDED_LAP_PATH}")
        print("=" * 70)
        print()

    # Initialize collision system (suppress output)
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    collision_system = CollisionSystem("SimpleTrack/track_metadata.json")
    sys.stdout = old_stdout

    # Evaluate with stationary frame filtering (recorded laps have idle start)
    result = evaluate_run_from_file(recorded_path, collision_system, min_frame=50, verbose=verbose, filter_stationary=True)

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if result['finished']:
        print(f"✓ Lap completed in {result['frames_to_finish']} frames")
        print(f"  Time: {result['frames_to_finish'] * 0.1:.1f} seconds (at 10 FPS)")
    else:
        print(f"✗ Did not finish lap")
        if result['collision_frame']:
            print(f"  Collision at frame {result['collision_frame']}")
        else:
            print(f"  Ran for {result['total_frames']} frames without finishing")

    print()

    return result


def evaluate_all_runs(ga_runs_dir="ga_runs", recorded_runs_dir="runs"):
    # Collect runs from both directories
    ga_runs_path = Path(ga_runs_dir)
    recorded_runs_path = Path(recorded_runs_dir)

    ga_runs = sorted(ga_runs_path.glob("run_*.npz")) if ga_runs_path.exists() else []
    recorded_runs = sorted(recorded_runs_path.glob("*.npz")) if recorded_runs_path.exists() else []

    if not ga_runs and not recorded_runs:
        print(f"✗ No runs found in {ga_runs_dir}/ or {recorded_runs_dir}/")
        return

    total_runs = len(ga_runs) + len(recorded_runs)
    print("=" * 70)
    print(f"EVALUATING ALL RUNS ({total_runs} total)")
    print(f"  GA runs: {len(ga_runs)} in {ga_runs_dir}/")
    print(f"  Recorded runs: {len(recorded_runs)} in {recorded_runs_dir}/")
    print("=" * 70)
    print()

    # Initialize collision system once (suppress output)
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    collision_system = CollisionSystem("SimpleTrack/track_metadata.json")
    sys.stdout = old_stdout

    results = []

    # Process GA runs (no filtering needed)
    if ga_runs:
        print(f"--- Evaluating GA runs from {ga_runs_dir}/ ---")
        for run_path in ga_runs:
            print(f"Evaluating {run_path.name}...", end=" ", flush=True)
            try:
                result = evaluate_run_from_file(run_path, collision_system, min_frame=50, verbose=False, filter_stationary=False)
                result['name'] = run_path.name
                result['source'] = 'GA'
                results.append(result)

                if result['finished']:
                    print(f"✓ {result['frames_to_finish']} frames")
                else:
                    print(f"✗ Did not finish")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"✗ Corrupted file (skipping)")
        print()

    # Process recorded runs (with filtering)
    if recorded_runs:
        print(f"--- Evaluating recorded runs from {recorded_runs_dir}/ ---")
        for run_path in recorded_runs:
            print(f"Evaluating {run_path.name}...", end=" ", flush=True)
            try:
                result = evaluate_run_from_file(run_path, collision_system, min_frame=50, verbose=False, filter_stationary=True)
                result['name'] = run_path.name
                result['source'] = 'Recorded'
                results.append(result)

                if result['finished']:
                    print(f"✓ {result['frames_to_finish']} frames")
                else:
                    print(f"✗ Did not finish")
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"✗ Corrupted file (skipping)")
        print()

    # Sort by completion time (finished runs first, then by frames)
    finished = [r for r in results if r['finished']]
    unfinished = [r for r in results if not r['finished']]

    finished.sort(key=lambda r: r['frames_to_finish'])

    print("=" * 70)
    print("RANKINGS")
    print("=" * 70)
    print()

    if finished:
        print("Completed laps (fastest first):")
        for i, r in enumerate(finished, 1):
            time_sec = r['frames_to_finish'] * 0.1
            source_label = f"[{r['source']}]"
            print(f"  {i}. {source_label:12} {r['name']}: {r['frames_to_finish']} frames ({time_sec:.1f}s)")

    if unfinished:
        print()
        print("Did not finish:")
        for r in unfinished:
            source_label = f"[{r['source']}]"
            if r['collision_frame']:
                print(f"  - {source_label:12} {r['name']}: collision at frame {r['collision_frame']}")
            else:
                print(f"  - {source_label:12} {r['name']}: ran {r['total_frames']} frames")

    print()

    # Return best run info
    if finished:
        best = finished[0]
        print(f"🏆 Fastest run: [{best['source']}] {best['name']} - {best['frames_to_finish']} frames ({best['frames_to_finish'] * 0.1:.1f}s)")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GA runs")
    parser.add_argument("--all", action="store_true", help="Evaluate all runs from ga_runs/ and runs/ and rank them")
    parser.add_argument("--run", type=str, help="Evaluate specific run file")
    parser.add_argument("--recorded", action="store_true", help="Evaluate the recorded lap from config.RECORDED_LAP_PATH")

    args = parser.parse_args()

    if args.all:
        evaluate_all_runs()
    elif args.recorded:
        evaluate_recorded_lap()
    elif args.run:
        run_path = Path(args.run)
        if not run_path.exists():
            print(f"✗ Run file not found: {run_path}")
            return

        # Initialize collision system (suppress output)
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        collision_system = CollisionSystem("SimpleTrack/track_metadata.json")
        sys.stdout = old_stdout

        result = evaluate_run_from_file(run_path, collision_system, verbose=True)

        print()
        if result['finished']:
            time_sec = result['frames_to_finish'] * 0.1
            print(f"✓ Completed in {result['frames_to_finish']} frames ({time_sec:.1f}s)")
        else:
            print(f"✗ Did not finish")
        print()
    else:
        evaluate_latest_run()


if __name__ == "__main__":
    main()
