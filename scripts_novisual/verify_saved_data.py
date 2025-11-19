"""
Verification script to check if saved GA runs contain position and speed data.

This script loads a .npz file and checks what attributes are available in the snapshots.
"""

import sys
from pathlib import Path
import pickle
import lzma

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Define ControlSnapshot for unpickling (matches the class in GA scripts)
class ControlSnapshot:
    def __init__(self, forward, backward, left, right, car_position=None, car_speed=None):
        self.current_controls = (forward, backward, left, right)
        self.car_position = car_position if car_position is not None else (0, 0, 0)
        self.car_speed = car_speed if car_speed is not None else 0.0


def verify_run_data(run_path):
    """
    Verify what data is saved in a run file.

    Args:
        run_path: Path to .npz file
    """
    print("=" * 70)
    print(f"VERIFYING: {run_path}")
    print("=" * 70)
    print()

    # Load run data
    with lzma.open(run_path, "rb") as file:
        snapshots = pickle.load(file)

    print(f"Total frames: {len(snapshots)}")
    print()

    if not snapshots:
        print("✗ No snapshots found in file")
        return

    # Check first snapshot
    first_snapshot = snapshots[0]
    print("First snapshot attributes:")

    # Check for controls
    if hasattr(first_snapshot, 'current_controls'):
        print(f"  ✓ current_controls: {first_snapshot.current_controls}")
    else:
        print("  ✗ current_controls: NOT FOUND")

    # Check for position
    if hasattr(first_snapshot, 'car_position'):
        print(f"  ✓ car_position: {first_snapshot.car_position}")
    else:
        print("  ✗ car_position: NOT FOUND (old format)")

    # Check for speed
    if hasattr(first_snapshot, 'car_speed'):
        print(f"  ✓ car_speed: {first_snapshot.car_speed}")
    else:
        print("  ✗ car_speed: NOT FOUND (old format)")

    print()

    # Sample a few frames to verify data
    if hasattr(first_snapshot, 'car_position') and hasattr(first_snapshot, 'car_speed'):
        print("Sample frames (showing position and speed):")
        sample_indices = [0, len(snapshots) // 4, len(snapshots) // 2, 3 * len(snapshots) // 4, len(snapshots) - 1]

        for idx in sample_indices:
            if idx < len(snapshots):
                snap = snapshots[idx]
                pos = snap.car_position
                speed = snap.car_speed
                print(f"  Frame {idx:3d}: pos=({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}), speed={speed:6.2f}")
        print()
        print("✓ File contains NEW FORMAT with position and speed data")
    else:
        print("✗ File contains OLD FORMAT (controls only, no position/speed)")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify saved run data format")
    parser.add_argument("run_file", nargs="?", help="Path to run file to verify")
    parser.add_argument("--latest", action="store_true", help="Verify latest run from ga_runs/")

    args = parser.parse_args()

    if args.latest:
        # Find latest run
        ga_runs_dir = Path("ga_runs")
        runs = list(ga_runs_dir.glob("run_*.npz"))
        if not runs:
            print("✗ No runs found in ga_runs/")
            return
        run_path = max(runs, key=lambda p: p.stat().st_mtime)
    elif args.run_file:
        run_path = Path(args.run_file)
    else:
        # Default: check latest run
        ga_runs_dir = Path("ga_runs")
        runs = list(ga_runs_dir.glob("run_*.npz"))
        if not runs:
            print("✗ No runs found in ga_runs/")
            print("Usage: python verify_saved_data.py [run_file]")
            return
        run_path = max(runs, key=lambda p: p.stat().st_mtime)

    if not run_path.exists():
        print(f"✗ File not found: {run_path}")
        return

    verify_run_data(run_path)


if __name__ == "__main__":
    main()
