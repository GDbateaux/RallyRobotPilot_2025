"""
Replay evolved GA controls in visual mode.

Automatically finds the latest GA run and replays the evolved controls.
"""

from PyQt6 import QtWidgets
from data_collector import DataCollectionUI
import pickle
import lzma
import time
from pathlib import Path
import argparse


class ControlSnapshot:
    """Simple snapshot with controls for replay."""
    def __init__(self, forward, backward, left, right):
        self.current_controls = (forward, backward, left, right)


class ReplayController:
    def __init__(self, recording_path="record_0.npz"):
        print(f"Loading recording from: {recording_path}")
        with lzma.open(recording_path, "rb") as file:
            self.recorded_data = pickle.load(file)

        print(f"Loaded {len(self.recorded_data)} frames")
        self.current_frame = 0

        self.start_time = None
        self.frame_times = []

    def get_next_controls(self, message):
        if self.current_frame >= len(self.recorded_data):
            return [("forward", False), ("back", False), ("left", False), ("right", False)]

        snapshot = self.recorded_data[self.current_frame]
        forward, backward, left, right = snapshot.current_controls

        current_time = time.time()

        if self.start_time is None:
            self.start_time = current_time
        else:
            dt = current_time - self.frame_times[-1] if self.frame_times else 0
            self.frame_times.append(current_time)

            if self.current_frame % 10 == 0:
                avg_dt = sum([self.frame_times[i] - self.frame_times[i-1]
                              for i in range(1, len(self.frame_times))]) / (len(self.frame_times) - 1) if len(self.frame_times) > 1 else 0
                print(f"Frame {self.current_frame}: dt={dt:.3f}s, avg={avg_dt:.3f}s (should be 0.1s)")

        commands = [
            ("forward", bool(forward)),
            ("back", bool(backward)),
            ("left", bool(left)),
            ("right", bool(right))
        ]

        self.current_frame += 1
        return commands

    def process_message(self, message, data_collector):
        commands = self.get_next_controls(message)
        for command, start in commands:
            data_collector.onCarControlled(command, start)


def find_latest_ga_run():
    """Find the latest GA run directory."""
    ga_runs_dir = Path("ga_runs")
    if not ga_runs_dir.exists():
        return None

    # Find all run directories
    run_dirs = [d for d in ga_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None

    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    return run_dirs[0]


def find_evolved_file(segment=0, run_dir=None):
    """
    Find the evolved control file for a specific segment.

    Checks multiple locations:
    1. run_dir/controls/segment_X.npz (new organized structure)
    2. ga_runs/segment_X.npz (old location)
    3. ga_runs/full_lap_evolved_TIMESTAMP.npz (full lap file)

    Args:
        segment (int): Segment number (default: 0)
        run_dir (Path): Specific run directory (default: latest)

    Returns:
        Path: Path to evolved file, or None if not found
    """
    # Try new organized structure first
    if run_dir:
        controls_file = run_dir / "controls" / f"segment_{segment}.npz"
        if controls_file.exists():
            return controls_file

    # Try old location in ga_runs root
    ga_runs_file = Path("ga_runs") / f"segment_{segment}.npz"
    if ga_runs_file.exists():
        return ga_runs_file

    # Try full lap evolved file
    ga_runs_dir = Path("ga_runs")
    full_lap_files = list(ga_runs_dir.glob("full_lap_evolved_*.npz"))
    if full_lap_files:
        # Return newest
        full_lap_files.sort(reverse=True)
        return full_lap_files[0]

    return None


if __name__ == "__main__":
    import sys

    # Setup exception hook
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    # Parse arguments
    parser = argparse.ArgumentParser(description='Replay evolved GA controls')
    parser.add_argument('--segment', type=int, default=0,
                       help='Segment to replay (default: 0)')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific .npz file to replay')
    parser.add_argument('--list-runs', action='store_true',
                       help='List available GA runs and exit')
    args = parser.parse_args()

    # List runs mode
    if args.list_runs:
        ga_runs_dir = Path("ga_runs")
        run_dirs = [d for d in ga_runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        run_dirs.sort(reverse=True)

        print("Available GA runs:")
        for i, run_dir in enumerate(run_dirs):
            summary_file = run_dir / "summary.txt"
            if summary_file.exists():
                # Parse summary for quick info
                with open(summary_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Timestamp:" in line:
                            timestamp = line.split(":")[1].strip()
                        elif "Total improvement:" in line:
                            improvement = line.split(":")[1].strip()

                print(f"{i+1}. {run_dir.name} - Improvement: {improvement}")
            else:
                print(f"{i+1}. {run_dir.name}")

        # List .npz files in root
        print("\nEvolved files in ga_runs/:")
        for npz_file in ga_runs_dir.glob("*.npz"):
            print(f"  - {npz_file.name}")

        sys.exit(0)

    # Determine file to replay
    if args.file:
        recording_path = Path(args.file)
        if not recording_path.exists():
            print(f"Error: File not found: {recording_path}")
            sys.exit(1)
    else:
        # Auto-find latest run
        latest_run = find_latest_ga_run()
        if latest_run:
            print(f"Found latest GA run: {latest_run}")
            recording_path = find_evolved_file(segment=args.segment, run_dir=latest_run)
        else:
            recording_path = find_evolved_file(segment=args.segment)

        if not recording_path:
            print(f"Error: Could not find evolved file for segment {args.segment}")
            print(f"\nAvailable options:")
            print(f"  1. Specify file manually: python {sys.argv[0]} --file path/to/file.npz")
            print(f"  2. List available runs: python {sys.argv[0]} --list-runs")
            sys.exit(1)

    print("="*70)
    print("REPLAYING EVOLVED GA CONTROLS")
    print("="*70)
    print(f"File: {recording_path}")
    print("="*70)
    print()

    # Create Qt application
    app = QtWidgets.QApplication(sys.argv)

    # Create replay controller
    replay_brain = ReplayController(recording_path=str(recording_path))
    data_window = DataCollectionUI(replay_brain.process_message)

    # Enable autopilot
    data_window.autopiloting = True
    data_window.AutopilotButton.setText("AutoPilot:\nON")

    data_window.show()
    app.exec()
