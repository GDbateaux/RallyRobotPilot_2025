from PyQt6 import QtWidgets
from data_collector import DataCollectionUI
import pickle
import lzma
import time


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
                              for i in range(1, len(self.frame_times))]) / (len(self.frame_times) - 1)
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


if __name__ == "__main__":
    import sys
    from pathlib import Path

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    # Find the latest evolved run or use default recorded lap
    default_recording = "runs/simpletrack_recorded_1.npz"
    ga_runs_dir = Path("ga_runs")

    recording_path = default_recording  # Default to recorded lap

    if ga_runs_dir.exists():
        # Find all run_*.npz files
        evolved_runs = list(ga_runs_dir.glob("run_*.npz"))
        if evolved_runs:
            # Extract run numbers and find the highest
            run_numbers = []
            for f in evolved_runs:
                try:
                    num = int(f.stem.split('_')[1])
                    run_numbers.append((num, f))
                except (IndexError, ValueError):
                    continue

            if run_numbers:
                # Select the run with highest number
                latest_run = max(run_numbers, key=lambda x: x[0])[1]
                recording_path = str(latest_run)
                print(f"Found evolved run: {recording_path}")
            else:
                print(f"No evolved runs found, using default: {recording_path}")
        else:
            print(f"No evolved runs found, using default: {recording_path}")
    else:
        print(f"ga_runs folder not found, using default: {recording_path}")

    app = QtWidgets.QApplication(sys.argv)

    replay_brain = ReplayController(recording_path=recording_path)
    data_window = DataCollectionUI(replay_brain.process_message)

    data_window.autopiloting = True
    data_window.AutopilotButton.setText("AutoPilot:\nON")

    data_window.show()
    app.exec()