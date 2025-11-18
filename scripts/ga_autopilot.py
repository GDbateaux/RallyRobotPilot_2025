from PyQt6 import QtWidgets
from data_collector import DataCollectionUI
import pickle
import lzma
import time

from genetics_algo.config import (
    RECORDED_LAP_PATH
)


class ControlSnapshot:
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

        # DEBUG: Track collision state
        self.collision_detected = False
        self.first_collision_logged = False
        self.previous_speed = 0
        self.speed_drop_threshold = 5.0  # Sudden speed drop indicating collision

        # Open collision log file
        self.collision_log_file = open("collision_debug.log", "w")
        self.collision_log_file.write("=== VISUAL GAME COLLISION DEBUG LOG ===\n\n")

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

        # DEBUG: Check if car is accessible
        if self.current_frame == 5:  # Only log once
            if hasattr(data_collector, 'car'):
                self.collision_log_file.write(f"DEBUG: data_collector.car exists: {data_collector.car is not None}\n")
                self.collision_log_file.write(f"DEBUG: data_collector type: {type(data_collector)}\n")
                self.collision_log_file.write(f"DEBUG: data_collector attributes: {dir(data_collector)}\n")
                self.collision_log_file.flush()
            else:
                self.collision_log_file.write("DEBUG: data_collector has NO 'car' attribute!\n")
                self.collision_log_file.flush()

        # DEBUG: Log collision information
        if hasattr(data_collector, 'car') and data_collector.car is not None:
            car = data_collector.car

            # Multiple collision detection methods
            collision_reasons = []

            # Method 1: hitting_wall flag (vertical collisions)
            if hasattr(car, 'hitting_wall') and car.hitting_wall:
                collision_reasons.append("hitting_wall=True")

            # Method 2: Sudden speed drop
            speed_drop = abs(self.previous_speed - car.speed)
            if speed_drop > self.speed_drop_threshold and self.previous_speed > 5.0:
                collision_reasons.append(f"speed_drop={speed_drop:.1f}")
            self.previous_speed = car.speed

            # Method 3: Very close obstacle via boxcast
            try:
                from ursina import boxcast
                front_check = boxcast(
                    origin=car.world_position,
                    direction=car.forward,
                    thickness=(0.1, 0.1),
                    distance=car.scale_x + 2.0,  # Check 2 units ahead
                    ignore=[car,]
                )
                if front_check.hit and front_check.distance < car.scale_x + 0.5:
                    collision_reasons.append(f"obstacle_ahead={front_check.distance:.2f}")
            except:
                pass

            # If any collision detected and not yet logged
            if collision_reasons and not self.first_collision_logged:
                self.first_collision_logged = True

                # Write to both console and log file
                def log(msg):
                    print(msg)
                    self.collision_log_file.write(msg + "\n")
                    self.collision_log_file.flush()

                log("\n" + "="*80)
                log("🚨 COLLISION DETECTED IN VISUAL GAME")
                log("="*80)
                log(f"Detection methods: {', '.join(collision_reasons)}")
                log(f"Frame: {self.current_frame}")
                log(f"Position: ({car.x:.2f}, {car.y:.2f}, {car.z:.2f})")
                log(f"Speed: {car.speed:.2f} units/s (previous: {self.previous_speed:.2f})")
                log(f"Rotation Y: {car.rotation_y:.2f}°")
                log(f"Forward vector: ({car.forward.x:.3f}, {car.forward.y:.3f}, {car.forward.z:.3f})")

                # Check raycast sensors if available
                if hasattr(car, 'multiray_sensor'):
                    sensor_values = car.multiray_sensor.collect_sensor_values()
                    log(f"\nProximity Sensors (15 rays):")
                    log(f"  Min distance: {min(sensor_values):.2f}")
                    log(f"  Sensor values: {[f'{v:.1f}' for v in sensor_values]}")

                # Perform boxcast to see what's ahead
                try:
                    from ursina import boxcast
                    front_collision = boxcast(
                        origin=car.world_position,
                        direction=car.forward,
                        thickness=(0.1, 0.1),
                        distance=car.scale_x + 5.0,
                        ignore=[car,]
                    )
                    if front_collision.hit:
                        log(f"\nBoxcast Detection:")
                        log(f"  Hit: {front_collision.hit}")
                        log(f"  Distance: {front_collision.distance:.2f}")
                        log(f"  Entity: {front_collision.entity}")
                        if hasattr(front_collision, 'world_point'):
                            log(f"  Hit point: ({front_collision.world_point.x:.2f}, {front_collision.world_point.y:.2f}, {front_collision.world_point.z:.2f})")
                        if hasattr(front_collision, 'world_normal'):
                            log(f"  Hit normal: ({front_collision.world_normal.x:.3f}, {front_collision.world_normal.y:.3f}, {front_collision.world_normal.z:.3f})")
                except Exception as e:
                    log(f"  Boxcast error: {e}")

                log("="*80 + "\n")
                log("\n>>> Check collision_debug.log for full details <<<\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    # Find the latest evolved run or use default recorded lap
    default_recording = RECORDED_LAP_PATH
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
        print(f"ga_runs_backup folder not found, using default: {recording_path}")

    app = QtWidgets.QApplication(sys.argv)

    replay_brain = ReplayController(recording_path=recording_path)
    data_window = DataCollectionUI(replay_brain.process_message)

    data_window.autopiloting = True
    data_window.AutopilotButton.setText("AutoPilot:\nON")

    data_window.show()
    app.exec()