#!/usr/bin/env python3
"""
Replay GA Run With Screenshots

This script replays a GA run (.npz with ControlSnapshot objects) on the visual
game engine and captures ALL sensor data INCLUDING screenshots, producing output
that matches data_collector.py format exactly.

Usage:
    python scripts/replay_with_screenshots.py --input ga_runs/run_0.npz --output runs_visual/run_0_visual.npz

The script:
1. Loads the GA run file (ControlSnapshot objects)
2. Creates the visual game engine locally (Ursina)
3. Replays controls frame-by-frame
4. Captures all sensor data + screenshots
5. Saves as SensingSnapshot objects (same format as data_collector.py)
"""

import argparse
import pickle
import lzma
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import game components
from rallyrobopilot import prepare_game_app
from rallyrobopilot.sensing_message import SensingSnapshot
from ursina import held_keys, application

# Import ControlSnapshot from ga_autopilot
sys.path.insert(0, str(Path(__file__).parent))
from ga_autopilot import ControlSnapshot


class ReplayWithScreenshots:
    def __init__(self, input_path, output_path, max_frames=None):
        self.input_path = input_path
        self.output_path = output_path
        self.max_frames = max_frames

        # Load GA run data
        print(f"\n[1/5] Loading GA run from: {input_path}")
        with lzma.open(input_path, "rb") as f:
            self.control_data = pickle.load(f)

        # Limit frames if specified
        if self.max_frames is not None and self.max_frames < len(self.control_data):
            print(f"      Loaded {len(self.control_data)} frames (will process first {self.max_frames})")
            self.control_data = self.control_data[:self.max_frames]
        else:
            print(f"      Loaded {len(self.control_data)} frames")

        # Verify it's ControlSnapshot format
        if len(self.control_data) > 0:
            if not hasattr(self.control_data[0], 'current_controls'):
                print(f"      ERROR: Invalid format - expected ControlSnapshot objects")
                sys.exit(1)
            print(f"      Format: ControlSnapshot ✓")

        # Storage for captured data
        self.captured_data = []

        # Game components (initialized later)
        self.app = None
        self.car = None

    def setup_game(self):
        """Initialize the visual game engine"""
        print("\n[2/5] Setting up visual game engine...")

        from genetics_algo.config import TRACK_NAME

        # Create game app with visual rendering
        print(f"      Loading track: {TRACK_NAME}")
        self.app, self.car = prepare_game_app(f"{TRACK_NAME}/track_metadata.json")

        # Verify car has required components
        if not hasattr(self.car, 'multiray_sensor'):
            print("      ERROR: Car missing multiray_sensor!")
            sys.exit(1)

        print(f"      Game initialized ✓")
        print(f"      Car sensors: {len(self.car.multiray_sensor.rays)} rays")

    def replay_and_capture(self):
        """Replay controls and capture all data + screenshots"""
        print(f"\n[3/5] Replaying and capturing...")
        print(f"      Total frames to process: {len(self.control_data)}")

        # Access Panda3D base (global object, not imported)
        # This is available after prepare_game_app() creates the Ursina application
        import builtins
        if not hasattr(builtins, 'base'):
            print("      ERROR: Panda3D base not available!")
            sys.exit(1)
        base = builtins.base

        # Reset car to start position
        self.car.reset_car()

        # Wait for window to fully initialize and settle
        print("      Waiting for window to settle...")
        import time
        time.sleep(1.0)

        # Step a few frames to ensure window resize is complete
        for _ in range(10):
            self.app.step()

        print("      Window settled, starting capture...")

        # Process each frame
        for frame_idx, control_snapshot in enumerate(self.control_data):
            # Apply controls from recording
            forward, backward, left, right = control_snapshot.current_controls
            held_keys['w'] = bool(forward)
            held_keys['s'] = bool(backward)
            held_keys['a'] = bool(left)
            held_keys['d'] = bool(right)

            # Step physics and rendering ONE frame
            # This updates car position, speed, rotation, etc.
            self.app.step()

            # Create SensingSnapshot (EXACTLY like RemoteController does)
            snapshot = SensingSnapshot()

            # Current controls (what keys are pressed)
            snapshot.current_controls = control_snapshot.current_controls

            # Car state
            snapshot.car_position = (self.car.x, self.car.y, self.car.z)
            snapshot.car_speed = self.car.speed
            snapshot.car_angle = self.car.rotation_y

            # Rotation speed (angular velocity)
            if hasattr(self.car, 'rotation_speed'):
                snapshot.rotation_speed = self.car.rotation_speed
            else:
                snapshot.rotation_speed = 0.0

            # Proximity sensors (raycast distances)
            snapshot.raycast_distances = self.car.multiray_sensor.collect_sensor_values()

            # Capture screenshot from Ursina (EXACTLY like RemoteController does)
            try:
                tex = base.win.getDisplayRegion(0).getScreenshot()
                arr = tex.getRamImageAs("RGB")
                data = np.frombuffer(arr, np.uint8)
                image = data.reshape(tex.getYSize(), tex.getXSize(), 3)
                snapshot.image = image[::-1, :, :]  # Flip Y axis (image arrives inverted)
            except Exception as e:
                print(f"      WARNING: Could not capture screenshot at frame {frame_idx}: {e}")
                snapshot.image = None

            # Add to captured data
            self.captured_data.append(snapshot)

            # Progress indicator
            if (frame_idx + 1) % 50 == 0:
                progress = (frame_idx + 1) / len(self.control_data) * 100
                has_image = snapshot.image is not None
                img_size = f"{snapshot.image.shape}" if has_image else "NO IMAGE"
                print(f"      Frame {frame_idx + 1}/{len(self.control_data)} ({progress:.1f}%) - Image: {img_size}")

        print(f"\n      Capture complete! Captured {len(self.captured_data)} frames")

        # Check image capture success
        frames_with_images = sum(1 for s in self.captured_data if s.image is not None)
        print(f"      Frames with images: {frames_with_images}/{len(self.captured_data)}")

    def save_output(self):
        """Save captured data to output file"""
        print(f"\n[4/5] Saving to: {self.output_path}")

        if len(self.captured_data) == 0:
            print("      ERROR: No data captured!")
            return False

        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"      Created directory: {output_dir}")

        # Save using same format as data_collector.py
        try:
            with lzma.open(self.output_path, "wb") as f:
                pickle.dump(self.captured_data, f)

            # Report file size
            file_size = os.path.getsize(self.output_path)
            size_mb = file_size / (1024 * 1024)
            print(f"      Saved successfully!")
            print(f"      File size: {size_mb:.2f} MB")

            return True

        except Exception as e:
            print(f"      ERROR: Could not save file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Clean up and close game"""
        print("\n[5/5] Cleaning up...")

        # Release all keys
        held_keys['w'] = False
        held_keys['s'] = False
        held_keys['a'] = False
        held_keys['d'] = False

        # Quit application
        if self.app is not None:
            application.quit()

        print("      Done!")

    def run(self):
        """Main execution flow"""
        print("\n" + "="*80)
        print("REPLAY GA RUN WITH SCREENSHOTS")
        print("="*80)

        try:
            # Setup game engine
            self.setup_game()

            # Replay and capture
            self.replay_and_capture()

            # Save output
            if not self.save_output():
                return False

            # Summary
            print("\n" + "="*80)
            print("SUCCESS!")
            print("="*80)
            print(f"\nInput:  {self.input_path}")
            print(f"Output: {self.output_path}")
            print(f"Frames: {len(self.captured_data)}")

            # Check data completeness
            frames_with_images = sum(1 for s in self.captured_data if s.image is not None)
            print(f"\nData captured per frame:")
            print(f"  ✓ Controls (forward/back/left/right)")
            print(f"  ✓ Position (x, y, z)")
            print(f"  ✓ Speed + Angle + Rotation speed")
            print(f"  ✓ Proximity sensors ({len(self.captured_data[0].raycast_distances)} rays)")
            print(f"  ✓ Screenshots: {frames_with_images}/{len(self.captured_data)} frames")

            print("\n" + "="*80 + "\n")

            return True

        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            return False

        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Always cleanup
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Replay GA run and save with screenshots (matching data_collector.py format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay a GA run and save with screenshots
  python scripts/replay_with_screenshots.py --input ga_runs/run_0.npz --output runs_visual/run_0_visual.npz

  # Replay an optimal run
  python scripts/replay_with_screenshots.py --input optimal_runs_DONOTDELETE/run_1.npz --output runs_visual/optimal_run1.npz

Output Format:
  The output file contains a list of SensingSnapshot objects with:
  - current_controls (forward, backward, left, right)
  - car_position (x, y, z)
  - car_speed, car_angle, rotation_speed
  - raycast_distances (15 proximity sensor readings)
  - image (screenshot numpy array: height × width × 3)

  This matches EXACTLY the format produced by data_collector.py when
  recording with "save images" checkbox enabled.
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input .npz file (GA run with ControlSnapshot objects)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output .npz file (will contain SensingSnapshot objects with screenshots)"
    )

    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: process all frames). Example: --max-frames 400"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Warn if output exists
    if os.path.exists(args.output):
        response = input(f"WARNING: Output file exists: {args.output}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    # Run replay
    replayer = ReplayWithScreenshots(args.input, args.output, max_frames=args.max_frames)
    success = replayer.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
