#!/usr/bin/env python3
"""
Replay .npz Recording and Save With Images

This script replays a recorded .npz file (from GA runs or manual recordings)
on the visual game and captures all data WITH images from the visual engine.

Usage:
    # Terminal 1: Start visual game
    python scripts/main.py

    # Terminal 2: Replay and save with images
    python scripts/replay_and_save.py --input ga_runs/run_10.npz --output replayed_with_images.npz

The script:
1. Loads the input .npz file
2. Connects to the running visual game
3. Replays controls frame-by-frame
4. Captures SensingSnapshot messages (with images) from the game
5. Saves all captured data to output .npz file
"""

import argparse
import pickle
import lzma
import time
import os
import sys
from pathlib import Path

# Add parent directory to path to import rallyrobopilot
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallyrobopilot import NetworkDataCmdInterface


class ReplayAndSaveController:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        # Load input recording
        print(f"\n[1/5] Loading recording from: {input_path}")
        with lzma.open(input_path, "rb") as file:
            self.recorded_data = pickle.load(file)
        print(f"      Loaded {len(self.recorded_data)} frames")

        # Check if input has images
        has_images = any(snapshot.image is not None for snapshot in self.recorded_data[:10])
        print(f"      Input file {'HAS' if has_images else 'DOES NOT HAVE'} images")

        # Replay state
        self.current_frame = 0
        self.start_time = None
        self.frame_times = []

        # Captured data (new snapshots with images from visual game)
        self.captured_data = []

        # Network interface (will be initialized after connection)
        self.network_interface = None
        self.recording_started = False

    def connect_to_game(self):
        """Connect to the running visual game"""
        print("\n[2/5] Connecting to visual game (main.py must be running)...")
        try:
            self.network_interface = NetworkDataCmdInterface(
                self.on_snapshot_received,
                address="127.0.0.1",
                port=7654
            )
            print("      Connected successfully!")

            # Enable recording mode so game captures images
            print("      Enabling recording mode (images will be captured)...")
            self.network_interface.send_cmd("set recording true;")
            time.sleep(0.1)  # Give game time to process command

            return True

        except Exception as e:
            print(f"      ERROR: Could not connect to game: {e}")
            print("      Make sure 'python scripts/main.py' is running in another terminal!")
            return False

    def on_snapshot_received(self, snapshot):
        """Callback when a SensingSnapshot is received from the game"""
        # Only start recording after we've sent the first command
        if self.recording_started:
            self.captured_data.append(snapshot)

            # Show progress
            if len(self.captured_data) % 10 == 0:
                has_image = snapshot.image is not None
                image_info = f"image: {snapshot.image.shape}" if has_image else "NO IMAGE"
                print(f"      Frame {len(self.captured_data)}/{len(self.recorded_data)}: {image_info}")

    def replay_recording(self):
        """Replay the recording frame by frame"""
        print(f"\n[3/5] Replaying {len(self.recorded_data)} frames...")
        print("      (This will take approximately {:.1f} seconds)".format(len(self.recorded_data) * 0.1))

        self.recording_started = True
        self.start_time = time.time()

        # Reset car position to start
        if len(self.recorded_data) > 0:
            first_snapshot = self.recorded_data[0]
            pos = first_snapshot.car_position
            angle = first_snapshot.car_angle

            self.network_interface.send_cmd(f"set position {pos[0]},{pos[1]},{pos[2]};")
            self.network_interface.send_cmd(f"set rotation {angle};")
            self.network_interface.send_cmd("reset;")
            time.sleep(0.2)  # Wait for reset to complete

        # Replay frame by frame
        for frame_idx, snapshot in enumerate(self.recorded_data):
            self.current_frame = frame_idx

            # Get controls from recorded snapshot
            forward, backward, left, right = snapshot.current_controls

            # Send control commands
            self._send_controls(forward, backward, left, right)

            # Receive messages from game (triggers on_snapshot_received callback)
            self.network_interface.recv_msg()

            # Wait for frame period (0.1s = 10 fps)
            time.sleep(0.1)

            # Show progress every 50 frames
            if (frame_idx + 1) % 50 == 0:
                elapsed = time.time() - self.start_time
                progress = (frame_idx + 1) / len(self.recorded_data) * 100
                print(f"      Progress: {frame_idx + 1}/{len(self.recorded_data)} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")

        # Release all controls at end
        self.network_interface.send_cmd("release all;")

        # Wait a bit for final messages
        time.sleep(0.2)
        self.network_interface.recv_msg()

        # Disable recording mode
        self.network_interface.send_cmd("set recording false;")

        print(f"\n      Replay complete! Captured {len(self.captured_data)} frames")

    def _send_controls(self, forward, backward, left, right):
        """Send control commands to the game"""
        # Handle forward/backward (mutually exclusive in practice)
        if forward:
            self.network_interface.send_cmd("push forward;")
        else:
            self.network_interface.send_cmd("release forward;")

        if backward:
            self.network_interface.send_cmd("push back;")
        else:
            self.network_interface.send_cmd("release back;")

        # Handle left/right steering
        if left:
            self.network_interface.send_cmd("push left;")
        else:
            self.network_interface.send_cmd("release left;")

        if right:
            self.network_interface.send_cmd("push right;")
        else:
            self.network_interface.send_cmd("release right;")

    def save_captured_data(self):
        """Save captured data to output .npz file"""
        print(f"\n[4/5] Saving captured data to: {self.output_path}")

        if len(self.captured_data) == 0:
            print("      WARNING: No data was captured! Output file will be empty.")
            return False

        # Check if we got images
        has_images = any(snapshot.image is not None for snapshot in self.captured_data)
        print(f"      Captured {len(self.captured_data)} frames")
        print(f"      Images: {'YES' if has_images else 'NO - WARNING: Game may not have sent images!'}")

        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"      Created output directory: {output_dir}")

        # Save with lzma compression (same format as original recordings)
        try:
            with lzma.open(self.output_path, "wb") as f:
                pickle.dump(self.captured_data, f)

            # Get file size
            file_size = os.path.getsize(self.output_path)
            size_mb = file_size / (1024 * 1024)
            print(f"      Saved successfully! File size: {size_mb:.2f} MB")
            return True

        except Exception as e:
            print(f"      ERROR: Could not save file: {e}")
            return False

    def run(self):
        """Main execution flow"""
        print("\n" + "="*80)
        print("REPLAY AND SAVE WITH IMAGES")
        print("="*80)

        # Connect to game
        if not self.connect_to_game():
            return False

        # Replay and capture
        try:
            self.replay_recording()
        except KeyboardInterrupt:
            print("\n\n      Interrupted by user!")
            self.network_interface.send_cmd("release all;")
            self.network_interface.send_cmd("set recording false;")
        except Exception as e:
            print(f"\n\n      ERROR during replay: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Save captured data
        if not self.save_captured_data():
            return False

        print("\n[5/5] DONE!")
        print("="*80)
        print(f"\nSuccessfully replayed and saved:")
        print(f"  Input:  {self.input_path}")
        print(f"  Output: {self.output_path}")
        print(f"  Frames: {len(self.captured_data)}")
        print("\n" + "="*80 + "\n")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Replay .npz recording and save with images from visual game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay a GA run and save with images
  python scripts/replay_and_save.py --input ga_runs/run_10.npz --output replayed_run10.npz

  # Replay a manual recording
  python scripts/replay_and_save.py --input runs/simpletrack_recorded_1.npz --output simpletrack_with_images.npz

Prerequisites:
  The visual game must be running! Start it in another terminal:
    python scripts/main.py
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input .npz file to replay (e.g., ga_runs/run_10.npz)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output .npz file to save (e.g., replayed_with_images.npz)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Warn if output file already exists
    if os.path.exists(args.output):
        response = input(f"WARNING: Output file already exists: {args.output}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    # Run replay and save
    controller = ReplayAndSaveController(args.input, args.output)
    success = controller.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
