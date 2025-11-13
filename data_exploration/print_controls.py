"""
Print all frame indices with their associated controls.
"""

import pickle
import lzma

# Load recorded data
record_file = "record_6.npz"
print(f"Loading: {record_file}\n")

with lzma.open(record_file, "rb") as file:
    data = pickle.load(file)

print(f"Loaded {len(data)} snapshots\n")
print("=" * 100)
print(f"{'Frame':<8} {'Forward':<10} {'Backward':<10} {'Left':<10} {'Right':<10} {'Speed':<12}")
print("=" * 100)

for idx, snapshot in enumerate(data):
    forward, backward, left, right = snapshot.current_controls
    speed = snapshot.car_speed
    print(f"{idx:<8} {forward:<10} {backward:<10} {left:<10} {right:<10} {speed:<12.4f}")

print("=" * 100)
print(f"\nTotal frames: {len(data)}")