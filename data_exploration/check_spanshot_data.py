"""
Check what data is available in recorded snapshots.
"""

import pickle
import lzma

# Load recorded data
record_file = "record_5.npz"
print(f"Loading: {record_file}\n")

with lzma.open(record_file, "rb") as file:
    data = pickle.load(file)

print(f"Loaded {len(data)} snapshots\n")

# Check first snapshot
sample_snapshot = data[0]

print("=== Frame 0 Attributes ===")
for attr in dir(sample_snapshot):
    if not attr.startswith('_'):
        try:
            value = getattr(sample_snapshot, attr)
            if not callable(value):
                print(f"{attr}: {type(value).__name__} = {value}")
        except:
            print(f"{attr}: <could not access>")

print("\n=== Checking rotation_speed across multiple frames ===")
frames_to_check = [0, 50, 100, 150, 200]
for frame_idx in frames_to_check:
    if frame_idx < len(data):
        snapshot = data[frame_idx]
        rs = snapshot.rotation_speed if hasattr(snapshot, 'rotation_speed') else 'N/A'
        print(f"Frame {frame_idx}: rotation_speed = {rs:.4f}, speed = {snapshot.car_speed:.2f}, angle = {snapshot.car_angle:.1f}")

print("\n=== Statistics ===")
if hasattr(data[0], 'rotation_speed'):
    rotation_speeds = [s.rotation_speed for s in data]
    print(f"Min rotation_speed: {min(rotation_speeds):.4f}")
    print(f"Max rotation_speed: {max(rotation_speeds):.4f}")
    print(f"Non-zero count: {sum(1 for rs in rotation_speeds if abs(rs) > 0.01)}/{len(rotation_speeds)}")
else:
    print("rotation_speed not found in data")