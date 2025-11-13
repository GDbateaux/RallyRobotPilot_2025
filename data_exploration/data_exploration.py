#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import lzma
import matplotlib.pyplot as plt
import glob


# In[2]:


record_file = "old_data.bak"
print(f"Loading record file: {record_file}")

with lzma.open(record_file, "rb") as file:
    data = pickle.load(file)

print(f"Loaded {len(data)} snapshots")


# ### Looking at our data

# cleaning up data
# 

# In[3]:


def cleanup_recording(data):
    cleaned_data = []

    for snapshot in data:
        # Check if car is moving or any control is pressed
        has_speed = snapshot.car_speed > 0.01  # Small threshold for floating point
        has_control = any(snapshot.current_controls)  # Any control pressed

        # Keep frame if car is moving OR controls are pressed
        if has_speed or has_control:
            cleaned_data.append(snapshot)

    print(f"Original frames: {len(data)}")
    print(f"Cleaned frames: {len(cleaned_data)}")
    print(f"Removed {len(data) - len(cleaned_data)} idle frames")

    return cleaned_data


# Clean the data
data = cleanup_recording(data)


# In[3]:


# List all attributes of a snapshot
print("=== Snapshot Attributes ===")
sample_snapshot = data[0]
for attr in dir(sample_snapshot):
    if not attr.startswith('_'):  # Skip private attributes
        value = getattr(sample_snapshot, attr)
        if not callable(value):  # Skip methods
            print(f"{attr}: {type(value).__name__}")

print("\n=== Sample Values from First Snapshot ===")
print(f"current_controls: {sample_snapshot.current_controls}")
print(f"car_position: {sample_snapshot.car_position}")
print(f"car_speed: {sample_snapshot.car_speed}")
print(f"car_angle: {sample_snapshot.car_angle}")
print(f"raycast_distances: {len(sample_snapshot.raycast_distances)} sensors")
print(f"image: {sample_snapshot.image.shape if sample_snapshot.image is not None else 'None'}")

print("\n=== Available Data Columns ===")
print("- current_controls: (forward, backward, left, right) - 4 booleans")
print("- car_position: (x, y, z) - 3 floats")
print("- car_speed: scalar float")
print("- car_angle: rotation_y in degrees")
print("- raycast_distances: list of 15 distance values")
print("- image: RGB image array (H, W, 3) or None")


# ### Printing location data

# Trying to find out which coordinate from x, y , z is constant.

# In[4]:


import matplotlib.pyplot as plt

# Extract data
indices = list(range(len(data)))
x_values = [s.car_position[0] for s in data]
y_values = [s.car_position[1] for s in data]
z_values = [s.car_position[2] for s in data]

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(indices, x_values, color='blue', label='X position', linewidth=2)
plt.plot(indices, y_values, color='orange', label='Y position', linewidth=2)
plt.plot(indices, z_values, color='green', label='Z position', linewidth=2)

plt.xlabel('Frame Index')
plt.ylabel('Position')
plt.title('Car Position Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[5]:


import matplotlib.pyplot as plt

# Extract X and Z positions
x_values = [s.car_position[0] for s in data]
z_values = [s.car_position[2] for s in data]

# Create 2D plot of the trajectory
plt.figure(figsize=(10, 10))
plt.plot(x_values, z_values, color='blue', linewidth=2, marker='o', markersize=2)

# Mark start and end
plt.scatter(x_values[0], z_values[0], color='green', s=100, label='Start', zorder=5)
plt.scatter(x_values[-1], z_values[-1], color='red', s=100, label='End', zorder=5)

plt.xlabel('X Position')
plt.ylabel('Z Position')
plt.title('Car Trajectory (Top-Down View)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Keep aspect ratio square
plt.tight_layout()
plt.show()


# In[6]:


# Get the very first snapshot (start/finish position)
start_snapshot = data[0]

print("=== Start/Finish Position ===")
print(f"X: {start_snapshot.car_position[0]}")
print(f"Y: {start_snapshot.car_position[1]}")
print(f"Z: {start_snapshot.car_position[2]}")
print(f"Speed: {start_snapshot.car_speed}")
print(f"Orientation (angle): {start_snapshot.car_angle}°")
print(f"Controls (forward, back, left, right): {start_snapshot.current_controls}")


# In[7]:


import numpy as np

def calculate_checkpoint_line(snapshot, width=10.0):
    # Extract car position and orientation
    x, y, z = snapshot.car_position
    angle = snapshot.car_angle  # In degrees

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


# Example usage with your data
finish_line = calculate_checkpoint_line(data[0], width=10.0)

print("=== Finish Line ===")
print(f"Center: {finish_line['center']}")
print(f"Point A: {finish_line['point_a']}")
print(f"Point B: {finish_line['point_b']}")
print(f"Width: {finish_line['width']}")
print(f"Orientation: {finish_line['orientation']}°")


# In[8]:


def check_line_crossing(pos1, pos2, line):
    # Extract 2D positions (ignore Y, work in X-Z plane)
    x1, z1 = pos1[0], pos1[2]
    x2, z2 = pos2[0], pos2[2]

    # Extract line endpoints (ignore Y)
    ax, az = line['point_a'][0], line['point_a'][2]
    bx, bz = line['point_b'][0], line['point_b'][2]

    # Check if line segments intersect using 2D line intersection
    # Line 1: (x1,z1) to (x2,z2) - car trajectory
    # Line 2: (ax,az) to (bx,bz) - checkpoint line

    def ccw(A, B, C):
        """Check if three points are counter-clockwise"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = (x1, z1)
    B = (x2, z2)
    C = (ax, az)
    D = (bx, bz)

    # Two segments intersect if endpoints are on opposite sides
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# Test with your recorded data
finish_line = calculate_checkpoint_line(data[0], width=10.0)

# Check when finish line is crossed
crossings = []
for i in range(len(data) - 1):
    pos1 = data[i].car_position
    pos2 = data[i+1].car_position

    if check_line_crossing(pos1, pos2, finish_line):
        crossings.append(i)
        print(f"Finish line crossed at frame {i} → {i+1}")

print(f"\nTotal crossings: {len(crossings)}")


# In[9]:


print("=== First 20 Snapshots ===")
print(f"{'Index':<6} {'Speed':<8} {'X':<12} {'Y':<12} {'Z':<12} {'Controls (F,B,L,R)'}")
print("-" * 80)

for i in range(min(20, len(data))):
    snapshot = data[i]
    x, y, z = snapshot.car_position
    speed = snapshot.car_speed
    controls = snapshot.current_controls

    # Format controls as (Forward, Backward, Left, Right)
    f, b, l, r = controls
    controls_str = f"({int(f)},{int(b)},{int(l)},{int(r)})"

    print(f"{i:<6} {speed:<8.3f} {x:<12.3f} {y:<12.3f} {z:<12.3f} {controls_str}")


# In[19]:


def create_checkpoints_with_lap(data, num_intermediate_checkpoints=4, checkpoint_width=10.0):
    # Start/finish line at frame 0
    start_finish = calculate_checkpoint_line(data[0], width=checkpoint_width)
    start_finish['frame_index'] = 0
    start_finish['checkpoint_id'] = 0
    start_finish['is_start'] = True
    start_finish['is_finish'] = False

    checkpoints = [start_finish]

    # Create intermediate checkpoints evenly distributed
    total_frames = len(data)

    for i in range(1, num_intermediate_checkpoints + 1):
        # Distribute evenly: skip checkpoint 0, place at 20%, 40%, 60%, 80% etc
        frame_idx = int((i / (num_intermediate_checkpoints + 1)) * total_frames)

        snapshot = data[frame_idx]
        checkpoint = calculate_checkpoint_line(snapshot, width=checkpoint_width)
        checkpoint['frame_index'] = frame_idx
        checkpoint['checkpoint_id'] = i
        checkpoint['is_start'] = False
        checkpoint['is_finish'] = False
        checkpoints.append(checkpoint)

    # Add finish line (same as start, but marked as finish for validation)
    finish_line = calculate_checkpoint_line(data[0], width=checkpoint_width)
    finish_line['frame_index'] = 0  # Same position as start
    finish_line['checkpoint_id'] = num_intermediate_checkpoints + 1
    finish_line['is_start'] = False
    finish_line['is_finish'] = True

    checkpoints.append(finish_line)

    return checkpoints


# Create checkpoints: Start/Finish + 4 intermediate = 6 total validation points
checkpoints = create_checkpoints_with_lap(data, num_intermediate_checkpoints=4, checkpoint_width=10.0)

# Display
print("=== Checkpoints ===")
for cp in checkpoints:
    if cp['is_start']:
        cp_type = "START"
    elif cp['is_finish']:
        cp_type = "FINISH"
    else:
        cp_type = f"CP {cp['checkpoint_id']}"
    print(f"{cp_type} at frame {cp['frame_index']}: Center {cp['center']}")


# In[20]:


import matplotlib.pyplot as plt

# Extract X and Z positions
x_values = [s.car_position[0] for s in data]
z_values = [s.car_position[2] for s in data]

# Create 2D plot of the trajectory
plt.figure(figsize=(10, 10))
plt.plot(x_values, z_values, color='blue', linewidth=2, marker='o', markersize=2, label='Track')

# Mark start and end
plt.scatter(x_values[0], z_values[0], color='green', s=100, label='Start', zorder=5)
plt.scatter(x_values[-1], z_values[-1], color='red', s=100, label='End', zorder=5)

# Draw checkpoint lines
for cp in checkpoints:
    # Extract X and Z coordinates from point_a and point_b
    ax, az = cp['point_a'][0], cp['point_a'][2]
    bx, bz = cp['point_b'][0], cp['point_b'][2]

    # Draw line from point A to point B
    color = 'red' if cp['is_finish'] else 'orange'
    linewidth = 3 if cp['is_finish'] else 2
    label = 'Finish Line' if cp['is_finish'] else ('Checkpoint' if cp['checkpoint_id'] == 1 else None)

    plt.plot([ax, bx], [az, bz], color=color, linewidth=linewidth, label=label, zorder=4)

    # Optional: Mark checkpoint center
    plt.scatter(cp['center'][0], cp['center'][2], color=color, s=50, marker='x', zorder=5)

plt.xlabel('X Position')
plt.ylabel('Z Position')
plt.title('Car Trajectory with Checkpoints (Top-Down View)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')  # Keep aspect ratio square
plt.tight_layout()
plt.show()


# In[21]:


def validate_checkpoint_order(data, checkpoints):
    current_checkpoint_idx = 0  # Next checkpoint we're expecting
    checkpoints_passed = []
    crossing_frames = []

    # Go through trajectory frame by frame
    for i in range(len(data) - 1):
        pos1 = data[i].car_position
        pos2 = data[i+1].car_position

        # Check if we crossed the CURRENT expected checkpoint
        if current_checkpoint_idx < len(checkpoints):
            expected_checkpoint = checkpoints[current_checkpoint_idx]

            if check_line_crossing(pos1, pos2, expected_checkpoint):
                # Passed the expected checkpoint!
                checkpoints_passed.append(expected_checkpoint['checkpoint_id'])
                crossing_frames.append(i)
                current_checkpoint_idx += 1

                print(f"✓ Checkpoint {expected_checkpoint['checkpoint_id']} passed at frame {i}")

    # Check if we passed ALL checkpoints
    all_passed = (current_checkpoint_idx == len(checkpoints))
    failed_at = None if all_passed else current_checkpoint_idx

    result = {
        'valid': all_passed,
        'checkpoints_passed': checkpoints_passed,
        'crossing_frames': crossing_frames,
        'failed_at': failed_at,
        'total_checkpoints': len(checkpoints),
        'checkpoints_completed': len(checkpoints_passed)
    }

    return result


# Test with your recorded data
print("=== Validating Recorded Run ===")
validation = validate_checkpoint_order(data, checkpoints)

print("\n=== Validation Results ===")
print(f"Valid run: {validation['valid']}")
print(f"Checkpoints passed: {validation['checkpoints_completed']}/{validation['total_checkpoints']}")
print(f"Checkpoint IDs: {validation['checkpoints_passed']}")
print(f"Crossing frames: {validation['crossing_frames']}")

if not validation['valid']:
    print(f"❌ Failed at checkpoint {validation['failed_at']}")
else:
    print("✅ All checkpoints passed in correct order!")

    # Calculate lap time (from first to last checkpoint)
    lap_frames = validation['crossing_frames'][-1] - validation['crossing_frames'][0]
    lap_time = lap_frames * 0.1  # 10 FPS = 0.1s per frame
    print(f"Lap time: {lap_time:.2f} seconds ({lap_frames} frames)")


# In[ ]:




