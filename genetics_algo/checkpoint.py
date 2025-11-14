import numpy as np

def create_checkpoints_from_data(data, num_checkpoints, checkpoint_width):
    total_frames = len(data)
    checkpoints = []

    # Create intermediate checkpoints evenly distributed
    # If num_checkpoints=4, place at: 20%, 40%, 60% (skip last one that's too close to finish)
    for i in range(1, num_checkpoints):
        # Calculate frame index
        frame_idx = int((i / (num_checkpoints + 1)) * total_frames)
        frame_idx = min(frame_idx, total_frames - 1)  # Safety bounds check

        snapshot = data[frame_idx]
        checkpoint = calculate_checkpoint_line(
            snapshot.car_position,
            snapshot.car_angle,
            checkpoint_width
        )
        checkpoint['frame_index'] = frame_idx
        checkpoint['checkpoint_id'] = i

        checkpoints.append(checkpoint)

    return checkpoints


def calculate_checkpoint_line(position, angle, width):
    x, y, z = position

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


def check_line_crossing(pos1, pos2, checkpoint_line):
    # Extract 2D positions (ignore Y, work in X-Z plane)
    x1, z1 = pos1[0], pos1[2]
    x2, z2 = pos2[0], pos2[2]

    # Extract line endpoints (ignore Y)
    ax, az = checkpoint_line['point_a'][0], checkpoint_line['point_a'][2]
    bx, bz = checkpoint_line['point_b'][0], checkpoint_line['point_b'][2]

    def ccw(A, B, C):
        """Check if three points are counter-clockwise"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = (x1, z1)
    B = (x2, z2)
    C = (ax, az)
    D = (bx, bz)

    # Two segments intersect if endpoints are on opposite sides
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
