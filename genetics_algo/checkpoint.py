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


def filter_and_renumber_checkpoints(checkpoints, checkpoint_ids_to_remove):
    """
    Filter out specific checkpoints and renumber the remaining ones sequentially.

    Args:
        checkpoints: List of checkpoint dictionaries (each has 'checkpoint_id' key)
        checkpoint_ids_to_remove: List of checkpoint IDs to remove

    Returns:
        tuple: (filtered_checkpoints, new_finish_line_id)
            - filtered_checkpoints: List with removed checkpoints filtered out and IDs renumbered
            - new_finish_line_id: The new checkpoint ID for the finish line

    Raises:
        ValueError: If trying to remove all checkpoints or invalid checkpoint IDs
    """
    if not checkpoint_ids_to_remove:
        # No checkpoints to remove, return original list with finish line ID
        if not checkpoints:
            raise ValueError("No checkpoints provided")
        max_id = max(cp['checkpoint_id'] for cp in checkpoints)
        return checkpoints, max_id

    # Validation: Check if all removal IDs are valid
    existing_ids = {cp['checkpoint_id'] for cp in checkpoints}
    invalid_ids = [cid for cid in checkpoint_ids_to_remove if cid not in existing_ids]
    if invalid_ids:
        print(f"Warning: Checkpoint IDs {invalid_ids} do not exist and will be ignored")

    # Filter out checkpoints to remove
    filtered = [cp for cp in checkpoints if cp['checkpoint_id'] not in checkpoint_ids_to_remove]

    # Validation: Ensure at least one checkpoint remains
    if not filtered:
        raise ValueError(f"Cannot remove all checkpoints! Attempted to remove {len(checkpoint_ids_to_remove)} "
                        f"from {len(checkpoints)} total checkpoints")

    # Store original finish line checkpoint (the one with highest ID)
    original_finish_id = max(cp['checkpoint_id'] for cp in checkpoints)
    was_finish_removed = original_finish_id in checkpoint_ids_to_remove

    if was_finish_removed:
        print(f"Warning: Original finish line (checkpoint {original_finish_id}) was removed. "
              f"The last remaining checkpoint will become the new finish line.")

    # Renumber checkpoints sequentially starting from 1
    # Keep the same order, just update the checkpoint_id
    for new_id, checkpoint in enumerate(filtered, start=1):
        old_id = checkpoint['checkpoint_id']
        checkpoint['checkpoint_id'] = new_id
        if old_id == original_finish_id and not was_finish_removed:
            # Track the new ID of the original finish line
            new_finish_line_id = new_id

    # The new finish line is the last checkpoint in the filtered list
    new_finish_line_id = len(filtered)

    print(f"Checkpoint filtering: {len(checkpoints)} → {len(filtered)} checkpoints")
    print(f"Removed checkpoint IDs: {sorted([cid for cid in checkpoint_ids_to_remove if cid in existing_ids])}")
    print(f"New finish line checkpoint ID: {new_finish_line_id}")

    return filtered, new_finish_line_id
