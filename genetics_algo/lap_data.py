"""
Load and provide access to recorded lap data.
Handles all data extraction from recorded lap file.
"""

import pickle
import lzma
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent))

from config import RECORDED_LAP_PATH, NUM_CHECKPOINTS, CHECKPOINT_WIDTH, SEGMENT_OVERLAP_FRAMES, SEGMENT_TIMEOUT_BUFFER

# Import SensingSnapshot for saving evolved runs
sys.path.insert(0, str(Path(__file__).parent.parent))
from rallyrobopilot.sensing_message import SensingSnapshot

# Global cache for loaded data
_cached_data = None
_cached_checkpoints = None  # ADD THIS LINE


def load_recorded_lap(filepath=None):
    """
    Load the recorded lap data from file.

    Args:
        filepath (str): Path to recorded lap file. If None, uses RECORDED_LAP_PATH from config

    Returns:
        list: List of SensingSnapshot objects
    """
    global _cached_data

    if filepath is None:
        filepath = RECORDED_LAP_PATH

    print(f"Loading recorded lap from: {filepath}")

    try:
        with lzma.open(filepath, "rb") as file:
            data = pickle.load(file)

        print(f"✓ Loaded {len(data)} snapshots")
        return data

    except FileNotFoundError:
        print(f"✗ Error: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def validate_snapshot(snapshot):
    """
    Validate that a snapshot has all required attributes.

    Args:
        snapshot: SensingSnapshot object

    Returns:
        tuple: (is_valid, missing_attributes)
    """
    required_attributes = [
        'current_controls',
        'car_position',
        'car_speed',
        'car_angle',
    ]

    missing = []
    for attr in required_attributes:
        if not hasattr(snapshot, attr):
            missing.append(attr)

    return len(missing) == 0, missing


def validate_all_snapshots(data):
    """
    Validate all snapshots in the dataset.

    Args:
        data (list): List of snapshots

    Returns:
        bool: True if all valid
    """
    print("=== Validating Snapshots ===")

    all_valid = True
    for i, snapshot in enumerate(data):
        is_valid, missing = validate_snapshot(snapshot)
        if not is_valid:
            print(f"✗ Snapshot {i} missing: {missing}")
            all_valid = False

    if all_valid:
        print(f"✓ All {len(data)} snapshots are valid")

    return all_valid


def cleanup_recording(data):
    """
    Remove idle frames at the beginning of the race.
    Keep only the LAST idle frame before movement starts.

    Idle frame = speed == 0.0 AND no controls pressed

    Args:
        data (list): List of snapshots

    Returns:
        list: Cleaned data
    """
    print("=== Cleaning Recording ===")

    # Find consecutive idle frames at the start
    idle_count = 0
    for snapshot in data:
        is_idle = (snapshot.car_speed == 0.0) and not any(snapshot.current_controls)

        if is_idle:
            idle_count += 1
        else:
            # Movement started, stop counting
            break

    if idle_count > 1:
        # Keep only the last idle frame (remove idle_count - 1 frames)
        frames_to_remove = idle_count - 1
        cleaned_data = data[frames_to_remove:]
        print(f"Removed {frames_to_remove} idle frames from start")
        print(f"Original frames: {len(data)}")
        print(f"Cleaned frames: {len(cleaned_data)}")
        return cleaned_data
    else:
        print(f"No idle frames to remove (found {idle_count} idle frames at start)")
        print(f"Total frames: {len(data)}")
        return data

def get_checkpoints():
    """
    Get or create checkpoint list from recorded lap.
    Calculated once and cached for thread-safe reading.

    Returns:
        list: List of checkpoint dicts, each containing:
            - center, point_a, point_b: checkpoint geometry
            - width, orientation: checkpoint properties
            - frame_index: frame where checkpoint is placed in recording
            - checkpoint_id: ID (1 to NUM_CHECKPOINTS)
            - crossing_frame: frame where checkpoint was crossed in recorded lap
    """
    global _cached_data, _cached_checkpoints

    # Return cached if available
    if _cached_checkpoints is not None:
        return _cached_checkpoints

    # Load and clean data if needed
    if _cached_data is None:
        _cached_data = load_recorded_lap()
        if _cached_data:
            _cached_data = cleanup_recording(_cached_data)

    if not _cached_data:
        return []

    # Import here to avoid circular dependency
    from checkpoint import create_checkpoints_from_data, check_line_crossing, calculate_checkpoint_line

    # Create finish line from frame 0 (start position)
    finish_snapshot = _cached_data[0]
    finish_line = calculate_checkpoint_line(
        finish_snapshot.car_position,
        finish_snapshot.car_angle,
        CHECKPOINT_WIDTH
    )
    finish_line['frame_index'] = 0
    finish_line['checkpoint_id'] = NUM_CHECKPOINTS

    # Create intermediate checkpoints (not including finish)
    intermediate_checkpoints = create_checkpoints_from_data(
        _cached_data,
        NUM_CHECKPOINTS - 1,
        CHECKPOINT_WIDTH
    )

    # Combine all checkpoints
    all_checkpoints = intermediate_checkpoints + [finish_line]

    # Detect crossing frames for each checkpoint (SILENT)
    search_start_frame = 0

    for checkpoint in all_checkpoints:
        crossing_found = False
        for i in range(search_start_frame, len(_cached_data) - 1):
            pos1 = _cached_data[i].car_position
            pos2 = _cached_data[i + 1].car_position

            if check_line_crossing(pos1, pos2, checkpoint):
                checkpoint['crossing_frame'] = i
                crossing_found = True
                search_start_frame = i + 1
                break

        if not crossing_found:
            checkpoint['crossing_frame'] = None

    # Cache and return
    _cached_checkpoints = all_checkpoints
    return _cached_checkpoints

def get_checkpoint_geometry(checkpoint_id):
    """
    Get ONLY the geometric definition of a checkpoint.

    This function extracts just the checkpoint line geometry WITHOUT any
    frame indices or segment lengths from the original recording.

    Use this for evolved segments (segment 1+) where the original recording's
    frame indices are meaningless.

    Args:
        checkpoint_id (int): Checkpoint identifier (0 to NUM_CHECKPOINTS-1)

    Returns:
        dict: {
            'checkpoint_id': int,
            'center': tuple (x, y, z),
            'point_a': tuple (x, y, z),
            'point_b': tuple (x, y, z),
            'width': float,
            'orientation': float (degrees)
        }
    """
    checkpoints = get_checkpoints()

    if checkpoint_id < 0 or checkpoint_id >= len(checkpoints):
        raise ValueError(f"Invalid checkpoint_id {checkpoint_id}. Must be 0 to {len(checkpoints)-1}")

    checkpoint = checkpoints[checkpoint_id]

    # Extract ONLY geometry (no frame indices, no crossing info)
    return {
        'checkpoint_id': checkpoint['checkpoint_id'],
        'center': checkpoint['center'],
        'point_a': checkpoint['point_a'],
        'point_b': checkpoint['point_b'],
        'width': checkpoint['width'],
        'orientation': checkpoint['orientation']
    }

def print_checkpoint_summary():
    """
    Print summary of all checkpoints.
    Useful for debugging and verification.
    """
    checkpoints = get_checkpoints()

    print("=== Checkpoint Summary ===")
    print(f"Total checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        crossing_info = f"crossed at frame {cp['crossing_frame']}" if cp['crossing_frame'] is not None else "NOT CROSSED"
        print(f"  Checkpoint {cp['checkpoint_id']}: {crossing_info}")

def clear_cache():
    """
    Clear cached data and checkpoints.
    Call this when you want to reload with different NUM_CHECKPOINTS.
    """
    global _cached_data, _cached_checkpoints
    _cached_data = None
    _cached_checkpoints = None
    print("Cache cleared")

def get_segment_info(segment_id):
    """
    Get information about a specific segment.
    Segments are defined between checkpoints with overlap.

    Args:
        segment_id (int): Segment identifier (0 to NUM_CHECKPOINTS-1)

    Returns:
        dict: {
            'segment_id': int,
            'start_frame': int,
            'end_frame': int,
            'length': int,
            'start_state': {'position': tuple, 'speed': float, 'angle': float},
            'target_checkpoint': dict (checkpoint definition),
            'recorded_controls': list of [bool, bool] pairs
        }
    """
    global _cached_data

    if _cached_data is None:
        _cached_data = load_recorded_lap()
        if _cached_data:
            _cached_data = cleanup_recording(_cached_data)

    checkpoints = get_checkpoints()

    if segment_id < 0 or segment_id >= len(checkpoints):
        raise ValueError(f"Invalid segment_id {segment_id}. Must be 0 to {len(checkpoints)-1}")

    # Determine start frame (with overlap, except for segment 0)
    if segment_id == 0:
        start_frame = 0
    else:
        # Start SEGMENT_OVERLAP_FRAMES before previous checkpoint crossing
        prev_crossing = checkpoints[segment_id - 1]['crossing_frame']
        start_frame = max(0, prev_crossing - SEGMENT_OVERLAP_FRAMES)

    # End frame is when this segment's checkpoint is crossed
    target_checkpoint = checkpoints[segment_id]
    end_frame = target_checkpoint['crossing_frame']

    if end_frame is None:
        raise ValueError(f"Checkpoint {segment_id} was not crossed in recorded lap!")

    # Extract start state
    start_snapshot = _cached_data[start_frame]
    start_state = {
        'position': start_snapshot.car_position,
        'speed': start_snapshot.car_speed,
        'angle': start_snapshot.car_angle
    }

    # Extract recorded controls for this segment
    recorded_controls = get_recorded_controls(segment_id)

    return {
        'segment_id': segment_id,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'length': end_frame - start_frame + 1,
        'start_state': start_state,
        'target_checkpoint': target_checkpoint,
        'recorded_controls': recorded_controls
    }


def get_recorded_controls(segment_id):
    """
    Extract control sequence for a specific segment.
    Keeps original 4-boolean format: (forward, backward, left, right)

    IMPORTANT: Extracts segment frames + buffer frames to give individuals
    extra time to reach the checkpoint.

    Args:
        segment_id (int): Segment identifier

    Returns:
        list: List of (forward, backward, left, right) tuples
    """
    global _cached_data

    if _cached_data is None:
        _cached_data = load_recorded_lap()
        if _cached_data:
            _cached_data = cleanup_recording(_cached_data)

    checkpoints = get_checkpoints()

    # Determine segment frame range
    if segment_id == 0:
        start_frame = 0
    else:
        prev_crossing = checkpoints[segment_id - 1]['crossing_frame']
        start_frame = max(0, prev_crossing - SEGMENT_OVERLAP_FRAMES)

    end_frame = checkpoints[segment_id]['crossing_frame']

    # CRITICAL FIX: Include buffer frames beyond checkpoint crossing
    # This gives individuals extra time to reach checkpoint if they're slower
    end_frame_with_buffer = min(
        end_frame + SEGMENT_TIMEOUT_BUFFER,
        len(_cached_data) - 1  # Don't go beyond recording
    )

    # Extract controls including buffer
    controls = []
    for i in range(start_frame, end_frame_with_buffer + 1):
        snapshot = _cached_data[i]
        controls.append(snapshot.current_controls)

    return controls


def get_all_segments():
    """
    Get information for all segments.
    Useful for overview and parallel processing.

    Returns:
        list: List of segment info dicts
    """
    checkpoints = get_checkpoints()
    return [get_segment_info(i) for i in range(len(checkpoints))]

def print_segment_summary():
    """
    Print summary of all segments.
    Useful for debugging and verification.
    """
    segments = get_all_segments()

    print("=== Segment Summary ===")
    print(f"Total segments: {len(segments)}")
    for seg in segments:
        print(f"Segment {seg['segment_id']}: frames {seg['start_frame']}->{seg['end_frame']} ({seg['length']} frames)")
        print(f"  Start: speed={seg['start_state']['speed']:.2f}, angle={seg['start_state']['angle']:.1f}")

def get_total_frames():
    """
    Get total number of frames in recorded lap.

    Returns:
        int: Total frame count
    """
    global _cached_data

    if _cached_data is None:
        _cached_data = load_recorded_lap()
        if _cached_data:
            _cached_data = cleanup_recording(_cached_data)

    return len(_cached_data) if _cached_data else 0


def get_checkpoint_crossing_frames():
    """
    Get list of frame indices where checkpoints are crossed.

    Returns:
        list: Frame indices where checkpoints are crossed
    """
    checkpoints = get_checkpoints()
    return [cp['crossing_frame'] for cp in checkpoints if cp['crossing_frame'] is not None]


# Test function
if __name__ == "__main__":
    print("=== Testing lap_data.py ===\n")

    # Load and validate data
    data = load_recorded_lap()
    if not data:
        print("Failed to load data")
        exit(1)

    validate_all_snapshots(data)
    print()

    # Clean data
    data = cleanup_recording(data)

    # Set cache for other functions
    _cached_data = data

    # Get checkpoints (SILENT - no output)
    checkpoints = get_checkpoints()

    # Now explicitly print what we want to see
    print()
    print_checkpoint_summary()

    print()
    print_segment_summary()

    # Show segment 0 details
    print(f"\n=== Segment 0 Details ===")
    seg0 = get_segment_info(0)
    print(f"Frames: {seg0['start_frame']} -> {seg0['end_frame']} ({seg0['length']} frames)")
    print(f"Start state: speed={seg0['start_state']['speed']:.2f}, angle={seg0['start_state']['angle']:.1f}")
    print(f"First 3 controls: {seg0['recorded_controls'][:3]}")

    print("\n✓ All tests passed!")


def save_evolved_segment(segment_id, best_individual, trajectory, output_dir="ga_runs"):
    """
    Save evolved segment controls to .npz file (pickled SensingSnapshot format).

    This creates a file compatible with autoreplay - same format as original recording.

    Args:
        segment_id (int): Segment identifier
        best_individual (Individual): Best evolved individual with controls
        trajectory (list): List of trajectory states [{position, angle, speed, frame}, ...]
        output_dir (str): Output directory (default: "ga_runs")

    Returns:
        str: Path to saved file
    """
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get controls in game format: [(forward, backward, left, right), ...]
    game_controls = best_individual.get_game_controls()

    # Create SensingSnapshot objects
    snapshots = []
    for i, controls in enumerate(game_controls):
        snapshot = SensingSnapshot()
        snapshot.current_controls = controls

        # Add trajectory data if available
        if i < len(trajectory):
            frame_data = trajectory[i]
            snapshot.car_position = frame_data['position']
            snapshot.car_angle = frame_data['angle']
            snapshot.car_speed = frame_data['speed']
            snapshot.rotation_speed = 0  # Not tracked, default to 0
        else:
            # Default values if trajectory not available
            snapshot.car_position = (0, 0, 0)
            snapshot.car_angle = 0
            snapshot.car_speed = 0
            snapshot.rotation_speed = 0

        # No raycast or image data
        snapshot.raycast_distances = []
        snapshot.image = None

        snapshots.append(snapshot)

    # Save to file (same format as original: pickle + lzma compression)
    output_file = output_path / f"segment_{segment_id}.npz"
    with lzma.open(output_file, "wb") as f:
        pickle.dump(snapshots, f)

    print(f"✓ Saved evolved segment {segment_id} to: {output_file}")
    print(f"  Frames: {len(snapshots)}")
    return str(output_file)


def save_evolved_full_lap(super_individual, all_trajectories, output_dir="ga_runs"):
    """
    Save full evolved lap (merged segments) to .npz file.

    Args:
        super_individual (Individual): Full lap individual with merged genome
        all_trajectories (list): List of all trajectories concatenated
        output_dir (str): Output directory (default: "ga_runs")

    Returns:
        str: Path to saved file
    """
    from pathlib import Path
    from datetime import datetime

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get controls in game format
    game_controls = super_individual.get_game_controls()

    # Create SensingSnapshot objects
    snapshots = []
    for i, controls in enumerate(game_controls):
        snapshot = SensingSnapshot()
        snapshot.current_controls = controls

        # Add trajectory data if available
        if all_trajectories and i < len(all_trajectories):
            frame_data = all_trajectories[i]
            snapshot.car_position = frame_data.get('position', (0, 0, 0))
            snapshot.car_angle = frame_data.get('angle', 0)
            snapshot.car_speed = frame_data.get('speed', 0)
            snapshot.rotation_speed = 0
        else:
            snapshot.car_position = (0, 0, 0)
            snapshot.car_angle = 0
            snapshot.car_speed = 0
            snapshot.rotation_speed = 0

        snapshot.raycast_distances = []
        snapshot.image = None

        snapshots.append(snapshot)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"full_lap_evolved_{timestamp}.npz"
    with lzma.open(output_file, "wb") as f:
        pickle.dump(snapshots, f)

    print(f"✓ Saved full evolved lap to: {output_file}")
    print(f"  Total frames: {len(snapshots)}")
    return str(output_file)