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

# Import SensingSnapshot for loading recorded laps (headless version)
sys.path.insert(0, str(Path(__file__).parent.parent))
from rallyrobopilot_novisual.sensing_message import SensingSnapshot

# Global cache for loaded data
_cached_data = None
_cached_checkpoints = None  # ADD THIS LINE


class _ModuleRedirector(pickle.Unpickler):
    """
    Custom unpickler to redirect rallyrobopilot imports to rallyrobopilot_novisual.

    This allows us to load lap files that were recorded with the visual version
    without triggering ursina initialization.
    """
    def find_class(self, module, name):
        # Redirect visual module to novisual version
        if module == 'rallyrobopilot.sensing_message':
            module = 'rallyrobopilot_novisual.sensing_message'
        return super().find_class(module, name)


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
            # Use custom unpickler to redirect module imports
            data = _ModuleRedirector(file).load()

        print(f"✓ Loaded {len(data)} snapshots")
        return data

    except FileNotFoundError:
        print(f"✗ Error: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


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
