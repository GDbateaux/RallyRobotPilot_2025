"""
Track metadata loader - NO visual components.
Only loads JSON data for checkpoints and starting positions.
"""
import json
from pathlib import Path


def load_track_metadata(track_name):
    """
    Load track metadata from JSON file.

    Args:
        track_name (str): Track name or path to metadata JSON

    Returns:
        dict: Track metadata
    """
    root_dir = Path(__file__).resolve().parent.parent

    if ".json" not in track_name:
        asset_path = root_dir / f"assets/{track_name}/track_metadata.json"
    else:
        asset_path = root_dir / f"assets/{track_name}"

    with open(asset_path, "r") as f:
        metadata = json.load(f)

    return metadata


class TrackData:
    """
    Track data container - only metadata, no visual elements.
    """

    def __init__(self, track_name):
        self.track_name = track_name
        self.metadata = load_track_metadata(track_name)

        # Extract relevant data
        self.car_default_reset_position = tuple(self.metadata["car_default_reset_position"])
        self.car_default_reset_orientation = tuple(self.metadata["car_default_reset_orientation"])

        # Store checkpoint data if available
        self.checkpoints = self.metadata.get("checkpoints", [])

    def get_start_position(self):
        """Get default car starting position."""
        return self.car_default_reset_position

    def get_start_orientation(self):
        """Get default car starting orientation (rotation_y is [1])."""
        return self.car_default_reset_orientation

    def __repr__(self):
        return f"TrackData(name={self.track_name}, start_pos={self.car_default_reset_position})"