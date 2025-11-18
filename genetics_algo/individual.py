"""
Individual representation for genetic algorithm.
Simple helper functions for control genome manipulation.

Genome format: List of (axis_fb, axis_lr) tuples where:
  axis_fb: -1 (backward), 0 (coast), 1 (forward)
  axis_lr: -1 (left), 0 (straight), 1 (right)
"""

import random
import numpy as np
import math
from pathlib import Path
from genetics_algo.config import (
    INIT_PROB_FORWARD, INIT_PROB_COAST, INIT_PROB_BACKWARD,
    INIT_PROB_LEFT, INIT_PROB_STRAIGHT, INIT_PROB_RIGHT,
    START_AT
)

# Cache for loaded base genome (to avoid reloading from disk for each individual)
_cached_base_genome = None
_cache_loaded = False
_debug_printed = False  # Track if we've already printed debug info


def to_game_format(axis_fb, axis_lr):
    """
    Convert compact control format to game format.

    Args:
        axis_fb (int): Forward/backward axis (-1, 0, 1)
        axis_lr (int): Left/right axis (-1, 0, 1)

    Returns:
        tuple: (forward, backward, left, right) booleans
    """
    forward = (axis_fb == 1)
    backward = (axis_fb == -1)
    left = (axis_lr == -1)
    right = (axis_lr == 1)
    return (forward, backward, left, right)


def from_game_format(forward, backward, left, right):
    """
    Convert game format to compact control format.
    Used by evaluate_run.py to load saved runs.

    Args:
        forward (bool): Forward pressed
        backward (bool): Backward pressed
        left (bool): Left pressed
        right (bool): Right pressed

    Returns:
        tuple: (axis_fb, axis_lr) with values in {-1, 0, 1}
    """
    if forward:
        axis_fb = 1
    elif backward:
        axis_fb = -1
    else:
        axis_fb = 0

    if right:
        axis_lr = 1
    elif left:
        axis_lr = -1
    else:
        axis_lr = 0

    return (axis_fb, axis_lr)


def generate_random_control():
    """
    Generate one random control tuple with weighted probabilities.

    Throttle probabilities from config (default: 70% forward, 20% coast, 10% backward)
    Steering probabilities from config (default: equal for left/straight/right)

    Returns:
        tuple: (axis_fb, axis_lr) with values in {-1, 0, 1}
    """
    # Sample throttle with weighted probabilities
    axis_fb = random.choices(
        population=[1, 0, -1],
        weights=[INIT_PROB_FORWARD, INIT_PROB_COAST, INIT_PROB_BACKWARD],
        k=1
    )[0]

    # Sample steering with weighted probabilities
    axis_lr = random.choices(
        population=[1, 0, -1],
        weights=[INIT_PROB_RIGHT, INIT_PROB_STRAIGHT, INIT_PROB_LEFT],
        k=1
    )[0]

    return (axis_fb, axis_lr)


def load_latest_ga_run():
    """
    Load the latest .npz file from ga_runs/ directory.
    These files are lzma-compressed pickle files containing ControlSnapshot objects.
    Uses caching to avoid reloading the same file multiple times.

    Returns:
        list: List of (axis_fb, axis_lr) tuples extracted from snapshots, or None if no files found
    """
    global _cached_base_genome, _cache_loaded

    # Return cached genome if already loaded
    if _cache_loaded:
        return _cached_base_genome

    import lzma
    import pickle

    ga_runs_dir = Path(__file__).parent.parent / "ga_runs"

    if not ga_runs_dir.exists():
        print(f"Warning: ga_runs directory not found at {ga_runs_dir}")
        _cache_loaded = True
        _cached_base_genome = None
        return None

    # Find all .npz files (they're actually lzma-compressed pickle files)
    npz_files = list(ga_runs_dir.glob("*.npz"))

    if not npz_files:
        print(f"Warning: No .npz files found in {ga_runs_dir}")
        _cache_loaded = True
        _cached_base_genome = None
        return None

    # Get the latest file by modification time
    latest_file = max(npz_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading base genome from: {latest_file.name}")

    try:
        # Load lzma-compressed pickle file
        with lzma.open(latest_file, "rb") as file:
            snapshots = pickle.load(file)

        # Convert ControlSnapshot objects back to (axis_fb, axis_lr) tuples
        genome = []
        for snapshot in snapshots:
            # Extract control inputs from snapshot
            # ControlSnapshot has: current_controls = (forward, backward, left, right)
            forward, backward, left, right = snapshot.current_controls
            axis_fb, axis_lr = from_game_format(forward, backward, left, right)
            genome.append((axis_fb, axis_lr))

        print(f"  Loaded {len(genome)} frames from {latest_file.name}")

        # Cache the result
        _cached_base_genome = genome
        _cache_loaded = True

        return genome

    except Exception as e:
        print(f"Error loading {latest_file.name}: {e}")
        _cache_loaded = True
        _cached_base_genome = None
        return None


def generate_random_genome(length):
    """
    Generate a random genome of specified length.
    If START_AT is set in config, uses latest ga_runs/*.npz genome as base
    for frames 0 to START_AT, then generates random controls for remaining frames.

    Args:
        length (int or float): Number of control tuples in the genome (will be converted to int)

    Returns:
        list: List of (axis_fb, axis_lr) tuples
    """
    # Convert length to int using ceiling (in case it's a float from multiplier calculations)
    length = math.ceil(length)

    # Check if we should bootstrap from existing solution
    if START_AT is not None and START_AT > 0:
        global _debug_printed

        # Load the latest GA run (returns genome as list of tuples)
        base_genome = load_latest_ga_run()

        if base_genome is not None:
            # Debug output (only print once for first individual)
            if not _debug_printed:
                print(f"  Base genome has {len(base_genome)} frames")
                print(f"  START_AT config = {START_AT}")
                print(f"  Target genome length = {length}")

            # Validate START_AT is within bounds
            start_at = min(START_AT, len(base_genome), length)

            # Debug output BEFORE setting flag
            print_debug = not _debug_printed
            if print_debug:
                print(f"  Will copy {start_at} frames from base genome")
                print(f"  Bootstrapping: Frames 0-{start_at-1} from base, frames {start_at}-{length-1} random")
                print(f"  First 5 controls from base genome: {base_genome[:5]}")

            # Create genome: base[0:START_AT] + random[START_AT:]
            genome = []

            # Copy base genome until START_AT (simply copy, no modification)
            for i in range(start_at):
                if i < len(base_genome):
                    genome.append(base_genome[i])  # Direct copy - already a tuple
                else:
                    genome.append(generate_random_control())

            # Generate random for remaining frames
            for i in range(start_at, length):
                genome.append(generate_random_control())

            # Verify the copy worked (only for first individual)
            if print_debug:
                print(f"  Verification: First 5 controls in new genome: {genome[:5]}")
                print(f"  Generating {100} individuals with this configuration...")
                _debug_printed = True

            return genome

    # Default: fully random genome
    return [generate_random_control() for _ in range(length)]