"""
Individual representation for genetic algorithm.
Simple helper functions for control genome manipulation.

Genome format: List of (axis_fb, axis_lr) tuples where:
  axis_fb: -1 (backward), 0 (coast), 1 (forward)
  axis_lr: -1 (left), 0 (straight), 1 (right)
"""

import random
from genetics_algo.config import (
    INIT_PROB_FORWARD, INIT_PROB_COAST, INIT_PROB_BACKWARD,
    INIT_PROB_LEFT, INIT_PROB_STRAIGHT, INIT_PROB_RIGHT
)


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


def generate_random_genome(length):
    """
    Generate a random genome of specified length.

    Args:
        length (int): Number of control tuples in the genome

    Returns:
        list: List of (axis_fb, axis_lr) tuples
    """
    return [generate_random_control() for _ in range(length)]