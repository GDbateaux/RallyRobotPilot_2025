import random
import copy
from genetics_algo.config import (
    MUTATION_WINDOW_SIZE, MUTATION_RATE_NO_CRASH,
    MUTATION_DURATION_MIN, MUTATION_DURATION_MAX, BRAKE_MUTATION_DURATION_MAX,
    NUM_MUTATIONS_PER_EVENT,
    MUTATION_PROB_FORWARD, MUTATION_PROB_COAST, MUTATION_PROB_BACKWARD,
    MUTATION_PROB_LEFT, MUTATION_PROB_STRAIGHT, MUTATION_PROB_RIGHT
)


def mutate_genome(genome, collision_frame, window_size=None, duration_min=None, duration_max=None, num_mutations=None):

    if window_size is None:
        window_size = MUTATION_WINDOW_SIZE
    if duration_min is None:
        duration_min = MUTATION_DURATION_MIN
    if duration_max is None:
        duration_max = MUTATION_DURATION_MAX
    if num_mutations is None:
        num_mutations = NUM_MUTATIONS_PER_EVENT

    # Create a copy of the genome
    mutated_genome = copy.deepcopy(genome)

    # Determine mutation window: [max(0, collision_frame - window), collision_frame]
    window_start = max(0, collision_frame - window_size)
    window_end = collision_frame

    # If window is empty (crash at frame 0), mutate frame 0
    if window_start >= window_end:
        window_start = 0
        window_end = min(1, len(genome))

    # Perform multiple mutations (1 to num_mutations)
    actual_num_mutations = random.randint(1, num_mutations)

    for _ in range(actual_num_mutations):
        # Pick random frame in the window with 1/x distribution
        # Heavily favors frames closer to collision (more likely to contain the mistake)
        if window_start < window_end:
            # Create list of candidate frames
            candidate_frames = list(range(window_start, window_end))

            # Calculate 1/x weights (higher weight = closer to collision)
            # Frame right before collision gets weight=1.0, frames farther back get weight=1/distance
            weights = [1.0 / (collision_frame - frame) for frame in candidate_frames]

            # Select frame using weighted random choice
            mutate_frame = random.choices(candidate_frames, weights=weights)[0]
        else:
            mutate_frame = 0

        # Ensure frame is within genome bounds
        if mutate_frame >= len(mutated_genome):
            mutate_frame = len(mutated_genome) - 1

        # Randomly decide to mutate throttle and/or steering (50% each, independent)
        mutate_throttle = random.random() < 0.5
        mutate_steering = random.random() < 0.5

        # If neither selected, pick one randomly
        if not mutate_throttle and not mutate_steering:
            if random.random() < 0.5:
                mutate_throttle = True
            else:
                mutate_steering = True

        # Calculate new values ONCE (outside loop) for sustained behaviors
        new_throttle = None
        new_steering = None

        if mutate_throttle:
            new_throttle = _mutate_value_throttle()

        if mutate_steering:
            new_steering = _mutate_value_steering()

        # Pick duration based on mutation type
        # Brake mutations (backward throttle) use shorter duration
        if mutate_throttle and new_throttle == -1:
            # Brake mutation - use shorter max duration
            duration = random.randint(duration_min, BRAKE_MUTATION_DURATION_MAX)
        else:
            # Normal mutation - use full duration range
            duration = random.randint(duration_min, duration_max)

        # Apply SAME values to all consecutive frames (creates sustained turns/acceleration)
        for i in range(duration):
            frame_idx = mutate_frame + i
            if frame_idx >= len(mutated_genome):
                break  # Don't go past end of genome

            current_fb, current_lr = mutated_genome[frame_idx]

            # Apply pre-calculated throttle value to all frames
            if mutate_throttle:
                current_fb = new_throttle

            # Apply pre-calculated steering value to all frames
            if mutate_steering:
                current_lr = new_steering

            mutated_genome[frame_idx] = (current_fb, current_lr)

    return mutated_genome


def _mutate_value_throttle():
    rand = random.random()
    if rand < MUTATION_PROB_FORWARD:
        return 1   # Forward
    elif rand < MUTATION_PROB_FORWARD + MUTATION_PROB_COAST:
        return 0   # Coast
    else:
        return -1  # Backward


def _mutate_value_steering():
    rand = random.random()
    if rand < MUTATION_PROB_LEFT:
        return -1  # Left
    elif rand < MUTATION_PROB_LEFT + MUTATION_PROB_STRAIGHT:
        return 0   # Straight
    else:
        return 1   # Right


def mutate_genome_random(genome, duration_min=None, duration_max=None, num_mutations=None):
    if duration_min is None:
        duration_min = MUTATION_DURATION_MIN
    if duration_max is None:
        duration_max = MUTATION_DURATION_MAX
    if num_mutations is None:
        num_mutations = NUM_MUTATIONS_PER_EVENT

    # Create a copy of the genome
    mutated_genome = copy.deepcopy(genome)

    if len(mutated_genome) == 0:
        return mutated_genome

    # Perform multiple mutations (1 to num_mutations)
    actual_num_mutations = random.randint(1, num_mutations)

    for _ in range(actual_num_mutations):
        # Pick random frame anywhere in genome
        mutate_frame = random.randint(0, len(mutated_genome) - 1)

        # Randomly decide to mutate throttle and/or steering (50% each, independent)
        mutate_throttle = random.random() < 0.5
        mutate_steering = random.random() < 0.5

        # If neither selected, pick one randomly
        if not mutate_throttle and not mutate_steering:
            if random.random() < 0.5:
                mutate_throttle = True
            else:
                mutate_steering = True

        # Calculate new values ONCE (outside loop) for sustained behaviors
        new_throttle = None
        new_steering = None

        if mutate_throttle:
            new_throttle = _mutate_value_throttle()

        if mutate_steering:
            new_steering = _mutate_value_steering()

        # Pick duration based on mutation type
        # Brake mutations (backward throttle) use shorter duration
        if mutate_throttle and new_throttle == -1:
            # Brake mutation - use shorter max duration
            duration = random.randint(duration_min, BRAKE_MUTATION_DURATION_MAX)
        else:
            # Normal mutation - use full duration range
            duration = random.randint(duration_min, duration_max)

        # Apply SAME values to all consecutive frames (creates sustained turns/acceleration)
        for i in range(duration):
            frame_idx = mutate_frame + i
            if frame_idx >= len(mutated_genome):
                break  # Don't go past end of genome

            current_fb, current_lr = mutated_genome[frame_idx]

            # Apply pre-calculated throttle value to all frames
            if mutate_throttle:
                current_fb = new_throttle

            # Apply pre-calculated steering value to all frames
            if mutate_steering:
                current_lr = new_steering

            mutated_genome[frame_idx] = (current_fb, current_lr)

    return mutated_genome


def create_children(parents, mutation_window_size=None, mutation_rate_no_crash=None, mutation_duration_min=None, mutation_duration_max=None):
    if mutation_window_size is None:
        mutation_window_size = MUTATION_WINDOW_SIZE
    if mutation_rate_no_crash is None:
        mutation_rate_no_crash = MUTATION_RATE_NO_CRASH
    if mutation_duration_min is None:
        mutation_duration_min = MUTATION_DURATION_MIN
    if mutation_duration_max is None:
        mutation_duration_max = MUTATION_DURATION_MAX

    children = []

    for parent in parents:
        if parent.get('collision_detected', False):
            # Crashed: mutate near crash point
            child_genome = mutate_genome(
                parent['genome'],
                parent['collision_frame'],
                window_size=mutation_window_size,
                duration_min=mutation_duration_min,
                duration_max=mutation_duration_max
            )
        else:
            # Didn't crash: clone OR randomly mutate
            if random.random() < mutation_rate_no_crash:
                # Mutate randomly
                child_genome = mutate_genome_random(
                    parent['genome'],
                    duration_min=mutation_duration_min,
                    duration_max=mutation_duration_max
                )
            else:
                # Clone unchanged
                child_genome = copy.deepcopy(parent['genome'])

        children.append({
            'genome': child_genome,
            'fitness': None,
            'collision_detected': None,
            'collision_frame': None
        })

    return children