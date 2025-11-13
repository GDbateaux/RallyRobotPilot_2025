"""
Fitness evaluation for genetic algorithm.
Calculates fitness based on checkpoints crossed, collision, speed, and order.
"""

import math
from genetics_algo.checkpoint import check_line_crossing
from genetics_algo.config import (
    CHECKPOINT_BONUS,
    SURVIVAL_POINTS_PER_FRAME,
    ENABLE_SURVIVAL_BONUS,
    COLLISION_PENALTY,
    OUT_OF_ORDER_PENALTY,
    OUT_OF_BOUNDS_PENALTY,
    OUT_OF_BOUNDS_DISTANCE,
    SPEED_PENALTY_FACTOR,
    FINISH_LINE_MIN_FRAME,
    NUM_CHECKPOINTS,
    FINISH_LINE_CHECKPOINT_ID,
    FINISH_LINE_SPEED_BONUS_FACTOR
)


def calculate_fitness(simulation_result, checkpoints, debug=True, genome_size=None):
    """
    Calculate fitness score for an individual based on simulation result.

    Fitness components:
    1. Checkpoint bonus: +CHECKPOINT_BONUS points per checkpoint crossed (in order)
    2. Survival bonus: +SURVIVAL_POINTS_PER_FRAME per frame survived
    3. Collision penalty: COLLISION_PENALTY if collision occurred
    4. Out-of-order penalty: OUT_OF_ORDER_PENALTY if checkpoints crossed out of order
    5. Speed penalty: -checkpoints_crossed * frames_to_last_checkpoint * SPEED_PENALTY_FACTOR
    6. Finish line speed bonus: +(genome_size - finish_frame) * FINISH_LINE_SPEED_BONUS_FACTOR

    Args:
        simulation_result (dict): Result from simulate_individual()
        checkpoints (list): List of checkpoint dictionaries, sorted by checkpoint_id
        debug (bool): If True, print detailed fitness breakdown
        genome_size (int): Size of genome (for finish line speed bonus calculation)

    Returns:
        dict: {
            'fitness': float,
            'checkpoints_crossed': list of checkpoint IDs in order crossed,
            'checkpoints_crossed_count': int,
            'crossed_in_order': bool,
            'collision': bool,
            'frames_to_last_checkpoint': int or None
        }
    """
    fitness = 0.0

    # For debug output
    fitness_components = {
        'checkpoint_bonus': 0.0,
        'survival_bonus': 0.0,
        'progress_bonus': 0.0,
        'finish_line_speed_bonus': 0.0,
        'collision_penalty': 0.0,
        'out_of_order_penalty': 0.0,
        'out_of_bounds_penalty': 0.0,
        'speed_penalty': 0.0
    }
    trajectory = simulation_result['trajectory']

    # Sort checkpoints by ID to ensure correct order
    sorted_checkpoints = sorted(checkpoints, key=lambda cp: cp['checkpoint_id'])

    # Track which checkpoints were crossed and when
    checkpoints_crossed = []  # List of (checkpoint_id, frame)
    checkpoint_frames = {}  # Map checkpoint_id -> frame where crossed

    # Check each checkpoint for crossing
    for checkpoint in sorted_checkpoints:
        checkpoint_id = checkpoint['checkpoint_id']
        is_finish_line = (checkpoint_id == FINISH_LINE_CHECKPOINT_ID)

        # Check trajectory for crossing this checkpoint
        for i in range(len(trajectory) - 1):
            pos1 = trajectory[i]['position']
            pos2 = trajectory[i + 1]['position']
            frame = i + 1

            # Special handling for finish line
            if is_finish_line:
                # Skip finish line if we haven't crossed all other checkpoints yet
                # Valid checkpoints are [1, 2, ..., FINISH_LINE_CHECKPOINT_ID - 1]
                all_others_crossed = all(
                    cp_id in checkpoint_frames
                    for cp_id in range(1, FINISH_LINE_CHECKPOINT_ID)
                )
                if not all_others_crossed:
                    continue  # Skip this frame, check next

                # Skip finish line if we're still within the minimum frame window
                if frame < FINISH_LINE_MIN_FRAME:
                    continue  # Skip this frame, check next

            if check_line_crossing(pos1, pos2, checkpoint):
                # Checkpoint crossed!
                if checkpoint_id not in checkpoint_frames:  # Only count first crossing
                    checkpoints_crossed.append((checkpoint_id, frame))
                    checkpoint_frames[checkpoint_id] = frame
                break

    # Extract just the checkpoint IDs in order crossed
    crossed_ids = [cp_id for cp_id, frame in checkpoints_crossed]

    # Only count checkpoints that form a perfect sequence [1, 2, 3, ...] with no gaps
    # Stop at the first gap or out-of-order checkpoint
    valid_checkpoints = []
    valid_checkpoints_with_frames = []
    for i, (checkpoint_id, frame) in enumerate(checkpoints_crossed):
        expected_id = i + 1  # We expect checkpoints to be 1, 2, 3, ...
        if checkpoint_id == expected_id:
            valid_checkpoints.append(checkpoint_id)
            valid_checkpoints_with_frames.append((checkpoint_id, frame))
        else:
            # Hit a gap or out-of-order checkpoint - stop counting
            break

    # Replace with only the valid sequential checkpoints
    crossed_ids = valid_checkpoints
    checkpoints_crossed = valid_checkpoints_with_frames
    crossed_in_order = True  # If we got here, all counted checkpoints are in order

    # Add checkpoint bonus for valid sequential checkpoints only
    checkpoint_points = len(crossed_ids) * CHECKPOINT_BONUS
    fitness += checkpoint_points
    fitness_components['checkpoint_bonus'] = checkpoint_points

    # Finish line speed bonus: reward faster lap completion
    if FINISH_LINE_CHECKPOINT_ID in crossed_ids and genome_size is not None:
        # Get the frame when finish line was crossed
        finish_line_frame = checkpoint_frames[FINISH_LINE_CHECKPOINT_ID]
        # Calculate bonus: more frames saved = higher bonus
        frames_saved = genome_size - finish_line_frame
        finish_bonus = frames_saved * FINISH_LINE_SPEED_BONUS_FACTOR
        fitness += finish_bonus
        fitness_components['finish_line_speed_bonus'] = finish_bonus

    # Calculate frames to last checkpoint (for speed penalty)
    frames_to_last_checkpoint = None
    if len(checkpoints_crossed) > 0:
        last_checkpoint_frame = checkpoints_crossed[-1][1]
        frames_to_last_checkpoint = last_checkpoint_frame

        # Speed penalty: fewer frames = better
        speed_penalty = len(crossed_ids) * frames_to_last_checkpoint * SPEED_PENALTY_FACTOR
        fitness -= speed_penalty
        fitness_components['speed_penalty'] = -speed_penalty

        # Survival bonus AFTER last checkpoint (rewards progress toward next checkpoint)
        # 1 point per frame survived after last checkpoint
        total_frames = simulation_result['frames_simulated']
        frames_after_last_checkpoint = total_frames - last_checkpoint_frame
        progress_bonus = frames_after_last_checkpoint * 1.0  # 1 point per frame
        fitness += progress_bonus
        fitness_components['progress_bonus'] = progress_bonus

        # Out-of-bounds penalty: check distance from last checkpoint
        # If car is too far from last checkpoint, it likely found a way to cross walls illegally
        last_checkpoint_id = crossed_ids[-1]

        # Find the last checkpoint's position
        last_checkpoint = None
        for cp in sorted_checkpoints:
            if cp['checkpoint_id'] == last_checkpoint_id:
                last_checkpoint = cp
                break

        if last_checkpoint is not None:
            # Get final car position
            final_position = trajectory[-1]['position']
            checkpoint_center = last_checkpoint['center']

            # Calculate 3D distance
            dx = final_position[0] - checkpoint_center[0]
            dy = final_position[1] - checkpoint_center[1]
            dz = final_position[2] - checkpoint_center[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Apply penalty if too far (likely crossed wall illegally)
            if distance > OUT_OF_BOUNDS_DISTANCE:
                fitness += OUT_OF_BOUNDS_PENALTY
                fitness_components['out_of_bounds_penalty'] = OUT_OF_BOUNDS_PENALTY

    # Handle collision
    if simulation_result['collision_detected']:
        fitness += COLLISION_PENALTY
        fitness_components['collision_penalty'] = COLLISION_PENALTY

        # Survival bonus up to collision frame (if enabled)
        if ENABLE_SURVIVAL_BONUS:
            collision_frame = simulation_result['collision_frame']
            survival_points = collision_frame * SURVIVAL_POINTS_PER_FRAME
            fitness += survival_points
            fitness_components['survival_bonus'] = survival_points
    else:
        # Survival bonus for all frames (if enabled)
        if ENABLE_SURVIVAL_BONUS:
            survival_points = simulation_result['frames_simulated'] * SURVIVAL_POINTS_PER_FRAME
            fitness += survival_points
            fitness_components['survival_bonus'] = survival_points

    # Print debug information if requested
    if debug:
        print("  --- Fitness Breakdown ---")
        print(f"  Checkpoint bonus:      {fitness_components['checkpoint_bonus']:+8.1f}  ({len(crossed_ids)} checkpoints × {CHECKPOINT_BONUS})")
        if fitness_components['finish_line_speed_bonus'] > 0:
            finish_frame = checkpoint_frames[FINISH_LINE_CHECKPOINT_ID]
            frames_saved = int(fitness_components['finish_line_speed_bonus'] / FINISH_LINE_SPEED_BONUS_FACTOR)
            print(f"  Finish line bonus:     {fitness_components['finish_line_speed_bonus']:+8.1f}  (completed in {finish_frame} frames, saved {frames_saved} frames × {FINISH_LINE_SPEED_BONUS_FACTOR})")
        if fitness_components['progress_bonus'] > 0:
            print(f"  Progress bonus:        {fitness_components['progress_bonus']:+8.1f}  ({int(fitness_components['progress_bonus'])} frames after last checkpoint)")
        if ENABLE_SURVIVAL_BONUS:
            print(f"  Survival bonus:        {fitness_components['survival_bonus']:+8.1f}  ({int(fitness_components['survival_bonus'])} frames × {SURVIVAL_POINTS_PER_FRAME})")
        print(f"  Collision penalty:     {fitness_components['collision_penalty']:+8.1f}")
        print(f"  Out-of-order penalty:  {fitness_components['out_of_order_penalty']:+8.1f}")
        if fitness_components['out_of_bounds_penalty'] < 0:
            print(f"  Out-of-bounds penalty: {fitness_components['out_of_bounds_penalty']:+8.1f}  (distance from last checkpoint > {OUT_OF_BOUNDS_DISTANCE})")
        print(f"  Speed penalty:         {fitness_components['speed_penalty']:+8.1f}  ({len(crossed_ids)} checkpoints × {frames_to_last_checkpoint} frames × {SPEED_PENALTY_FACTOR})")
        print(f"  {'─' * 40}")
        print(f"  TOTAL FITNESS:         {fitness:+8.1f}")
        print()

    return {
        'fitness': fitness,
        'checkpoints_crossed': crossed_ids,
        'checkpoints_crossed_count': len(crossed_ids),
        'crossed_in_order': crossed_in_order,
        'collision': simulation_result['collision_detected'],
        'frames_to_last_checkpoint': frames_to_last_checkpoint,
        'frames_simulated': simulation_result['frames_simulated']
    }