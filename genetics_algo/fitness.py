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
    FINISH_LINE_SPEED_BONUS_FACTOR,
    PROGRESS_BONUS_PER_DISTANCE
)


def calculate_fitness(simulation_result, checkpoints, debug=True, genome_size=None):
    fitness = 0.0

    # For debug output
    fitness_components = {
        'checkpoint_bonus': 0.0,
        'survival_bonus': 0.0,
        'progress_bonus': 0.0,
        'progress_distance': 0.0,
        'finish_line_speed_bonus': 0.0,
        'collision_penalty': 0.0,
        'out_of_order_penalty': 0.0,
        'out_of_bounds_penalty': 0.0,
        'speed_penalty': 0.0
    }
    trajectory = simulation_result['trajectory']

    # Sort checkpoints by ID to ensure correct order
    sorted_checkpoints = sorted(checkpoints, key=lambda cp: cp['checkpoint_id'])

    # Determine finish line checkpoint ID dynamically (the checkpoint with highest ID)
    # This allows the finish line to change after checkpoint filtering
    finish_line_checkpoint_id = max(cp['checkpoint_id'] for cp in checkpoints) if checkpoints else FINISH_LINE_CHECKPOINT_ID

    # Get all checkpoint IDs for iteration (after filtering, this may not be a continuous range)
    all_checkpoint_ids = sorted([cp['checkpoint_id'] for cp in checkpoints])

    # Track ALL checkpoint crossings (not just first)
    # This handles self-crossing tracks where checkpoints may be crossed multiple times
    checkpoint_crossings = {}  # Map checkpoint_id -> [frame1, frame2, ...] of ALL crossings

    # Check each checkpoint for crossing
    for checkpoint in sorted_checkpoints:
        checkpoint_id = checkpoint['checkpoint_id']
        is_finish_line = (checkpoint_id == finish_line_checkpoint_id)

        # Initialize list for this checkpoint
        if checkpoint_id not in checkpoint_crossings:
            checkpoint_crossings[checkpoint_id] = []

        # Check trajectory for crossing this checkpoint
        for i in range(len(trajectory) - 1):
            pos1 = trajectory[i]['position']
            pos2 = trajectory[i + 1]['position']
            frame = i + 1

            # Special handling for finish line
            if is_finish_line:
                # Skip finish line if we're still within the minimum frame window
                if frame < FINISH_LINE_MIN_FRAME:
                    continue  # Skip this frame, check next

            if check_line_crossing(pos1, pos2, checkpoint):
                # Checkpoint crossed! Store ALL crossings (not just first)
                checkpoint_crossings[checkpoint_id].append(frame)
                # Don't break - keep checking for more crossings of same checkpoint

    # Build valid checkpoint sequence for self-crossing tracks
    # For each checkpoint [1, 2, 3, ...], find the crossing that occurred AFTER the previous checkpoint
    # This handles tracks where checkpoint 7 might be crossed early (invalid) then later (valid)

    valid_checkpoints = []
    valid_checkpoints_with_frames = []
    checkpoint_frames = {}  # Map checkpoint_id -> frame (for compatibility)
    last_valid_frame = 0  # Frame of the last valid checkpoint

    # Iterate through actual checkpoint IDs (after filtering, this may not be continuous)
    for expected_id in all_checkpoint_ids:
        # Check if this checkpoint was crossed
        if expected_id not in checkpoint_crossings or len(checkpoint_crossings[expected_id]) == 0:
            # Checkpoint never crossed - sequence ends here
            break

        # Find the first crossing that happened AFTER the last valid checkpoint
        valid_crossing_frame = None
        for crossing_frame in checkpoint_crossings[expected_id]:
            if crossing_frame > last_valid_frame:
                valid_crossing_frame = crossing_frame
                break  # Use first valid crossing

        if valid_crossing_frame is None:
            # No valid crossing found (all crossings were before previous checkpoint)
            break

        # Special handling for finish line - must have all previous checkpoints
        if expected_id == finish_line_checkpoint_id:
            # Check if all other checkpoints were crossed (all IDs except finish line)
            all_others_crossed = all(
                cp_id in checkpoint_frames
                for cp_id in all_checkpoint_ids if cp_id != finish_line_checkpoint_id
            )
            if not all_others_crossed:
                break  # Can't count finish line yet

        # Valid checkpoint crossing!
        valid_checkpoints.append(expected_id)
        valid_checkpoints_with_frames.append((expected_id, valid_crossing_frame))
        checkpoint_frames[expected_id] = valid_crossing_frame
        last_valid_frame = valid_crossing_frame

    # Set final values
    crossed_ids = valid_checkpoints
    checkpoints_crossed = valid_checkpoints_with_frames
    crossed_in_order = True  # If we got here, all counted checkpoints are in order

    # Add checkpoint bonus for valid sequential checkpoints only
    checkpoint_points = len(crossed_ids) * CHECKPOINT_BONUS
    fitness += checkpoint_points
    fitness_components['checkpoint_bonus'] = checkpoint_points

    # Finish line speed bonus: reward faster lap completion
    if finish_line_checkpoint_id in crossed_ids and genome_size is not None:
        # Get the frame when finish line was crossed
        finish_line_frame = checkpoint_frames[finish_line_checkpoint_id]
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
        # FIXED: Don't multiply by checkpoints - that created perverse incentive
        speed_penalty = frames_to_last_checkpoint * SPEED_PENALTY_FACTOR
        fitness -= speed_penalty
        fitness_components['speed_penalty'] = -speed_penalty

        # Progress bonus: reward distance traveled AFTER last checkpoint
        # This encourages actual forward progress, prevents gaming by slowing down/circling
        checkpoint_position = trajectory[last_checkpoint_frame]['position']
        final_position = trajectory[-1]['position']

        # Calculate 3D Euclidean distance traveled after checkpoint
        dx = final_position[0] - checkpoint_position[0]
        dy = final_position[1] - checkpoint_position[1]
        dz = final_position[2] - checkpoint_position[2]
        distance_traveled = math.sqrt(dx*dx + dy*dy + dz*dz)

        progress_bonus = distance_traveled * PROGRESS_BONUS_PER_DISTANCE
        fitness += progress_bonus
        fitness_components['progress_bonus'] = progress_bonus
        fitness_components['progress_distance'] = distance_traveled  # Store for debug output

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
            finish_frame = checkpoint_frames[finish_line_checkpoint_id]
            frames_saved = int(fitness_components['finish_line_speed_bonus'] / FINISH_LINE_SPEED_BONUS_FACTOR)
            print(f"  Finish line bonus:     {fitness_components['finish_line_speed_bonus']:+8.1f}  (completed in {finish_frame} frames, saved {frames_saved} frames × {FINISH_LINE_SPEED_BONUS_FACTOR})")
        if fitness_components['progress_bonus'] > 0:
            distance = fitness_components['progress_distance']
            print(f"  Progress bonus:        {fitness_components['progress_bonus']:+8.1f}  ({distance:.1f} units traveled after last checkpoint × {PROGRESS_BONUS_PER_DISTANCE})")
        if ENABLE_SURVIVAL_BONUS:
            print(f"  Survival bonus:        {fitness_components['survival_bonus']:+8.1f}  ({int(fitness_components['survival_bonus'])} frames × {SURVIVAL_POINTS_PER_FRAME})")
        print(f"  Collision penalty:     {fitness_components['collision_penalty']:+8.1f}")
        print(f"  Out-of-order penalty:  {fitness_components['out_of_order_penalty']:+8.1f}")
        if fitness_components['out_of_bounds_penalty'] < 0:
            print(f"  Out-of-bounds penalty: {fitness_components['out_of_bounds_penalty']:+8.1f}  (distance from last checkpoint > {OUT_OF_BOUNDS_DISTANCE})")
        print(f"  Speed penalty:         {fitness_components['speed_penalty']:+8.1f}  ({frames_to_last_checkpoint} frames × {SPEED_PENALTY_FACTOR})")
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
        'frames_simulated': simulation_result['frames_simulated'],
        # NEW: Data for trajectory enhancement
        'checkpoint_frames': checkpoint_frames,  # Map: checkpoint_id -> frame
        'fitness_components': fitness_components
    }


def enhance_trajectory_with_fitness(trajectory, fitness_result, simulation_result, checkpoints, genome_size=None):
    """
    Enhance trajectory by adding checkpoints_crossed and cumulative_fitness to each frame.

    This allows later optimizations to resume simulation from any frame with complete state.

    Args:
        trajectory: List of trajectory frames from simulation
        fitness_result: Result from calculate_fitness()
        simulation_result: Result from simulate_individual()
        checkpoints: Checkpoint data
        genome_size: Size of genome (for finish line bonus calculation)

    Returns:
        Enhanced trajectory with checkpoints_crossed and cumulative_fitness per frame
    """
    checkpoint_frames = fitness_result['checkpoint_frames']  # Map: checkpoint_id -> frame
    fitness_components = fitness_result['fitness_components']

    # Build reverse lookup: frame -> list of checkpoints crossed at that frame
    frame_to_checkpoints = {}
    for checkpoint_id, frame_num in checkpoint_frames.items():
        if frame_num not in frame_to_checkpoints:
            frame_to_checkpoints[frame_num] = []
        frame_to_checkpoints[frame_num].append(checkpoint_id)

    # Sort checkpoints by ID for each frame
    for frame_num in frame_to_checkpoints:
        frame_to_checkpoints[frame_num].sort()

    # Enhance each trajectory frame
    checkpoints_crossed_so_far = []
    cumulative_fitness = 0.0

    for i, frame_data in enumerate(trajectory):
        frame_num = frame_data['frame']

        # Check if any checkpoints were crossed at this frame
        if frame_num in frame_to_checkpoints:
            for checkpoint_id in frame_to_checkpoints[frame_num]:
                checkpoints_crossed_so_far.append(checkpoint_id)
                # Add checkpoint bonus
                cumulative_fitness += CHECKPOINT_BONUS

        # Calculate cumulative fitness at this frame
        # Components that accumulate per frame:
        if ENABLE_SURVIVAL_BONUS:
            cumulative_fitness += SURVIVAL_POINTS_PER_FRAME

        # Speed penalty accumulates with each frame to last checkpoint
        if len(checkpoints_crossed_so_far) > 0:
            cumulative_fitness -= SPEED_PENALTY_FACTOR

        # Add enhanced data to frame
        frame_data['checkpoints_crossed'] = checkpoints_crossed_so_far.copy()
        frame_data['cumulative_fitness'] = cumulative_fitness

    # Add collision penalty to final frame if collision occurred
    if simulation_result['collision_detected']:
        trajectory[-1]['cumulative_fitness'] += COLLISION_PENALTY

    # Add finish line speed bonus if lap was completed
    finish_line_checkpoint_id = max(cp['checkpoint_id'] for cp in checkpoints) if checkpoints else FINISH_LINE_CHECKPOINT_ID
    if finish_line_checkpoint_id in checkpoint_frames and genome_size is not None:
        finish_frame_num = checkpoint_frames[finish_line_checkpoint_id]
        # Find the trajectory frame and add the bonus
        for frame_data in trajectory:
            if frame_data['frame'] >= finish_frame_num:
                frames_saved = genome_size - finish_frame_num
                finish_bonus = frames_saved * FINISH_LINE_SPEED_BONUS_FACTOR
                frame_data['cumulative_fitness'] += finish_bonus

    # Add progress bonus to frames after last checkpoint
    if len(checkpoint_frames) > 0:
        last_checkpoint_frame_num = max(checkpoint_frames.values())

        for i, frame_data in enumerate(trajectory):
            if frame_data['frame'] > last_checkpoint_frame_num:
                # Calculate progress bonus from last checkpoint to this frame
                checkpoint_position = trajectory[last_checkpoint_frame_num]['position']
                current_position = frame_data['position']

                dx = current_position[0] - checkpoint_position[0]
                dy = current_position[1] - checkpoint_position[1]
                dz = current_position[2] - checkpoint_position[2]
                distance_traveled = math.sqrt(dx*dx + dy*dy + dz*dz)

                progress_bonus = distance_traveled * PROGRESS_BONUS_PER_DISTANCE
                frame_data['cumulative_fitness'] += progress_bonus

    # Add out-of-bounds penalty to final frame if applicable
    if fitness_components['out_of_bounds_penalty'] < 0:
        trajectory[-1]['cumulative_fitness'] += fitness_components['out_of_bounds_penalty']

    return trajectory