from rallyrobopilot_novisual import CarPhysics, TrackData, DirectController, CollisionSystem
from genetics_algo.individual import to_game_format


def extract_car_state_from_trajectory(trajectory, frame_idx):
    """
    Extract car physics state from a trajectory frame.
    Returns a state dict suitable for initializing partial simulation.

    The game engine only needs position, speed, and angle to recreate the state.
    """
    if frame_idx >= len(trajectory):
        return None

    frame = trajectory[frame_idx]

    # Extract essential state (position, speed, angle)
    state = {
        'position': frame['position'],
        'rotation_y': frame['angle'],
        'speed': frame['speed']
    }

    return state


def simulate_with_optimization(individual, collision_system, track_path="SimpleTrack/track_metadata.json"):
    """
    Intelligently simulate an individual using partial simulation when possible.

    If the individual has mutation_frames (indicating mutations from a parent),
    we can reuse the parent's trajectory up to the earliest mutation and only
    simulate from that point forward.

    Args:
        individual: Dict with 'genome', 'mutation_frames', and 'trajectory' (from parent)
        collision_system: The collision detection system
        track_path: Path to track metadata

    Returns:
        Simulation result dict (same format as simulate_individual)
    """
    genome = individual['genome']
    mutation_frames = individual.get('mutation_frames', [])
    parent_trajectory = individual.get('trajectory')

    # Check if we can use optimization
    can_optimize = (
        mutation_frames and  # Has mutations tracked
        len(mutation_frames) > 0 and  # At least one mutation
        parent_trajectory and  # Has parent trajectory
        len(parent_trajectory) > 0  # Trajectory is not empty
    )

    if not can_optimize:
        # Full simulation from beginning
        return simulate_individual(genome, collision_system, track_path, start_frame=0, initial_state=None)

    # Find earliest mutation frame
    start_frame = min(mutation_frames)

    # Safety check: ensure start_frame is valid
    if start_frame <= 0 or start_frame >= len(parent_trajectory):
        # Can't optimize - fall back to full simulation
        return simulate_individual(genome, collision_system, track_path, start_frame=0, initial_state=None)

    # Extract initial state from parent trajectory at mutation frame
    # Trajectory records state BEFORE controls are applied, so trajectory[N] is the state at the start of frame N
    initial_state = extract_car_state_from_trajectory(parent_trajectory, start_frame)

    if initial_state is None:
        # Failed to extract state - fall back to full simulation
        return simulate_individual(genome, collision_system, track_path, start_frame=0, initial_state=None)

    # Perform partial simulation from the mutation point
    sim_result = simulate_individual(
        genome,
        collision_system,
        track_path,
        start_frame=start_frame,
        initial_state=initial_state
    )

    # Combine parent trajectory (up to mutation) with new trajectory
    # Parent trajectory: frames 0 to start_frame-1
    # New trajectory: frames start_frame onwards
    combined_trajectory = parent_trajectory[:start_frame] + sim_result['trajectory']

    # Update result with combined trajectory
    sim_result['trajectory'] = combined_trajectory
    sim_result['optimization_used'] = True
    sim_result['frames_reused'] = start_frame

    return sim_result


def simulate_individual(genome, collision_system, track_path="SimpleTrack/track_metadata.json", start_frame=0, initial_state=None):
    # Initialize physics simulation
    track = TrackData(track_path)

    if initial_state is None:
        # Full simulation from start
        start_pos = track.get_start_position()
        start_rot = track.get_start_orientation()
        car = CarPhysics(position=start_pos, rotation_y=start_rot[1], collision_system=collision_system)
    else:
        # Partial simulation - restore from previous state (position, speed, angle)
        car = CarPhysics(
            position=initial_state['position'],
            rotation_y=initial_state['rotation_y'],
            collision_system=collision_system
        )
        # Set speed - the engine will derive velocity from speed and angle
        car.speed = initial_state['speed']

    controller = DirectController(car)

    # Trajectory recording
    trajectory = []
    collision_detected = False
    collision_frame = None
    collision_point = None

    # Simulate from start_frame to end
    for frame_idx in range(start_frame, len(genome)):
        gene = genome[frame_idx]
        # Record state BEFORE applying control
        trajectory.append({
            'frame': frame_idx,
            'position': (car.x, car.y, car.z),
            'speed': car.speed,
            'angle': car.rotation_y
        })

        # Convert genome control to game format
        axis_fb, axis_lr = gene
        forward, backward, left, right = to_game_format(axis_fb, axis_lr)

        # Apply control
        controller.apply_controls(forward, backward, left, right)

        # Reset collision flag before update
        car.collision_occurred = False

        # Update physics (one timestep)
        car.update(forward, backward, left, right)

        # Check for collision
        if car.collision_occurred:
            collision_detected = True
            collision_frame = frame_idx
            collision_point = car.collision_point if car.collision_point else (car.x, car.y, car.z)
            # Stop simulation immediately
            break

    # Record final state
    final_frame = start_frame + len(trajectory)
    trajectory.append({
        'frame': final_frame,
        'position': (car.x, car.y, car.z),
        'speed': car.speed,
        'angle': car.rotation_y
    })

    return {
        'trajectory': trajectory,
        'collision_detected': collision_detected,
        'collision_frame': collision_frame,
        'collision_point': collision_point,
        'frames_simulated': len(trajectory),
        'start_frame': start_frame,  # Track where simulation started
        'completed': not collision_detected and final_frame >= len(genome)
    }