from rallyrobopilot_novisual import CarPhysics, TrackData, DirectController, CollisionSystem
from genetics_algo.individual import to_game_format


def simulate_individual(genome, collision_system, track_path="SimpleTrack/track_metadata.json"):
    # Initialize physics simulation (reuse existing collision_system)
    track = TrackData(track_path)
    start_pos = track.get_start_position()
    start_rot = track.get_start_orientation()

    car = CarPhysics(position=start_pos, rotation_y=start_rot[1], collision_system=collision_system)
    controller = DirectController(car)

    # Trajectory recording
    trajectory = []
    collision_detected = False
    collision_frame = None
    collision_point = None

    # Simulate frame by frame
    for frame_idx, gene in enumerate(genome):
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
    final_frame = len(trajectory)
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
        'completed': not collision_detected and len(trajectory) >= len(genome)
    }