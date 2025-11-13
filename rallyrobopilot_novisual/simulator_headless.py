"""
Headless simulator using pure physics - NO visual components.

This module provides GA simulation without any Ursina/rendering dependencies.
"""

from .car_physics import CarPhysics
from .track_data import TrackData
from .controller import DirectController
from .simulation_loop import SimulationLoop
from .collision import CollisionSystem


def setup_headless_simulation(track_path="SimpleTrack/track_metadata.json", enable_collisions=True):
    """
    Setup pure physics simulation - no Ursina, no window, no rendering.

    Args:
        track_path (str): Path to track metadata JSON
        enable_collisions (bool): Enable collision detection (default: True)

    Returns:
        tuple: (car, controller, sim_loop, collision_system)
            - car: CarPhysics instance
            - controller: DirectController instance
            - sim_loop: SimulationLoop instance
            - collision_system: CollisionSystem instance (or None if disabled)
    """
    print(f"=== Setting up HEADLESS PHYSICS simulation ===")
    print(f"Track: {track_path}")
    print(f"Physics timestep: 0.1s (10 FPS physics)")
    print(f"NO window, NO rendering, NO Ursina")
    print(f"Collision detection: {'ENABLED' if enable_collisions else 'DISABLED'}")
    print()

    # Load track metadata
    track = TrackData(track_path)
    start_pos = track.get_start_position()
    start_rot = track.get_start_orientation()

    print(f"Track loaded: {track.track_name}")
    print(f"Start position: {start_pos}")
    print(f"Start rotation: {start_rot}")
    print()

    # Create collision system if enabled
    collision_system = None
    if enable_collisions:
        collision_system = CollisionSystem(track_path)

    # Create physics car with collision system
    car = CarPhysics(position=start_pos, rotation_y=start_rot[1], collision_system=collision_system)

    # Create controller
    controller = DirectController(car)

    # Create simulation loop
    sim_loop = SimulationLoop(controller)

    print("✓ Headless physics simulation ready")
    print()

    return car, controller, sim_loop, collision_system