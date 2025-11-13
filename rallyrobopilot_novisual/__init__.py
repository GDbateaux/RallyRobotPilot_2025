"""
Physics-only rally robot pilot - Uses Panda3D collision without rendering.

This package provides physics simulation with collision detection:
- CarPhysics: Exact copy of car.py physics with collision
- TrackData: Track metadata only (no models/textures)
- DirectController: Control interface
- SimulationLoop: Physics update loop (no Ursina task manager)
- CollisionSystem: Panda3D collision detection without window

Use this for genetic algorithm training and headless simulations.
"""

from .car_physics import CarPhysics
from .track_data import TrackData, load_track_metadata
from .controller import DirectController
from .simulation_loop import SimulationLoop
from .collision import CollisionSystem

__all__ = [
    'CarPhysics',
    'TrackData',
    'load_track_metadata',
    'DirectController',
    'SimulationLoop',
    'CollisionSystem',
]