"""
Headless game launcher for fast GA training.
No visible window, 100 FPS for 10x speedup.
"""
from rallyrobopilot import Car, Track, SunLight, MultiRaySensor
from ursina import *


def prepare_game_app_headless(track_name = "SimpleTrack"):
    """
    Prepare game app in headless mode for fast training.

    Uses Ursina's built-in headless mode:
    - headless=True: No window, no GPU rendering, no audio
    - Physics & collisions still work (CPU-side)
    - FPS controlled by Panda3D clock (set to 100 in simulator.py)

    Args:
        track_name (str): Track to load

    Returns:
        tuple: (app, car) - same as visual mode
    """
    from ursina import Ursina

    print("Initializing headless mode...")
    print("  Using Ursina headless mode (no window, no rendering)")
    print("  Physics & collisions: enabled (CPU-side)")
    print("  FPS will be controlled by Panda3D clock (set to 100 in simulator.py)")

    # Create Ursina in headless mode
    # This disables: window, rendering, audio
    # This keeps: physics, collisions, entity logic
    app = Ursina(headless=True)
    print("  Ursina initialized in headless mode")

    # Set assets folder
    application.asset_folder = application.asset_folder.parent

    # Load track (skip visual assets in headless mode)
    track = Track(track_name)
    # Don't load textures/models in headless - just load collision data
    track.load_assets([], [])  # Empty lists - no visual assets needed

    # Car setup
    car = Car()
    car.sports_car()
    car.set_track(track)

    # Sensors (needed for physics)
    car.multiray_sensor = MultiRaySensor(car, 15, 90)
    car.multiray_sensor.enable()

    # Skip all visual components in headless mode:
    # - No lights
    # - No shaders
    # - No sky
    # - No camera setup (not needed)

    car.visible = False  # Not rendering anyway

    car.enable()

    track.activate()
    track.played = True

    print("Headless mode initialized: no rendering, physics only")
    return app, car