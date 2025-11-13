"""
Debug script to visualize the invisible collision walls.
This shows you EXACTLY where Obstacles.obj collision meshes are.
"""
from rallyrobopilot import Car, Track, MultiRaySensor
from ursina import *

def main():
    app = Ursina(size=(1280, 1024))

    # Set assets folder
    application.asset_folder = application.asset_folder.parent

    # Load track
    track = Track("SimpleTrack")

    # MAKE OBSTACLES VISIBLE with a color!
    print(f"\n=== Found {len(track.obstacles)} collision obstacles ===")
    for i, obs in enumerate(track.obstacles):
        obs.visible = True  # Make visible!
        obs.color = color.red  # Color them red
        obs.alpha = 0.3  # Semi-transparent
        print(f"{i+1}. Position: {obs.position}, Scale: {obs.scale}")

    # Also show the track
    track.activate(activate_details=True)
    track.color = color.white

    # Add car for reference
    car = Car()
    car.sports_car()
    car.set_track(track)
    car.visible = True
    car.enable()

    # Setup camera
    EditorCamera()

    print("\n=== Controls ===")
    print("Mouse: Look around")
    print("WASD: Move camera")
    print("Q/E: Up/Down")
    print("Scroll: Zoom")
    print("\nRED SEMI-TRANSPARENT OBJECTS = Your invisible collision walls!")
    print("These are what your car is hitting!\n")

    app.run()

if __name__ == "__main__":
    main()