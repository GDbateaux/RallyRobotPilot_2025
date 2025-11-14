import sys
from pathlib import Path
import pickle
import lzma
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallyrobopilot_novisual import CarPhysics, TrackData, DirectController, SimulationLoop, CollisionSystem


def extract_track_contours(collision_system, y_plane=1.0):
    from panda3d.core import GeomNode, GeomVertexReader

    print(f"Extracting track contours at y={y_plane}...")

    contour_segments = []

    # Process track mesh
    def extract_mesh_contours(node_path, y_plane):
        segments = []

        # Traverse all children to find GeomNodes
        for node in node_path.findAllMatches('**/+GeomNode'):
            geom_node = node.node()

            # Process each Geom in the GeomNode
            for i in range(geom_node.getNumGeoms()):
                geom = geom_node.getGeom(i)
                vdata = geom.getVertexData()

                # Get vertex reader for positions
                vertex_reader = GeomVertexReader(vdata, 'vertex')

                # Read all vertices
                vertices = []
                while not vertex_reader.isAtEnd():
                    v = vertex_reader.getData3()
                    # Transform to world coordinates
                    world_pos = node.getRelativePoint(collision_system.base.render, v)
                    vertices.append((world_pos.x, world_pos.y, world_pos.z))

                # Process primitives (triangles)
                for j in range(geom.getNumPrimitives()):
                    prim = geom.getPrimitive(j)

                    # Iterate through triangles
                    for k in range(0, prim.getNumVertices(), 3):
                        if k + 2 < prim.getNumVertices():
                            # Get triangle vertices
                            v0_idx = prim.getVertex(k)
                            v1_idx = prim.getVertex(k + 1)
                            v2_idx = prim.getVertex(k + 2)

                            v0 = vertices[v0_idx]
                            v1 = vertices[v1_idx]
                            v2 = vertices[v2_idx]

                            # Check if triangle intersects y_plane
                            # Find edges that cross the plane
                            edge_intersections = []

                            # Check each edge of the triangle
                            edges = [(v0, v1), (v1, v2), (v2, v0)]
                            for va, vb in edges:
                                # Check if edge crosses y_plane
                                if (va[1] <= y_plane <= vb[1]) or (vb[1] <= y_plane <= va[1]):
                                    if abs(va[1] - vb[1]) > 1e-6:  # Avoid division by zero
                                        # Linear interpolation to find intersection point
                                        t = (y_plane - va[1]) / (vb[1] - va[1])
                                        x = va[0] + t * (vb[0] - va[0])
                                        z = va[2] + t * (vb[2] - va[2])
                                        edge_intersections.append((x, z))

                            # If we have exactly 2 intersections, we have a contour segment
                            if len(edge_intersections) == 2:
                                p1, p2 = edge_intersections
                                segments.append((p1[0], p1[1], p2[0], p2[1]))

        return segments

    # Extract contours from track
    print("  Processing track mesh...")
    track_segments = extract_mesh_contours(collision_system.track_node, y_plane)
    contour_segments.extend(track_segments)
    print(f"    Found {len(track_segments)} track contour segments")

    # Extract contours from obstacles
    for i, obstacle_node in enumerate(collision_system.obstacle_nodes):
        print(f"  Processing obstacle {i+1}/{len(collision_system.obstacle_nodes)}...")
        obstacle_segments = extract_mesh_contours(obstacle_node, y_plane)
        contour_segments.extend(obstacle_segments)
        print(f"    Found {len(obstacle_segments)} obstacle contour segments")

    print(f"✓ Total contour segments extracted: {len(contour_segments)}")
    return contour_segments


def load_recording(filepath):
    print(f"Loading recording from: {filepath}")
    with lzma.open(filepath, "rb") as file:
        data = pickle.load(file)

    print(f"✓ Loaded {len(data)} frames")
    return data


def simulate_with_novisual(recording_data, track_path="SimpleTrack/track_metadata.json", override_from_frame=None):
    print("\nSimulating with novisual physics engine...")

    # Setup collision system
    print("Initializing collision system...")
    collision_system = CollisionSystem(track_path)

    # Setup track and car
    track = TrackData(track_path)
    start_pos = track.get_start_position()
    start_rot = track.get_start_orientation()

    # Create car at starting position WITH collision system
    car = CarPhysics(position=start_pos, rotation_y=start_rot[1], collision_system=collision_system)
    controller = DirectController(car)
    sim_loop = SimulationLoop(controller)

    simulated_positions = []
    original_positions = []

    # Track full simulation state for frame comparison
    simulated_states = []  # Will store (pos, speed, angle) for each frame

    print(f"Simulating {len(recording_data)} frames...")

    # Record initial position (frame 0 BEFORE any controls applied)
    initial_pos = recording_data[0].car_position
    simulated_positions.append(initial_pos)  # Start from same initial position
    original_positions.append(initial_pos)
    simulated_states.append({
        'position': initial_pos,
        'speed': recording_data[0].car_speed,
        'angle': recording_data[0].car_angle
    })

    # Also initialize car to match recording's initial state
    car.x = initial_pos[0]
    car.y = initial_pos[1]
    car.z = initial_pos[2]
    car.position = list(initial_pos)
    car.rotation_y = recording_data[0].car_angle
    car.speed = recording_data[0].car_speed

    # Control override notification
    if override_from_frame is not None:
        print(f"\n⚠️  CONTROL OVERRIDE ENABLED: From frame {override_from_frame} onwards, controls will be FORWARD ONLY")
        print(f"   Simulation will run for 10 frames after override, then stop")

    # Debug: Track angle and speed differences
    import math
    max_angle_diff = 0
    max_speed_diff = 0

    # Collision tracking
    total_collisions = 0
    collision_frames = []
    collision_positions = []  # Store (x, z) positions where collisions occurred
    first_collision_frame = None  # Track the first collision frame number

    # Now simulate frame by frame
    # Frame i controls -> produces frame i+1 position
    for frame_idx in range(len(recording_data) - 1):
        # Check if we should stop simulation (10 frames after override)
        if override_from_frame is not None and frame_idx >= override_from_frame + 10:
            print(f"\n⏹️  SIMULATION STOPPED at Frame {frame_idx} (10 frames after override)")
            print(f"   Skipping remaining {len(recording_data) - 1 - frame_idx} frames")
            break
        snapshot = recording_data[frame_idx]

        # Extract controls from frame i
        forward, backward, left, right = snapshot.current_controls

        # OVERRIDE CONTROLS: If override_from_frame is set and we've reached that frame
        if override_from_frame is not None and frame_idx >= override_from_frame:
            forward = True
            backward = False
            left = False
            right = False
            # Print notification only once at the override frame
            if frame_idx == override_from_frame:
                print(f"\n🎮 OVERRIDE ACTIVATED at Frame {frame_idx}: Controls now FORWARD ONLY")

        # Apply controls
        controller.apply_controls(forward, backward, left, right)

        # Reset collision flag before update
        car.collision_occurred = False

        # Step physics (no debug output)
        car.update(forward, backward, left, right, debug_frame=None)

        # Check if collision occurred this frame
        if car.collision_occurred:
            total_collisions += 1
            collision_frames.append(frame_idx + 1)  # +1 because frame_idx starts at 0

            # Track first collision frame
            if first_collision_frame is None:
                first_collision_frame = frame_idx + 1

            # Store collision position (use collision_point if available, otherwise car position)
            if car.collision_point:
                collision_positions.append((car.collision_point[0], car.collision_point[2]))
            else:
                collision_positions.append((car.x, car.z))

        # Record simulated position after applying frame i controls
        sim_pos = (car.x, car.y, car.z)
        simulated_positions.append(sim_pos)

        # Record full state for frame comparison
        simulated_states.append({
            'position': sim_pos,
            'speed': car.speed,
            'angle': car.rotation_y
        })

        # This should match frame i+1 position
        next_snapshot = recording_data[frame_idx + 1]
        orig_pos = next_snapshot.car_position
        original_positions.append(orig_pos)

        # Compare angle and speed
        angle_diff = abs(car.rotation_y - next_snapshot.car_angle)
        speed_diff = abs(car.speed - next_snapshot.car_speed)
        max_angle_diff = max(max_angle_diff, angle_diff)
        max_speed_diff = max(max_speed_diff, speed_diff)

    print("\n✓ Simulation complete")
    return simulated_positions, original_positions, collision_positions, collision_system


def calculate_errors(simulated_positions, original_positions):
    import math

    errors = []
    for sim_pos, orig_pos in zip(simulated_positions, original_positions):
        # Calculate 2D distance error (ignore Y)
        dx = sim_pos[0] - orig_pos[0]
        dz = sim_pos[2] - orig_pos[2]
        error = math.sqrt(dx*dx + dz*dz)
        errors.append(error)

    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)

    return {
        'errors': errors,
        'avg_error': avg_error,
        'max_error': max_error,
        'min_error': min_error
    }


def plot_comparison(simulated_positions, original_positions, collision_positions, track_contours=None, output_path="physics_comparison.png"):
    print(f"\nGenerating comparison plot...")

    # Extract X and Z coordinates
    sim_x = [pos[0] for pos in simulated_positions]
    sim_z = [pos[2] for pos in simulated_positions]

    orig_x = [pos[0] for pos in original_positions]
    orig_z = [pos[2] for pos in original_positions]

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw track contours first (so they're in the background)
    if track_contours:
        print(f"  Drawing {len(track_contours)} track contour segments...")
        for segment in track_contours:
            x1, z1, x2, z2 = segment
            ax.plot([x1, x2], [z1, z2], 'k-', linewidth=0.5, alpha=0.4, zorder=1)

    ax.plot(orig_x, orig_z, 'b-', linewidth=2, label='Original Recording', alpha=0.7, zorder=3)
    ax.plot(sim_x, sim_z, 'r--', linewidth=2, label='NoVisual Simulation', alpha=0.7, zorder=3)
    ax.scatter([orig_x[0]], [orig_z[0]], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter([orig_x[-1]], [orig_z[-1]], c='red', s=100, marker='X', label='End', zorder=5)

    # Mark collision points with orange crosses
    if collision_positions:
        collision_x = [pos[0] for pos in collision_positions]
        collision_z = [pos[1] for pos in collision_positions]
        ax.scatter(collision_x, collision_z, c='orange', s=80, marker='x',
                   linewidths=1.5, label=f'Collisions ({len(collision_positions)})', zorder=10)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Z Position', fontsize=12)
    ax.set_title('Car Path Comparison: Original vs NoVisual Physics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    print(f"  Open the file to view the comparison")

    # Don't show plot in headless environment
    # plt.show()


def main():
    """Run physics validation test."""
    print("=" * 70)
    print("PHYSICS VALIDATION TEST")
    print("Comparing NoVisual Physics vs Original Recording")
    print("=" * 70)
    print()

    # Configuration
    recording_path = "runs/simpletrack_recorded_1.npz"
    track_path = "SimpleTrack/track_metadata.json"

    # OVERRIDE CONTROLS: Set to frame number to override controls from that frame onwards
    # Set to None to disable override (use recorded controls for entire simulation)
    # When enabled, car will ONLY go forward (backward=False, left=False, right=False)
    OVERRIDE_FROM_FRAME = None  # Example: 103 to override from frame 103 onwards

    # Load recording
    recording_data = load_recording(recording_path)

    # Simulate with novisual engine
    simulated_positions, original_positions, collision_positions, collision_system = simulate_with_novisual(
        recording_data, track_path, override_from_frame=OVERRIDE_FROM_FRAME
    )

    # Calculate errors
    print("\nCalculating position errors...")
    error_stats = calculate_errors(simulated_positions, original_positions)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print(f"Average position error: {error_stats['avg_error']:.6f} units")
    print(f"Maximum position error: {error_stats['max_error']:.6f} units")
    print(f"Minimum position error: {error_stats['min_error']:.6f} units")
    print()

    if error_stats['avg_error'] < 0.01:
        print("✅ EXCELLENT: Physics are nearly identical! (error < 0.01)")
    elif error_stats['avg_error'] < 0.1:
        print("✅ GOOD: Physics are very similar (error < 0.1)")
    elif error_stats['avg_error'] < 1.0:
        print("⚠️  WARNING: Small differences detected (error < 1.0)")
    else:
        print("❌ ERROR: Significant physics differences detected!")

    print("=" * 70)

    # Extract track contours
    print()
    track_contours = extract_track_contours(collision_system, y_plane=1.0)

    # Plot comparison
    plot_comparison(simulated_positions, original_positions, collision_positions, track_contours)

    print("\n✓ Physics validation complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except FileNotFoundError as e:
        print(f"\n\nError: Recording file not found")
        print(f"Make sure 'runs/simpletrack_recorded_1.npz' exists")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()