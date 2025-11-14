import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallyrobopilot_novisual import CollisionSystem
from genetics_algo.config import NUM_CHECKPOINTS


def extract_track_contours(collision_system, y_plane=1.0, verbose=True):
    from panda3d.core import GeomNode, GeomVertexReader

    if verbose:
        print(f"Extracting track contours at y={y_plane}...")

    contour_segments = []

    def extract_mesh_contours(node_path, y_plane):
        segments = []

        for node in node_path.findAllMatches('**/+GeomNode'):
            geom_node = node.node()

            for i in range(geom_node.getNumGeoms()):
                geom = geom_node.getGeom(i)
                vdata = geom.getVertexData()
                vertex_reader = GeomVertexReader(vdata, 'vertex')

                vertices = []
                while not vertex_reader.isAtEnd():
                    v = vertex_reader.getData3()
                    world_pos = node.getRelativePoint(collision_system.base.render, v)
                    vertices.append((world_pos.x, world_pos.y, world_pos.z))

                for j in range(geom.getNumPrimitives()):
                    prim = geom.getPrimitive(j)

                    for k in range(0, prim.getNumVertices(), 3):
                        if k + 2 < prim.getNumVertices():
                            v0_idx = prim.getVertex(k)
                            v1_idx = prim.getVertex(k + 1)
                            v2_idx = prim.getVertex(k + 2)

                            v0 = vertices[v0_idx]
                            v1 = vertices[v1_idx]
                            v2 = vertices[v2_idx]

                            edge_intersections = []
                            edges = [(v0, v1), (v1, v2), (v2, v0)]
                            for va, vb in edges:
                                if (va[1] <= y_plane <= vb[1]) or (vb[1] <= y_plane <= va[1]):
                                    if abs(va[1] - vb[1]) > 1e-6:
                                        t = (y_plane - va[1]) / (vb[1] - va[1])
                                        x = va[0] + t * (vb[0] - va[0])
                                        z = va[2] + t * (vb[2] - va[2])
                                        edge_intersections.append((x, z))

                            if len(edge_intersections) == 2:
                                p1, p2 = edge_intersections
                                segments.append((p1[0], p1[1], p2[0], p2[1]))

        return segments

    track_segments = extract_mesh_contours(collision_system.track_node, y_plane)
    contour_segments.extend(track_segments)

    for obstacle_node in collision_system.obstacle_nodes:
        obstacle_segments = extract_mesh_contours(obstacle_node, y_plane)
        contour_segments.extend(obstacle_segments)

    if verbose:
        print(f"✓ Extracted {len(contour_segments)} contour segments")
    return contour_segments


def plot_track_with_checkpoints(track_contours, checkpoints, output_path="track_visualization.png", show_checkpoint_numbers=True, verbose=True):
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Draw track contours (walls and obstacles)
    for segment in track_contours:
        x1, z1, x2, z2 = segment
        ax.plot([x1, x2], [z1, z2], 'k-', linewidth=0.5, alpha=0.4, zorder=1)

    # Draw checkpoints in order
    sorted_checkpoints = sorted(checkpoints, key=lambda cp: cp['checkpoint_id'])

    for cp in sorted_checkpoints:
        # Extract point A and point B
        pa_x, pa_z = cp['point_a'][0], cp['point_a'][2]
        pb_x, pb_z = cp['point_b'][0], cp['point_b'][2]

        # Check if this is the finish line (last checkpoint)
        is_finish = (cp['checkpoint_id'] == NUM_CHECKPOINTS)

        if is_finish:
            # Draw finish line in red with thicker line
            ax.plot([pa_x, pb_x], [pa_z, pb_z], 'r-', linewidth=3, label='Finish Line', zorder=5)
        else:
            # Draw checkpoint in blue
            label = 'Checkpoints' if cp['checkpoint_id'] == 1 else None
            ax.plot([pa_x, pb_x], [pa_z, pb_z], 'b-', linewidth=2, label=label, zorder=4)

        # Add checkpoint number label at center
        if show_checkpoint_numbers:
            cx, cz = cp['center'][0], cp['center'][2]
            ax.text(cx, cz, str(cp['checkpoint_id']), fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='yellow', alpha=0.7), zorder=6)

    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Z Position', fontsize=12)
    ax.set_title('Track with Checkpoints', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if verbose:
        print(f"✓ Track visualization saved to: {output_path}")

    return fig, ax


def create_track_visualization(checkpoints, collision_system=None, track_contours=None, trajectory=None, simulation_result=None, simulation_results=None, track_path="SimpleTrack/track_metadata.json", output_path="track_visualization.png", verbose=False):
    import io
    import sys

    # Create collision system if not provided
    if collision_system is None:
        # Suppress CollisionSystem output by redirecting stdout temporarily
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            collision_system = CollisionSystem(track_path)

            # Restore stdout
            if not verbose:
                sys.stdout = old_stdout
        except Exception as e:
            # Restore stdout on error
            if not verbose:
                sys.stdout = old_stdout
            raise e

    # Extract track contours at y=1 (only if not provided)
    if track_contours is None:
        track_contours = extract_track_contours(collision_system, y_plane=1.0, verbose=verbose)

    # Plot track with all checkpoints
    fig, ax = plot_track_with_checkpoints(track_contours, checkpoints, output_path, verbose=verbose)

    # Handle multiple simulation results
    if simulation_results and len(simulation_results) > 0:
        import matplotlib.cm as cm
        import numpy as np

        # Generate colors for each individual
        colors = cm.rainbow(np.linspace(0, 1, len(simulation_results)))

        for idx, result in enumerate(simulation_results):
            trajectory = result['trajectory']
            if len(trajectory) == 0:
                continue

            # Extract trajectory positions
            traj_x = [t['position'][0] for t in trajectory]
            traj_z = [t['position'][2] for t in trajectory]

            color = colors[idx]

            # Plot trajectory (no label to avoid legend clutter)
            ax.plot(traj_x, traj_z, '-', linewidth=1.5, color=color, alpha=0.7, zorder=7)

            # Mark collision point if detected (same color as trajectory)
            if result['collision_detected']:
                collision_point = result['collision_point']
                ax.scatter([collision_point[0]], [collision_point[2]], c=color, s=10, marker='x',
                          linewidths=0.5, alpha=0.9, zorder=9)

        # Mark common start position (all start at same place)
        if len(simulation_results) > 0:
            first_traj = simulation_results[0]['trajectory']
            if len(first_traj) > 0:
                start_x = first_traj[0]['position'][0]
                start_z = first_traj[0]['position'][2]
                ax.scatter([start_x], [start_z], c='lime', s=10, marker='o',
                          edgecolors='darkgreen', linewidths=0.5, zorder=10)

    # Handle single trajectory (backward compatibility)
    elif trajectory and len(trajectory) > 0:
        # Extract trajectory positions
        traj_x = [t['position'][0] for t in trajectory]
        traj_z = [t['position'][2] for t in trajectory]

        # Plot trajectory
        ax.plot(traj_x, traj_z, 'g-', linewidth=2, label='Car Path', alpha=0.8, zorder=7)

        # Mark start position
        ax.scatter([traj_x[0]], [traj_z[0]], c='lime', s=10, marker='o',
                   label='Start', edgecolors='darkgreen', linewidths=0.5, zorder=8)

        # Mark collision point if detected
        if simulation_result and simulation_result['collision_detected']:
            collision_point = simulation_result['collision_point']
            ax.scatter([collision_point[0]], [collision_point[2]], c='red', s=40, marker='X',
                      label=f"Collision (frame {simulation_result['collision_frame']})",
                      edgecolors='darkred', linewidths=0.8, zorder=9)
        else:
            # Mark end position if no collision
            ax.scatter([traj_x[-1]], [traj_z[-1]], c='cyan', s=150, marker='s',
                      label='End', edgecolors='darkblue', linewidths=2, zorder=8)

        # Update legend
        ax.legend(fontsize=10)

    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Print only the final result
    if not verbose:
        print(f"✓ Track visualization saved: {output_path}")

    return fig, ax, collision_system, track_contours


if __name__ == "__main__":
    # Test visualization
    print("=== Testing Track Visualization ===\n")

    from genetics_algo.lap_data import get_checkpoints

    # Get checkpoints
    checkpoints = get_checkpoints()
    print(f"Loaded {len(checkpoints)} checkpoints\n")

    # Create visualization
    create_track_visualization(checkpoints)

    print("\n✓ Test complete!")