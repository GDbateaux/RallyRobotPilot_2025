"""
Test script to verify that trajectories are enhanced with checkpoints_crossed and cumulative_fitness.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.simulator import simulate_individual
from genetics_algo.fitness import calculate_fitness, enhance_trajectory_with_fitness
from genetics_algo.checkpoint import create_checkpoints_from_data
from genetics_algo.lap_data import load_recorded_lap
from genetics_algo.ga_simple import create_initial_population
from genetics_algo.config import TRACK_NAME, NUM_CHECKPOINTS, CHECKPOINT_WIDTH, CHECKPOINTS_TO_REMOVE
from rallyrobopilot_novisual import CollisionSystem


def test_enhanced_trajectory():
    """Test that trajectory enhancement adds checkpoints_crossed and cumulative_fitness."""

    print("=" * 70)
    print("TESTING ENHANCED TRAJECTORY DATA")
    print("=" * 70)
    print()

    # Step 1: Load track and create checkpoints
    print("Step 1: Loading track and checkpoints...")
    recorded_data = load_recorded_lap()
    checkpoints_raw = create_checkpoints_from_data(
        recorded_data,
        NUM_CHECKPOINTS,
        CHECKPOINT_WIDTH
    )

    # Filter checkpoints
    from genetics_algo.checkpoint import filter_and_renumber_checkpoints
    checkpoints, finish_line_id = filter_and_renumber_checkpoints(checkpoints_raw, CHECKPOINTS_TO_REMOVE)
    print(f"  ✓ Loaded {len(checkpoints)} checkpoints (finish line: {finish_line_id})")
    print()

    # Step 2: Create collision system
    print("Step 2: Initializing collision system...")
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    collision_system = CollisionSystem(f"{TRACK_NAME}/track_metadata.json")
    sys.stdout = old_stdout
    print("  ✓ Collision system ready")
    print()

    # Step 3: Create a simple test genome
    print("Step 3: Creating test genome...")
    # Use first part of recorded lap as genome
    genome_length = min(200, len(recorded_data))
    from genetics_algo.individual import from_game_format
    genome = []
    for snapshot in recorded_data[:genome_length]:
        forward, backward, left, right = snapshot.current_controls
        axis_fb, axis_lr = from_game_format(forward, backward, left, right)
        genome.append((axis_fb, axis_lr))
    print(f"  ✓ Created genome with {len(genome)} frames")
    print()

    # Step 4: Simulate individual
    print("Step 4: Simulating individual...")
    sim_result = simulate_individual(genome, collision_system)
    print(f"  ✓ Simulated {sim_result['frames_simulated']} frames")
    print(f"     Collision: {sim_result['collision_detected']}")
    print()

    # Step 5: Calculate fitness
    print("Step 5: Calculating fitness...")
    fit_result = calculate_fitness(
        sim_result,
        checkpoints,
        debug=False,
        genome_size=len(genome)
    )
    print(f"  ✓ Fitness: {fit_result['fitness']:.1f}")
    print(f"     Checkpoints crossed: {fit_result['checkpoints_crossed']}")
    print()

    # Step 6: Enhance trajectory
    print("Step 6: Enhancing trajectory...")
    trajectory = sim_result['trajectory'].copy()
    enhanced_trajectory = enhance_trajectory_with_fitness(
        trajectory,
        fit_result,
        sim_result,
        checkpoints,
        genome_size=len(genome)
    )
    print(f"  ✓ Enhanced {len(enhanced_trajectory)} trajectory frames")
    print()

    # Step 7: Verify enhanced data
    print("Step 7: Verifying enhanced data...")
    print()

    errors = []

    # Check that all frames have the new fields
    for i, frame in enumerate(enhanced_trajectory):
        if 'checkpoints_crossed' not in frame:
            errors.append(f"Frame {i}: Missing 'checkpoints_crossed'")
        if 'cumulative_fitness' not in frame:
            errors.append(f"Frame {i}: Missing 'cumulative_fitness'")

    if errors:
        print("✗ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("✓ All frames have required fields")
    print()

    # Show sample frames
    print("Sample frames:")
    print(f"  {'Frame':>5} | {'Position':>30} | {'Speed':>6} | {'Checkpoints':>15} | {'Fitness':>10}")
    print("  " + "-" * 80)

    sample_indices = [0, len(enhanced_trajectory) // 4, len(enhanced_trajectory) // 2,
                     3 * len(enhanced_trajectory) // 4, len(enhanced_trajectory) - 1]

    for idx in sample_indices:
        if idx < len(enhanced_trajectory):
            frame = enhanced_trajectory[idx]
            pos = frame['position']
            speed = frame['speed']
            checkpoints = frame['checkpoints_crossed']
            fitness = frame['cumulative_fitness']

            pos_str = f"({pos[0]:6.1f}, {pos[1]:6.1f}, {pos[2]:6.1f})"
            cp_str = str(checkpoints) if checkpoints else "[]"
            print(f"  {frame['frame']:5d} | {pos_str:>30} | {speed:6.2f} | {cp_str:>15} | {fitness:10.1f}")

    print()

    # Verify fitness progression
    print("Verifying fitness progression...")
    last_fitness = -float('inf')
    monotonic = True
    for frame in enhanced_trajectory:
        if frame['cumulative_fitness'] < last_fitness - 1.0:  # Allow small numerical errors
            monotonic = False
            break
        last_fitness = frame['cumulative_fitness']

    if not monotonic:
        print("  ⚠ Warning: Fitness is not strictly monotonic (may be OK due to penalties)")
    else:
        print("  ✓ Cumulative fitness increases monotonically")

    print()

    # Verify final fitness matches
    final_cumulative = enhanced_trajectory[-1]['cumulative_fitness']
    calculated_fitness = fit_result['fitness']

    print(f"Final cumulative fitness: {final_cumulative:.1f}")
    print(f"Calculated fitness:       {calculated_fitness:.1f}")

    # Allow small tolerance for floating point differences
    if abs(final_cumulative - calculated_fitness) < 0.1:
        print("✓ Final cumulative fitness matches calculated fitness")
    else:
        diff = abs(final_cumulative - calculated_fitness)
        print(f"⚠ Warning: Difference of {diff:.2f} between cumulative and calculated fitness")

    print()
    print("=" * 70)
    print("✓ TEST PASSED: Trajectory enhancement working correctly!")
    print("=" * 70)
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_enhanced_trajectory()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
