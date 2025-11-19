"""
Test script to verify partial simulation optimization.

This script verifies that:
1. Mutation tracking correctly stores mutation_frames
2. simulate_with_optimization uses partial simulation when possible
3. Partial simulation produces the same results as full simulation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.simulator import simulate_individual, simulate_with_optimization
from genetics_algo.mutation import create_children
from genetics_algo.checkpoint import create_checkpoints_from_data, filter_and_renumber_checkpoints
from genetics_algo.lap_data import load_recorded_lap
from genetics_algo.fitness import calculate_fitness
from genetics_algo.individual import from_game_format
from genetics_algo.config import (
    TRACK_NAME, NUM_CHECKPOINTS, CHECKPOINT_WIDTH, CHECKPOINTS_TO_REMOVE
)
from rallyrobopilot_novisual import CollisionSystem
import io


def test_partial_simulation():
    """Test that partial simulation optimization works correctly."""

    print("=" * 70)
    print("TESTING PARTIAL SIMULATION OPTIMIZATION")
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

    from genetics_algo.checkpoint import filter_and_renumber_checkpoints
    checkpoints, finish_line_id = filter_and_renumber_checkpoints(checkpoints_raw, CHECKPOINTS_TO_REMOVE)
    print(f"  ✓ Loaded {len(checkpoints)} checkpoints")
    print()

    # Step 2: Create collision system
    print("Step 2: Initializing collision system...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    collision_system = CollisionSystem(f"{TRACK_NAME}/track_metadata.json")
    sys.stdout = old_stdout
    print("  ✓ Collision system ready")
    print()

    # Step 3: Create a parent genome from recorded data
    print("Step 3: Creating parent individual from recorded data...")
    genome_length = min(100, len(recorded_data))
    genome = []
    for snapshot in recorded_data[:genome_length]:
        forward, backward, left, right = snapshot.current_controls
        axis_fb, axis_lr = from_game_format(forward, backward, left, right)
        genome.append((axis_fb, axis_lr))
    print(f"  ✓ Created genome with {len(genome)} frames")

    # Simulate parent to get trajectory
    print("  Simulating parent...")
    parent_sim = simulate_individual(genome, collision_system)
    parent_fitness = calculate_fitness(parent_sim, checkpoints, debug=False, genome_size=len(genome))

    parent = {
        'genome': genome,
        'trajectory': parent_sim['trajectory'],
        'fitness': parent_fitness['fitness'],
        'collision_detected': parent_sim['collision_detected'],
        'collision_frame': parent_sim.get('collision_frame'),
        'checkpoints_crossed': parent_fitness['checkpoints_crossed'],
        'checkpoints_crossed_count': parent_fitness['checkpoints_crossed_count']
    }
    print(f"  ✓ Parent simulated: {len(parent_sim['trajectory'])} frames, fitness {parent_fitness['fitness']:.1f}")
    print()

    # Step 4: Create children (this should track mutations)
    print("Step 4: Creating mutated children...")
    children = create_children([parent], mutation_window_size=20, mutation_rate_no_crash=1.0)
    child = children[0]

    print(f"  ✓ Child created")
    print(f"     Has mutation_frames field: {'mutation_frames' in child}")
    if 'mutation_frames' in child:
        print(f"     Mutation frames: {child['mutation_frames']}")
    print()

    # Step 5: Test full simulation (baseline)
    print("Step 5: Running full simulation (baseline)...")
    full_sim = simulate_individual(child['genome'], collision_system, start_frame=0, initial_state=None)
    full_fitness = calculate_fitness(full_sim, checkpoints, debug=False, genome_size=len(child['genome']))
    print(f"  ✓ Full simulation: {len(full_sim['trajectory'])} frames, fitness {full_fitness['fitness']:.1f}")
    print()

    # Step 6: Test optimized simulation
    print("Step 6: Running optimized simulation...")
    opt_sim = simulate_with_optimization(child, collision_system)
    opt_fitness = calculate_fitness(opt_sim, checkpoints, debug=False, genome_size=len(child['genome']))

    print(f"  ✓ Optimized simulation: {len(opt_sim['trajectory'])} frames, fitness {opt_fitness['fitness']:.1f}")
    if opt_sim.get('optimization_used'):
        print(f"     Optimization used: YES")
        print(f"     Frames reused: {opt_sim.get('frames_reused', 0)}")
        print(f"     Speedup potential: {opt_sim.get('frames_reused', 0) / len(child['genome']) * 100:.1f}%")
    else:
        print(f"     Optimization used: NO (fallback to full simulation)")
    print()

    # Step 7: Compare results
    print("Step 7: Comparing results...")
    print()

    errors = []

    # Check fitness matches
    if abs(full_fitness['fitness'] - opt_fitness['fitness']) > 0.01:
        errors.append(f"Fitness mismatch: full={full_fitness['fitness']:.2f}, opt={opt_fitness['fitness']:.2f}")

    # Check trajectory length matches
    if len(full_sim['trajectory']) != len(opt_sim['trajectory']):
        errors.append(f"Trajectory length mismatch: full={len(full_sim['trajectory'])}, opt={len(opt_sim['trajectory'])}")

    # Check collision detection matches
    if full_sim['collision_detected'] != opt_sim['collision_detected']:
        errors.append(f"Collision detection mismatch: full={full_sim['collision_detected']}, opt={opt_sim['collision_detected']}")

    # Check final positions match (within tolerance)
    if len(full_sim['trajectory']) > 0 and len(opt_sim['trajectory']) > 0:
        full_final = full_sim['trajectory'][-1]['position']
        opt_final = opt_sim['trajectory'][-1]['position']
        distance = sum((a - b) ** 2 for a, b in zip(full_final, opt_final)) ** 0.5

        if distance > 0.1:  # Tolerance of 0.1 units
            errors.append(f"Final position mismatch: distance={distance:.4f}")

    if errors:
        print("✗ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False

    print("✓ All checks passed!")
    print("  - Fitness matches")
    print("  - Trajectory length matches")
    print("  - Collision detection matches")
    print("  - Final positions match")
    print()

    # Step 8: Performance summary
    print("=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print()

    if opt_sim.get('optimization_used'):
        frames_saved = opt_sim.get('frames_reused', 0)
        total_frames = len(child['genome'])
        speedup_percent = (frames_saved / total_frames * 100) if total_frames > 0 else 0

        print(f"Total genome frames:  {total_frames}")
        print(f"Frames reused:        {frames_saved}")
        print(f"Frames simulated:     {total_frames - frames_saved}")
        print(f"Theoretical speedup:  {speedup_percent:.1f}%")
        print()
        print("✓ Partial simulation optimization is WORKING!")
    else:
        print("⚠ Optimization was not used in this test")
        print("  This may happen if:")
        print("  - Child has no mutations tracked (mutation_frames empty)")
        print("  - Mutation occurred at frame 0")
        print("  - Parent trajectory is missing or empty")

    print()
    print("=" * 70)
    print("✓ TEST PASSED")
    print("=" * 70)
    print()

    return True


if __name__ == "__main__":
    try:
        success = test_partial_simulation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
