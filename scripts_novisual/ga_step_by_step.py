import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.lap_data import load_recorded_lap, cleanup_recording
from genetics_algo.checkpoint import create_checkpoints_from_data, calculate_checkpoint_line
from genetics_algo.config import (
    NUM_CHECKPOINTS, CHECKPOINT_WIDTH, POPULATION_SIZE,
    NUM_GENERATIONS_INITIAL, SAVE_ALL_PLOTS, NUM_WORKERS,
    FINISH_LINE_CHECKPOINT_ID, EXTRA_GENERATIONS_AFTER_FINISH,
    TRACK_NAME
)
from genetics_algo.track_map import create_track_visualization
from genetics_algo.ga_simple import calculate_genome_length, create_initial_population
from genetics_algo.simulator import simulate_individual
from genetics_algo.fitness import calculate_fitness
from genetics_algo.individual import to_game_format
from genetics_algo.evolution import augment_population_with_results, create_next_generation, get_population_stats
from rallyrobopilot_novisual import CollisionSystem
import pickle
import lzma
from multiprocessing import Pool, cpu_count, get_context
from concurrent.futures import ThreadPoolExecutor
import os


class ControlSnapshot:
    def __init__(self, forward, backward, left, right):
        self.current_controls = (forward, backward, left, right)


# Global variables for worker processes (to avoid recreating collision system)
_collision_system = None
_checkpoints = None


def _init_worker(checkpoints, track_path="SimpleTrack/track_metadata.json"):
    global _collision_system, _checkpoints
    # Suppress output from collision system
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        _collision_system = CollisionSystem(track_path)
        _checkpoints = checkpoints
    finally:
        sys.stdout = old_stdout


def _evaluate_individual_worker(genome):
    global _collision_system, _checkpoints

    # Simulate individual
    sim_result = simulate_individual(genome, _collision_system)

    # Calculate fitness (pass genome_size for finish line speed bonus)
    genome_size = len(genome)
    fit_result = calculate_fitness(sim_result, _checkpoints, debug=False, genome_size=genome_size)

    return (sim_result, fit_result)


def evaluate_population_parallel(population, checkpoints, collision_system, pool=None, num_workers=None, track_path="SimpleTrack/track_metadata.json"):
    if num_workers is None:
        num_workers = NUM_WORKERS

    # Handle special case: -1 means use all cores
    if num_workers == -1:
        num_workers = cpu_count()

    # Extract genomes from population
    genomes = [ind['genome'] for ind in population]

    # Sequential execution if num_workers == 1
    if num_workers == 1:
        simulation_results = []
        fitness_results = []
        for genome in genomes:
            # Use collision system from main process
            sim_result = simulate_individual(genome, collision_system)
            genome_size = len(genome)
            fit_result = calculate_fitness(sim_result, checkpoints, debug=False, genome_size=genome_size)
            simulation_results.append(sim_result)
            fitness_results.append(fit_result)
        return simulation_results, fitness_results

    # Parallel execution using existing pool or creating new one
    if pool is None:
        # Create temporary pool (not recommended, prefer passing existing pool)
        ctx = get_context('spawn')
        with ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(checkpoints, track_path)) as temp_pool:
            results = temp_pool.map(_evaluate_individual_worker, genomes)
    else:
        # Use existing pool (efficient, workers already initialized)
        results = pool.map(_evaluate_individual_worker, genomes)

    # Separate simulation and fitness results
    simulation_results = [r[0] for r in results]
    fitness_results = [r[1] for r in results]

    return simulation_results, fitness_results


def main():
    # Build track metadata path from config
    TRACK_METADATA_PATH = f"{TRACK_NAME}/track_metadata.json"

    print("=" * 70)
    print("GENETIC ALGORITHM - STEP BY STEP")
    print("=" * 70)
    print()
    print(f"Track: {TRACK_NAME}")
    print()

    # Step 1: Load recorded lap
    print("Step 1: Loading recorded lap...")
    data = load_recorded_lap()
    if not data:
        print("✗ Error: Failed to load data")
        return
    data = cleanup_recording(data)
    print(f"✓ Loaded and cleaned: {len(data)} frames\n")

    # Step 2: Create checkpoints
    print("Step 2: Creating checkpoints...")
    finish_snapshot = data[0]
    finish_line = calculate_checkpoint_line(
        finish_snapshot.car_position,
        finish_snapshot.car_angle,
        CHECKPOINT_WIDTH
    )
    finish_line['frame_index'] = 0
    finish_line['checkpoint_id'] = FINISH_LINE_CHECKPOINT_ID

    intermediate_checkpoints = create_checkpoints_from_data(
        data,
        NUM_CHECKPOINTS - 1,
        CHECKPOINT_WIDTH
    )
    checkpoints = intermediate_checkpoints + [finish_line]
    print(f"✓ Created {len(checkpoints)} checkpoints (including finish line)\n")

    # Step 3: Initialize collision system
    print("Step 3: Initializing collision system...")
    collision_system = CollisionSystem(TRACK_METADATA_PATH)
    print("✓ Collision system ready\n")

    # Step 3.5: Extract track contours ONCE (will be reused for all visualizations)
    print("Step 3.5: Extracting track contours for visualizations...")
    from genetics_algo.track_map import extract_track_contours
    track_contours = extract_track_contours(collision_system, y_plane=1.0, verbose=True)
    print("✓ Track contours extracted (will be reused for all plots)\n")

    # Step 4: Calculate genome length
    print("Step 4: Calculating genome length...")
    genome_length = calculate_genome_length(data)
    print(f"✓ Genome length: {genome_length} frames (2x recorded lap of {len(data)} frames)\n")

    # Step 5: Create initial population
    print("Step 5: Creating initial population...")
    population = create_initial_population(genome_length, POPULATION_SIZE)
    print(f"✓ Created {len(population)} individuals, genome length: {genome_length}\n")

    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Step 6: Evolution loop
    print(f"Step 6: Evolution ({NUM_GENERATIONS_INITIAL} generations)...")

    # Create worker pool once (reuse across all generations)
    worker_pool = None
    if NUM_WORKERS > 1:
        print(f"Initializing {NUM_WORKERS} parallel workers...")
        ctx = get_context('spawn')
        worker_pool = ctx.Pool(processes=NUM_WORKERS, initializer=_init_worker, initargs=(checkpoints, TRACK_METADATA_PATH))
        print(f"✓ Worker pool ready")

    # Create visualization thread pool (single background thread for non-blocking PNG saves)
    viz_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viz")
    viz_futures = []

    # Track when finish line is crossed for early termination
    finish_line_crossed_gen = None

    print("=" * 70)

    try:
        for generation in range(NUM_GENERATIONS_INITIAL):
            print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS_INITIAL}")
            print("-" * 70)

            # Simulate and evaluate all individuals (parallel if NUM_WORKERS > 1)
            print(f"  Evaluating {len(population)} individuals...", end=" ", flush=True)
            simulation_results, fitness_results = evaluate_population_parallel(
                population,
                checkpoints,
                collision_system,
                pool=worker_pool,
                num_workers=NUM_WORKERS,
                track_path=TRACK_METADATA_PATH
            )
            print("✓")

            # Augment population with results
            population = augment_population_with_results(population, simulation_results, fitness_results)

            # Get statistics
            stats = get_population_stats(population)
            best = stats['best_individual']

            # Print statistics
            print(f"  Best fitness:  {stats['best_fitness']:.1f}")
            print(f"  Avg fitness:   {stats['avg_fitness']:.1f}")
            print(f"  Checkpoints:   {stats['max_checkpoints']}")
            print(f"  Collisions:    {stats['collisions']}/{len(population)}")
            print(f"  Best: {best['checkpoints_crossed_count']} checkpoints {best['checkpoints_crossed']}", end="")
            if best['collision_detected']:
                print(f" (collision at frame {best['collision_frame']})")
            else:
                print(" (no collision)")

            # Check if finish line was crossed (for early termination)
            if finish_line_crossed_gen is None and FINISH_LINE_CHECKPOINT_ID in best['checkpoints_crossed']:
                finish_line_crossed_gen = generation + 1
                print(f"\n  🏁 FINISH LINE CROSSED! Running {EXTRA_GENERATIONS_AFTER_FINISH} more generations...")

            # Submit visualization to background thread (non-blocking)
            if SAVE_ALL_PLOTS:
                viz_path = plots_dir / f"track_gen{generation + 1}.png"
                future = viz_executor.submit(
                    create_track_visualization,
                    checkpoints,
                    collision_system,
                    track_contours,
                    None, None,  # trajectory, simulation_result
                    simulation_results,
                    TRACK_METADATA_PATH,
                    str(viz_path),
                    False  # verbose
                )
                viz_futures.append((generation + 1, future))
                print(f"  Visualization queued (gen {generation + 1})")

            # Check if should terminate early (finish line + extra generations complete)
            should_terminate = False
            if finish_line_crossed_gen is not None:
                if generation + 1 >= finish_line_crossed_gen + EXTRA_GENERATIONS_AFTER_FINISH:
                    should_terminate = True
                    print(f"\n  ✓ Completed {EXTRA_GENERATIONS_AFTER_FINISH} generations after finish line")

            # Create next generation (except for last generation or early termination)
            if generation < NUM_GENERATIONS_INITIAL - 1 and not should_terminate:
                print(f"  Creating next generation...", end=" ")
                population = create_next_generation(population)
                print("✓")
            else:
                # Submit final generation visualization (always, even if SAVE_ALL_PLOTS is False)
                if not SAVE_ALL_PLOTS:
                    viz_path = plots_dir / f"track_final.png"
                    future = viz_executor.submit(
                        create_track_visualization,
                        checkpoints,
                        collision_system,
                        track_contours,
                        None, None,  # trajectory, simulation_result
                        simulation_results,
                        TRACK_METADATA_PATH,
                        str(viz_path),
                        False  # verbose
                    )
                    viz_futures.append(("final", future))
                    print(f"  Final visualization queued")

                # Break out of evolution loop if terminating early
                if should_terminate:
                    break

    finally:
        # Clean up worker pool
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()

        # Wait for all background visualizations to complete
        if viz_futures:
            print("\nWaiting for background visualizations to complete...")
            for gen_id, future in viz_futures:
                try:
                    future.result()  # Wait for completion
                    print(f"  ✓ Visualization saved (gen {gen_id})")
                except Exception as e:
                    print(f"  ✗ Visualization failed (gen {gen_id}): {e}")
            viz_executor.shutdown(wait=True)
            print("✓ All visualizations complete")

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print()

    # Step 7: Save best individual from final generation
    print("Step 7: Saving best individual to file...")
    best_individual = get_population_stats(population)['best_individual']

    # Create ga_runs_backup directory if it doesn't exist
    ga_runs_dir = Path("ga_runs")
    ga_runs_dir.mkdir(exist_ok=True)

    # Convert genome to game format (forward, backward, left, right)
    snapshots = []
    for gene in best_individual['genome']:
        axis_fb, axis_lr = gene
        forward, backward, left, right = to_game_format(axis_fb, axis_lr)
        snapshot = ControlSnapshot(forward, backward, left, right)
        snapshots.append(snapshot)

    # Determine next run number
    existing_runs = list(ga_runs_dir.glob("run_*.npz"))
    if existing_runs:
        # Extract numbers from existing files
        run_numbers = []
        for f in existing_runs:
            try:
                num = int(f.stem.split('_')[1])
                run_numbers.append(num)
            except (IndexError, ValueError):
                continue
        next_num = max(run_numbers) + 1 if run_numbers else 0
    else:
        next_num = 0

    filename = f"run_{next_num}.npz"
    output_path = ga_runs_dir / filename

    # Save using lzma compression (same format as recorded laps)
    with lzma.open(output_path, "wb") as file:
        pickle.dump(snapshots, file)

    print(f"✓ Saved to: {output_path}")
    print(f"  Fitness: {best_individual['fitness']:.1f}")
    print(f"  Checkpoints: {best_individual['checkpoints_crossed_count']} {best_individual['checkpoints_crossed']}")
    if best_individual['collision_detected']:
        print(f"  Collision at frame: {best_individual['collision_frame']}")
    else:
        print(f"  No collision!")
    print()

    print("✓ All steps complete")
    print()


if __name__ == "__main__":
    main()