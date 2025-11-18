"""
MPI-Distributed Genetic Algorithm for Rally Robot Pilot

This script parallelizes the genetic algorithm across multiple nodes using MPI.

Architecture:
- Master (rank 0): Coordinates evolution, creates generations, manages statistics
- Workers (rank 1-N): Evaluate individuals in parallel using local multiprocessing

Configuration:
- Uses same genetics_algo/config.py as single-node version
- Target: 2 nodes × 32 CPUs = 64 parallel workers
- Each MPI rank uses multiprocessing for additional parallelism

Usage:
    mpirun python3 scripts_mpi/ga_mpi.py
"""

import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MPI imports
from mpi4py import MPI
from genetics_algo.mpi_utils import (
    init_mpi, is_master, broadcast_setup_data,
    scatter_population, gather_results, print_mpi_info
)

# Existing GA modules (reuse)
from genetics_algo.lap_data import load_recorded_lap, cleanup_recording
from genetics_algo.checkpoint import (
    create_checkpoints_from_data, calculate_checkpoint_line,
    filter_and_renumber_checkpoints
)
from genetics_algo.config import (
    NUM_CHECKPOINTS, CHECKPOINT_WIDTH, POPULATION_SIZE,
    NUM_GENERATIONS_INITIAL, SAVE_ALL_PLOTS,
    FINISH_LINE_CHECKPOINT_ID, EXTRA_GENERATIONS_AFTER_FINISH,
    TRACK_NAME, CHECKPOINTS_TO_REMOVE
)
from genetics_algo.track_map import create_track_visualization, extract_track_contours
from genetics_algo.ga_simple import calculate_genome_length, create_initial_population
from genetics_algo.simulator import simulate_individual
from genetics_algo.fitness import calculate_fitness
from genetics_algo.individual import to_game_format
from genetics_algo.evolution import (
    augment_population_with_results, create_next_generation,
    get_population_stats
)
from rallyrobopilot_novisual import CollisionSystem

import pickle
import lzma
from multiprocessing import Pool, get_context
from concurrent.futures import ThreadPoolExecutor


class ControlSnapshot:
    """Control snapshot for saving final solution."""
    def __init__(self, forward, backward, left, right):
        self.current_controls = (forward, backward, left, right)


# ============================================================================
# WORKER PROCESS FUNCTIONS (for multiprocessing within each MPI rank)
# ============================================================================

# Global variables for worker processes (to avoid recreating collision system)
_collision_system = None
_checkpoints = None


def _init_worker(checkpoints, track_path):
    """
    Initialize worker process for multiprocessing pool.

    Each MPI rank creates a multiprocessing pool, and each worker
    in that pool needs its own collision system.

    Args:
        checkpoints: Checkpoint data
        track_path: Path to track metadata
    """
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
    """
    Evaluate a single individual (called by multiprocessing workers).

    Args:
        genome: Individual's genome

    Returns:
        (simulation_result, fitness_result)
    """
    global _collision_system, _checkpoints

    # Simulate individual
    sim_result = simulate_individual(genome, _collision_system)

    # Calculate fitness
    genome_size = len(genome)
    fit_result = calculate_fitness(sim_result, _checkpoints, debug=False, genome_size=genome_size)

    return (sim_result, fit_result)


# ============================================================================
# MPI-DISTRIBUTED EVALUATION
# ============================================================================

def evaluate_population_mpi(comm, rank, size, population, checkpoints, track_path, cpus_per_rank, collision_system, verbose=True):
    """
    Distributed population evaluation using MPI + multiprocessing.

    Master scatters population across all MPI ranks.
    Each rank evaluates its chunk using local multiprocessing pool.
    Results gathered back to master.

    Args:
        comm: MPI communicator
        rank: MPI rank
        size: Number of MPI processes
        population: List of individuals (valid on master only)
        checkpoints: Checkpoint data (valid on all ranks)
        track_path: Path to track metadata
        cpus_per_rank: Number of CPUs per MPI rank (for local multiprocessing)
        collision_system: Existing collision system instance (for sequential path)
        verbose: Print debug info

    Returns:
        (simulation_results, fitness_results) on master, (None, None) on workers
    """
    # Master: extract genomes from population
    if is_master(rank):
        genomes = [ind['genome'] for ind in population]
    else:
        genomes = None

    # Scatter genomes to all ranks (with debug output)
    local_genomes = scatter_population(comm, rank, size, genomes, verbose=verbose)

    # Debug: Each rank prints what it received
    if verbose:
        # Use barrier to ensure ordered printing
        for i in range(size):
            if rank == i:
                if cpus_per_rank > 1:
                    print(f"  [Rank {rank}] Received {len(local_genomes)} individuals, using {cpus_per_rank} CPU multiprocessing pool")
                else:
                    print(f"  [Rank {rank}] Received {len(local_genomes)} individuals, using sequential evaluation")
            comm.Barrier()  # Wait for this rank to finish printing

    # Each rank evaluates its chunk using multiprocessing
    if cpus_per_rank > 1:
        # Use multiprocessing pool
        if verbose and rank == 0:
            print(f"\n  [DEBUG] Starting parallel evaluation...")

        ctx = get_context('spawn')
        with ctx.Pool(processes=cpus_per_rank, initializer=_init_worker,
                      initargs=(checkpoints, track_path)) as pool:
            local_results = pool.map(_evaluate_individual_worker, local_genomes)

        if verbose:
            for i in range(size):
                if rank == i:
                    print(f"  [Rank {rank}] Completed evaluation of {len(local_genomes)} individuals")
                comm.Barrier()
    else:
        # Sequential evaluation (fallback - use existing collision_system)
        if verbose and rank == 0:
            print(f"\n  [DEBUG] Starting sequential evaluation...")

        local_results = []
        for idx, genome in enumerate(local_genomes):
            sim_result = simulate_individual(genome, collision_system)
            genome_size = len(genome)
            fit_result = calculate_fitness(sim_result, checkpoints, debug=False, genome_size=genome_size)
            local_results.append((sim_result, fit_result))

            if verbose and rank == 0 and (idx + 1) % 10 == 0:
                print(f"  [Rank {rank}] Evaluated {idx + 1}/{len(local_genomes)} individuals...")

    # Separate simulation and fitness results
    local_sim_results = [r[0] for r in local_results]
    local_fit_results = [r[1] for r in local_results]

    if verbose and rank == 0:
        print(f"\n  [DEBUG] Gathering results from all ranks...")

    # Gather results back to master
    all_sim_results, all_fit_results = gather_results(comm, rank, local_sim_results, local_fit_results)

    if verbose and rank == 0:
        print(f"  [DEBUG] All results gathered to master\n")

    return all_sim_results, all_fit_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Initialize MPI
    comm, rank, size = init_mpi()

    # Detect CPUs per MPI rank (from SLURM or default to 1)
    cpus_per_rank = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

    # Master: print MPI info
    if is_master(rank):
        print_mpi_info(comm, rank, size)
        print(f"CPUs per MPI rank: {cpus_per_rank}")
        print(f"Total parallel workers: {size * cpus_per_rank}")
        print()

    # Build track metadata path from config
    TRACK_METADATA_PATH = f"{TRACK_NAME}/track_metadata.json"

    # ========================================================================
    # SETUP PHASE (Master only)
    # ========================================================================
    if is_master(rank):
        print("=" * 70)
        print("GENETIC ALGORITHM - MPI DISTRIBUTED")
        print("=" * 70)
        print()
        print(f"Track: {TRACK_NAME}")
        print()

        # Step 1: Load recorded lap
        print("Step 1: Loading recorded lap...")
        data = load_recorded_lap()
        if not data:
            print("✗ Error: Failed to load data")
            comm.Abort(1)
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
        print(f"✓ Created {len(checkpoints)} checkpoints (including finish line)")

        # Step 2.5: Filter and renumber checkpoints (if configured)
        if CHECKPOINTS_TO_REMOVE:
            print(f"\nStep 2.5: Filtering checkpoints (removing {len(CHECKPOINTS_TO_REMOVE)} checkpoints)...")
            checkpoints, finish_line_checkpoint_id = filter_and_renumber_checkpoints(
                checkpoints,
                CHECKPOINTS_TO_REMOVE
            )
            print()
        else:
            finish_line_checkpoint_id = FINISH_LINE_CHECKPOINT_ID
            print()

        # Step 3: Calculate genome length
        print("Step 3: Calculating genome length...")
        genome_length = calculate_genome_length(data)
        print(f"✓ Genome length: {genome_length} frames\n")

    else:
        # Workers: initialize empty variables (will receive via broadcast)
        checkpoints = None
        genome_length = None
        finish_line_checkpoint_id = None

    # ========================================================================
    # BROADCAST SETUP DATA (Master -> All Workers)
    # ========================================================================
    if is_master(rank):
        print("Step 4: Broadcasting setup data to all MPI ranks...")

    setup_data = broadcast_setup_data(
        comm, rank,
        checkpoints=checkpoints,
        genome_length=genome_length,
        track_path=TRACK_METADATA_PATH,
        finish_line_checkpoint_id=finish_line_checkpoint_id
    )

    # All ranks: unpack setup data
    checkpoints = setup_data['checkpoints']
    genome_length = setup_data['genome_length']
    TRACK_METADATA_PATH = setup_data['track_path']
    finish_line_checkpoint_id = setup_data['finish_line_checkpoint_id']

    if is_master(rank):
        print("✓ All ranks ready\n")

    # ========================================================================
    # INITIALIZE COLLISION SYSTEM (All ranks)
    # ========================================================================
    if is_master(rank):
        print("Step 5: Initializing collision system on all ranks...")

    # Suppress output on workers
    if not is_master(rank):
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    collision_system = CollisionSystem(TRACK_METADATA_PATH)

    if not is_master(rank):
        sys.stdout = old_stdout

    if is_master(rank):
        print("✓ Collision system ready on all ranks\n")

    # ========================================================================
    # EXTRACT TRACK CONTOURS (Master only, for visualizations)
    # ========================================================================
    if is_master(rank):
        print("Step 6: Extracting track contours for visualizations...")
        track_contours = extract_track_contours(collision_system, y_plane=1.0, verbose=False)
        print("✓ Track contours extracted\n")

    # ========================================================================
    # CREATE INITIAL POPULATION (Master only)
    # ========================================================================
    if is_master(rank):
        print("Step 7: Creating initial population...")
        population = create_initial_population(genome_length, POPULATION_SIZE)
        print(f"✓ Created {len(population)} individuals\n")

        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
    else:
        population = None

    # ========================================================================
    # EVOLUTION LOOP
    # ========================================================================
    if is_master(rank):
        print(f"Step 8: Evolution ({NUM_GENERATIONS_INITIAL} generations)...")
        print("=" * 70)

        # Visualization thread pool (master only)
        viz_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viz")
        viz_futures = []
        finish_line_crossed_gen = None

    for generation in range(NUM_GENERATIONS_INITIAL):
        if is_master(rank):
            print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS_INITIAL}")
            print("-" * 70)
            print(f"  Evaluating {len(population)} individuals (distributed across {size} MPI ranks)...", end=" ", flush=True)

        # DISTRIBUTED EVALUATION
        simulation_results, fitness_results = evaluate_population_mpi(
            comm, rank, size, population, checkpoints, TRACK_METADATA_PATH, cpus_per_rank, collision_system
        )

        if is_master(rank):
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

            # Check if finish line was crossed
            if finish_line_crossed_gen is None and finish_line_checkpoint_id in best['checkpoints_crossed']:
                finish_line_crossed_gen = generation + 1
                print(f"\n  🏁 FINISH LINE CROSSED! Running {EXTRA_GENERATIONS_AFTER_FINISH} more generations...")

            # Visualization (if enabled)
            if SAVE_ALL_PLOTS:
                viz_path = plots_dir / f"track_gen{generation + 1}.png"
                future = viz_executor.submit(
                    create_track_visualization,
                    checkpoints, collision_system, track_contours,
                    None, None, simulation_results,
                    TRACK_METADATA_PATH, str(viz_path), False
                )
                viz_futures.append((generation + 1, future))

            # Check for early termination
            should_terminate = False
            if finish_line_crossed_gen is not None:
                if generation + 1 >= finish_line_crossed_gen + EXTRA_GENERATIONS_AFTER_FINISH:
                    should_terminate = True
                    print(f"\n  ✓ Completed {EXTRA_GENERATIONS_AFTER_FINISH} generations after finish line")

            # Create next generation
            if generation < NUM_GENERATIONS_INITIAL - 1 and not should_terminate:
                print(f"  Creating next generation...", end=" ")
                population = create_next_generation(population)
                print("✓")
            else:
                # Final visualization
                if not SAVE_ALL_PLOTS:
                    viz_path = plots_dir / f"track_final.png"
                    future = viz_executor.submit(
                        create_track_visualization,
                        checkpoints, collision_system, track_contours,
                        None, None, simulation_results,
                        TRACK_METADATA_PATH, str(viz_path), False
                    )
                    viz_futures.append(("final", future))

                if should_terminate:
                    break

    # ========================================================================
    # FINALIZATION (Master only)
    # ========================================================================
    if is_master(rank):
        # Wait for visualizations
        if viz_futures:
            print("\nWaiting for background visualizations to complete...")
            for gen_id, future in viz_futures:
                try:
                    future.result()
                    print(f"  ✓ Visualization saved (gen {gen_id})")
                except Exception as e:
                    print(f"  ✗ Visualization failed (gen {gen_id}): {e}")
            viz_executor.shutdown(wait=True)
            print("✓ All visualizations complete")

        print("\n" + "=" * 70)
        print("EVOLUTION COMPLETE")
        print("=" * 70)
        print()

        # Save best individual
        print("Step 9: Saving best individual to file...")
        best_individual = get_population_stats(population)['best_individual']

        ga_runs_dir = Path("ga_runs")
        ga_runs_dir.mkdir(exist_ok=True)

        # Convert genome to game format
        snapshots = []
        for gene in best_individual['genome']:
            axis_fb, axis_lr = gene
            forward, backward, left, right = to_game_format(axis_fb, axis_lr)
            snapshot = ControlSnapshot(forward, backward, left, right)
            snapshots.append(snapshot)

        # Determine next run number
        existing_runs = list(ga_runs_dir.glob("run_*.npz"))
        if existing_runs:
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

        # Save using lzma compression
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