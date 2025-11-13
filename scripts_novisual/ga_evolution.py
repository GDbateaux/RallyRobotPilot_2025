"""
Genetic Algorithm Evolution Demo
Demonstrates the complete evolution pipeline: evaluation → selection → mutation → next generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.lap_data import load_recorded_lap, cleanup_recording
from genetics_algo.checkpoint import create_checkpoints_from_data, calculate_checkpoint_line
from genetics_algo.config import NUM_CHECKPOINTS, CHECKPOINT_WIDTH, POPULATION_SIZE, NUM_GENERATIONS_INITIAL, FINISH_LINE_CHECKPOINT_ID, EXTRA_GENERATIONS_AFTER_FINISH
from genetics_algo.ga_simple import calculate_genome_length, create_initial_population
from genetics_algo.simulator import simulate_individual
from genetics_algo.fitness import calculate_fitness
from genetics_algo.evolution import augment_population_with_results, create_next_generation, get_population_stats
from genetics_algo.individual import to_game_format
from rallyrobopilot_novisual import CollisionSystem
import pickle
import lzma


class ControlSnapshot:
    """Simple snapshot with controls for replay."""
    def __init__(self, forward, backward, left, right):
        self.current_controls = (forward, backward, left, right)


def evaluate_population(population, collision_system, checkpoints):
    """
    Evaluate entire population: simulate + calculate fitness.

    Args:
        population (list): List of individuals
        collision_system (CollisionSystem): Collision system for simulation
        checkpoints (list): Track checkpoints

    Returns:
        tuple: (simulation_results, fitness_results)
    """
    simulation_results = []
    fitness_results = []

    for individual in population:
        # Simulate
        sim_result = simulate_individual(individual['genome'], collision_system)
        simulation_results.append(sim_result)

        # Calculate fitness (pass genome_size for finish line speed bonus)
        genome_size = len(individual['genome'])
        fit_result = calculate_fitness(sim_result, checkpoints, debug=False, genome_size=genome_size)
        fitness_results.append(fit_result)

    return simulation_results, fitness_results


def save_best_individual(individual, generation, output_dir="ga_runs"):
    """
    Save best individual to file.

    Args:
        individual (dict): Best individual
        generation (int): Generation number
        output_dir (str): Output directory
    """
    ga_runs_dir = Path(output_dir)
    ga_runs_dir.mkdir(exist_ok=True)

    # Convert genome to game format
    snapshots = []
    for gene in individual['genome']:
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

    return output_path


def main():
    print("=" * 70)
    print("GENETIC ALGORITHM - EVOLUTION DEMO")
    print("=" * 70)
    print()

    # Setup
    print("Step 1: Setup...")
    data = load_recorded_lap()
    if not data:
        print("✗ Error: Failed to load data")
        return
    data = cleanup_recording(data)
    print(f"✓ Loaded {len(data)} frames")

    # Create checkpoints
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
    print(f"✓ Created {len(checkpoints)} checkpoints")

    # Initialize collision system
    collision_system = CollisionSystem("SimpleTrack/track_metadata.json")
    print("✓ Collision system ready")

    # Calculate genome length
    genome_length = calculate_genome_length(data)
    print(f"✓ Genome length: {genome_length} frames\n")

    # Create initial population
    print(f"Step 2: Creating initial population ({POPULATION_SIZE} individuals)...")
    population = create_initial_population(genome_length, POPULATION_SIZE)
    print(f"✓ Created population\n")

    # Evolution loop
    print(f"Step 3: Evolution ({NUM_GENERATIONS_INITIAL} generations)...")
    print("=" * 70)

    # Track when finish line is crossed for early termination
    finish_line_crossed_gen = None

    for generation in range(NUM_GENERATIONS_INITIAL):
        print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS_INITIAL}")
        print("-" * 70)

        # Evaluate population
        print(f"  Evaluating {len(population)} individuals...", end=" ")
        simulation_results, fitness_results = evaluate_population(population, collision_system, checkpoints)
        print("✓")

        # Augment population with results
        population = augment_population_with_results(population, simulation_results, fitness_results)

        # Get statistics
        stats = get_population_stats(population)

        # Print stats
        print(f"  Best fitness:  {stats['best_fitness']:.1f}")
        print(f"  Avg fitness:   {stats['avg_fitness']:.1f}")
        print(f"  Worst fitness: {stats['worst_fitness']:.1f}")
        print(f"  Checkpoints:   {stats['max_checkpoints']}")
        print(f"  Collisions:    {stats['collisions']}/{len(population)}")

        best = stats['best_individual']
        print(f"  Best: {best['checkpoints_crossed_count']} checkpoints {best['checkpoints_crossed']}", end="")
        if best['collision_detected']:
            print(f" (collision at frame {best['collision_frame']})")
        else:
            print(" (no collision)")

        # Check if finish line was crossed (for early termination)
        if finish_line_crossed_gen is None and FINISH_LINE_CHECKPOINT_ID in best['checkpoints_crossed']:
            finish_line_crossed_gen = generation + 1
            print(f"\n  🏁 FINISH LINE CROSSED! Running {EXTRA_GENERATIONS_AFTER_FINISH} more generations...")

        # Check if should terminate early (finish line + extra generations complete)
        should_terminate = False
        if finish_line_crossed_gen is not None:
            if generation + 1 >= finish_line_crossed_gen + EXTRA_GENERATIONS_AFTER_FINISH:
                should_terminate = True
                print(f"\n  ✓ Completed {EXTRA_GENERATIONS_AFTER_FINISH} generations after finish line")
                break

        # Create next generation (except for last generation)
        if generation < NUM_GENERATIONS_INITIAL - 1:
            population = create_next_generation(population)
            print(f"  ✓ Next generation created")

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)

    # Save best individual
    best_individual = get_population_stats(population)['best_individual']
    output_path = save_best_individual(best_individual, NUM_GENERATIONS_INITIAL)
    print(f"\n✓ Best individual saved to: {output_path}")
    print(f"  Fitness: {best_individual['fitness']:.1f}")
    print(f"  Checkpoints: {best_individual['checkpoints_crossed_count']} {best_individual['checkpoints_crossed']}")
    if best_individual['collision_detected']:
        print(f"  Collision at frame: {best_individual['collision_frame']}")
    else:
        print(f"  No collision!")
    print()


if __name__ == "__main__":
    main()