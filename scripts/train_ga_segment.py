"""
Entry point for training genetic algorithm on a segment.

This script:
1. Sets up logging
2. Calls train_segment() with configuration
3. Saves results
4. Provides clean user interface
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.train_segment import train_segment
from genetics_algo.utils import Logger


# ============ CONFIGURATION ============
SEGMENT_ID = 0
NUM_GENERATIONS = 10  # Run multiple generations to see evolution
POPULATION_SIZE = 5
MUTATION_RATE = 0.10  # 10% mutation rate - crashes filter bad ones
WAIT_TIME_SECONDS = 2.0
VERBOSE_FRAMES = False  # Turn off frame-by-frame logging (too much output)

# Heuristic mutation settings
USE_HEURISTIC = True  # Use direction-based mutation guidance
HEURISTIC_BIAS = 0.7  # 70% bias toward checkpoint, 30% random exploration
# =======================================


def main():
    # Create logger with stdout capture
    with Logger(log_dir="logs", prefix="train_ga", console_echo=True, capture_stdout=True) as logger:

        print("="*60)
        print("GENETIC ALGORITHM TRAINING")
        print("="*60)
        print(f"Configuration:")
        print(f"  Segment: {SEGMENT_ID}")
        print(f"  Generations: {NUM_GENERATIONS}")
        print(f"  Population: {POPULATION_SIZE}")
        print(f"  Mutation rate: {MUTATION_RATE}")
        print(f"  Heuristic mutations: {'Enabled' if USE_HEURISTIC else 'Disabled'}")
        if USE_HEURISTIC:
            print(f"  Heuristic bias: {HEURISTIC_BIAS} ({HEURISTIC_BIAS*100:.0f}% guided, {(1-HEURISTIC_BIAS)*100:.0f}% random)")
        print(f"  Wait time: {WAIT_TIME_SECONDS}s")
        print(f"  Log file: {logger.log_path}")
        print("="*60)
        print()

        # Run training
        history = train_segment(
            segment_id=SEGMENT_ID,
            num_generations=NUM_GENERATIONS,
            pop_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            wait_time_seconds=WAIT_TIME_SECONDS,
            use_heuristic=USE_HEURISTIC,
            heuristic_bias=HEURISTIC_BIAS
        )

        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Segment: {SEGMENT_ID}")
        print(f"Generations completed: {len(history['generations'])}")
        print(f"Best fitness achieved: {history['best_fitness']:.2f}")
        print()

        print("Generation-by-generation progress:")
        for gen_stats in history['generations']:
            print(f"  Gen {gen_stats['generation']:2d}: "
                  f"best={gen_stats['best_fitness']:8.2f}, "
                  f"avg={gen_stats['avg_fitness']:8.2f}, "
                  f"success={gen_stats['success_count']}/{POPULATION_SIZE}")

        print()
        print(f"Best individual genome length: {len(history['best_individual'])}")
        print(f"Best trajectory points: {len(history['best_trajectory'])}")

        print()
        print(f"Log saved to: {logger.log_path}")
        print("="*60)


if __name__ == "__main__":
    main()