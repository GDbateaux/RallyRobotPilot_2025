"""
Entry point for training genetic algorithm on the full lap.

This script trains each segment sequentially and merges them into a super individual.

ADAPTIVE MODE: Ensure solution, optimize if found early
- 5 individuals per generation
- Run 5 generations initially
- If checkpoint crossed in first 5: STOP and select best
- If NO checkpoint crossed in first 5: Continue until solution found
- Guarantees we ALWAYS get a working solution
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.train_full_lap import train_full_lap
from genetics_algo.utils import Logger


# ============ ADAPTIVE CONFIGURATION ============
# Strategy: Ensure solution, optimize if found early
# - Run 5 generations initially
# - If checkpoint crossed in first 5: STOP and select best
# - If NO checkpoint crossed: Continue until solution found
NUM_GENERATIONS_INITIAL = 10        # Initial generations to try
NUM_GENERATIONS_MAX = 100          # Max generatsions if no solution found
POPULATION_SIZE = 10                # 5 individuals per population
# =================================================

# Regular configuration
MUTATION_RATE = 0.25  # 25% mutation rate (matches config.py)
WAIT_TIME_SECONDS = 1.0

# Heuristic mutation settings
USE_HEURISTIC = True    # Use direction-based mutation guidance
HEURISTIC_BIAS = 0.15   # +10% bonus toward checkpoint direction (was 0.0 before direction fix)

# Transition settings (for future smooth transitions feature)
USE_SMOOTH_TRANSITIONS = False  # TODO: Implement smooth transition logic
TRANSITION_OVERLAP = 5          # Frames to overlap between segments (when smooth transitions enabled)


def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train genetic algorithm on full lap')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no window, 100 FPS, 10x faster)')
    args = parser.parse_args()

    # Create logger with stdout capture
    with Logger(log_dir="logs", prefix="train_full_lap", console_echo=True, capture_stdout=True) as logger:

        print("="*70)
        print("GENETIC ALGORITHM - FULL LAP TRAINING")
        print("="*70)
        print(f"🎯 ADAPTIVE MODE: Ensure solution, optimize if found early")
        print(f"Configuration:")
        print(f"  Mode: {'HEADLESS (no window, 10x faster)' if args.headless else 'VISUAL (with window)'}")
        print(f"  Initial generations: {NUM_GENERATIONS_INITIAL} (stop if solution found)")
        print(f"  Max generations: {NUM_GENERATIONS_MAX} (if no solution)")
        print(f"  Population size: {POPULATION_SIZE}")
        print(f"  Mutation rate: {MUTATION_RATE}")
        print(f"  Heuristic mutations: {'Enabled' if USE_HEURISTIC else 'Disabled'}")
        if USE_HEURISTIC:
            if HEURISTIC_BIAS == 0.0:
                print(f"  Heuristic bias: {HEURISTIC_BIAS} (uniform random - no directional bias)")
            else:
                print(f"  Heuristic bias: {HEURISTIC_BIAS} (~{33 + HEURISTIC_BIAS*100:.0f}% toward checkpoint)")
        print(f"  Wait time: {WAIT_TIME_SECONDS}s")
        print(f"  Log file: {logger.log_path}")
        print("="*70)
        print()

        # Run full lap training
        result = train_full_lap(
            num_generations_per_segment=NUM_GENERATIONS_INITIAL,
            num_generations_max=NUM_GENERATIONS_MAX,
            pop_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            wait_time_seconds=WAIT_TIME_SECONDS,
            use_heuristic=USE_HEURISTIC,
            heuristic_bias=HEURISTIC_BIAS,
            transition_overlap_frames=TRANSITION_OVERLAP,
            use_smooth_transitions=USE_SMOOTH_TRANSITIONS,
            headless=args.headless
        )

        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total segments trained: {len(result['segment_results'])}")
        print(f"Original total frames: {result['total_original_frames']}")
        print(f"Evolved total frames: {result['total_evolved_frames']}")
        print(f"Total improvement: {result['total_improvement']} frames ({result['improvement_percentage']:.1f}%)")
        print()

        print("Per-segment results:")
        for seg_result in result['segment_results']:
            improvement_pct = (seg_result['improvement'] / seg_result['original_length']) * 100 if seg_result['original_length'] > 0 else 0
            print(f"  Segment {seg_result['segment_id']:2d}: "
                  f"{seg_result['original_length']:3d} → {seg_result['evolved_length']:3d} frames "
                  f"({seg_result['improvement']:+3d}, {improvement_pct:+6.1f}%) | "
                  f"fitness={seg_result['best_fitness']:.2f}")

        print()
        print(f"Super individual genome length: {len(result['super_individual'])} total controls")
        print()
        print(f"Log saved to: {logger.log_path}")
        print("="*70)


if __name__ == "__main__":
    main()
