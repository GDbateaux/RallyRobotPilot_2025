"""
Entry point for training genetic algorithm on segments.

CONFIGURATION
=============
Edit genetics_algo/config.py to change ALL training parameters:
- Number of segments to train (MAX_SEGMENTS)
- Population size (POPULATION_SIZE)
- Number of generations (NUM_GENERATIONS_INITIAL, NUM_GENERATIONS_MAX)
- Mutation settings (MUTATION_RATE, HEURISTIC_BIAS)
- Visualization options (PLOT_DURING_TRAINING, PLOT_AFTER_TRAINING)
- Track parameters (NUM_CHECKPOINTS, CHECKPOINT_WIDTH)
- And more...

This script uses the NEW rallyrobopilot_novisual physics engine for ~10x speedup!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.train_full_lap import train_full_lap
from genetics_algo.utils import Logger
from genetics_algo.config import (
    MAX_SEGMENTS,
    POPULATION_SIZE,
    NUM_GENERATIONS_INITIAL,
    NUM_GENERATIONS_MAX,
    MUTATION_RATE,
    USE_HEURISTIC,
    HEURISTIC_BIAS,
    ENABLE_CROSSOVER,
    CROSSOVER_RATE,
    CROSSOVER_TYPE,
    ELITE_COUNT,
    TOURNAMENT_SIZE,
    WAIT_TIME_SECONDS,
    PLOT_DURING_TRAINING,
    PLOT_AFTER_TRAINING,
    SAVE_PLOTS,
    VERBOSE_OUTPUT,
    USE_HEADLESS
)
from genetics_algo.visualizer import GAVisualizer
from genetics_algo.run_manager import RunManager
from genetics_algo.track_map import plot_full_track_map


def main():
    import argparse

    # Parse command line arguments (can override config file)
    parser = argparse.ArgumentParser(
        description='Train genetic algorithm on track segments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  Edit genetics_algo/config.py to change training parameters.
  Command-line arguments override config file settings.

Examples:
  # Use config file settings
  python scripts_novisual/train_ga_full_lap.py

  # Override max segments
  python scripts_novisual/train_ga_full_lap.py --max-segments 3

  # Override population and generations
  python scripts_novisual/train_ga_full_lap.py --pop-size 10 --generations 10

  # Force visual mode (slower but you can see the car)
  python scripts_novisual/train_ga_full_lap.py --visual
        """
    )
    parser.add_argument('--max-segments', type=int, default=MAX_SEGMENTS,
                       help=f'Max segments to train (default: {MAX_SEGMENTS})')
    parser.add_argument('--pop-size', type=int, default=POPULATION_SIZE,
                       help=f'Population size (default: {POPULATION_SIZE})')
    parser.add_argument('--generations', type=int, default=NUM_GENERATIONS_INITIAL,
                       help=f'Initial generations (default: {NUM_GENERATIONS_INITIAL})')
    parser.add_argument('--max-generations', type=int, default=NUM_GENERATIONS_MAX,
                       help=f'Max generations if no solution (default: {NUM_GENERATIONS_MAX})')
    parser.add_argument('--headless', action='store_true', default=USE_HEADLESS,
                       help='Use headless mode (fast physics, default: True)')
    parser.add_argument('--visual', action='store_true',
                       help='Force visual mode (slow, overrides --headless)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable all plotting')
    args = parser.parse_args()

    # Override headless if visual requested
    if args.visual:
        headless = False
    else:
        headless = args.headless

    # Determine plotting
    plot_enabled = not args.no_plot and (PLOT_DURING_TRAINING or PLOT_AFTER_TRAINING)

    # Create run manager for organized output
    run_manager = RunManager(base_dir="ga_runs")

    # Save configuration
    config_dict = {
        'timestamp': run_manager.timestamp,
        'max_segments': args.max_segments,
        'population_size': args.pop_size,
        'num_generations_initial': args.generations,
        'num_generations_max': args.max_generations,
        'mutation_rate': MUTATION_RATE,
        'use_heuristic': USE_HEURISTIC,
        'heuristic_bias': HEURISTIC_BIAS,
        'use_headless': headless,
        'wait_time_seconds': WAIT_TIME_SECONDS
    }
    run_manager.save_config(config_dict)

    # Use run manager's log path instead of default logs/ directory
    # Create logger with stdout capture
    with Logger(log_path=run_manager.get_log_path(), console_echo=True, capture_stdout=True) as logger:

        print("="*70)
        print("GENETIC ALGORITHM - SEGMENT TRAINING")
        print("="*70)
        print(f"🚀 Using NEW rallyrobopilot_novisual physics engine!")
        print(f"📁 Run directory: {run_manager.run_dir}")
        print(f"Configuration:")
        print(f"  Max segments: {args.max_segments} (train segments 0 to {args.max_segments-1})")
        print(f"  Mode: {'HEADLESS (fast physics)' if headless else 'VISUAL (with window)'}")
        print(f"  Population size: {args.pop_size}")
        print(f"  Initial generations: {args.generations} (stop if solution found)")
        print(f"  Max generations: {args.max_generations} (if no solution)")
        print(f"  Mutation rate: {MUTATION_RATE}")
        print(f"  Heuristic mutations: {'Enabled' if USE_HEURISTIC else 'Disabled'}")
        if USE_HEURISTIC:
            print(f"  Heuristic bias: {HEURISTIC_BIAS}")
        print(f"  Wait time: {WAIT_TIME_SECONDS}s")
        print(f"  Visualization: {'Enabled' if plot_enabled else 'Disabled'}")
        print(f"  Log file: {run_manager.get_log_path()}")
        print("="*70)
        print()

        # Create visualizer if plotting enabled
        visualizer = None
        if plot_enabled:
            visualizer = GAVisualizer(auto_save=SAVE_PLOTS, run_manager=run_manager)
            print("✓ Visualizer initialized")
            print()

        # Run training with MAX_SEGMENTS limit
        result = train_full_lap(
            num_generations_per_segment=args.generations,
            num_generations_max=args.max_generations,
            pop_size=args.pop_size,
            mutation_rate=MUTATION_RATE,
            wait_time_seconds=WAIT_TIME_SECONDS if not headless else 0.0,  # No wait for headless
            use_heuristic=USE_HEURISTIC,
            heuristic_bias=HEURISTIC_BIAS,
            use_crossover=ENABLE_CROSSOVER,
            crossover_rate=CROSSOVER_RATE,
            crossover_type=CROSSOVER_TYPE,
            elite_count=ELITE_COUNT,
            tournament_size=TOURNAMENT_SIZE,
            transition_overlap_frames=5,
            use_smooth_transitions=False,
            headless=headless,
            max_segments=args.max_segments,  # NEW: Limit segments
            visualizer=visualizer  # NEW: Pass visualizer
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
            if seg_result['original_length'] is not None and seg_result['original_length'] > 0:
                improvement_pct = (seg_result['improvement'] / seg_result['original_length']) * 100
                print(f"  Segment {seg_result['segment_id']:2d}: "
                      f"{seg_result['original_length']:3d} → {seg_result['evolved_length']:3d} frames "
                      f"({seg_result['improvement']:+3d}, {improvement_pct:+6.1f}%) | "
                      f"fitness={seg_result['best_fitness']:.2f}")
            else:
                # Evolved segment - no comparison to original
                print(f"  Segment {seg_result['segment_id']:2d}: "
                      f"{seg_result['evolved_length']:3d} frames (evolved) | "
                      f"fitness={seg_result['best_fitness']:.2f}")

        print()
        print(f"Super individual genome length: {len(result['super_individual'])} total controls")
        print()
        print(f"📁 All results saved to: {run_manager.run_dir}")
        print(f"   - Plots: {run_manager.plots_dir}")
        print(f"   - Controls: {run_manager.controls_dir}")
        print(f"   - Log: {run_manager.get_log_path()}")
        print("="*70)

        # Save run summary
        run_manager.save_summary(result)

        # Generate final visualization if enabled
        if visualizer and PLOT_AFTER_TRAINING:
            print("\n" + "="*70)
            print("GENERATING FINAL PLOTS")
            print("="*70)

            # Generate full track map
            plot_full_track_map(run_manager=run_manager)

            # Generate lap summary
            visualizer.plot_full_lap_summary(result['segment_results'], show=False)
            visualizer.print_full_lap_summary(
                result['segment_results'],
                {
                    'total_original_frames': result['total_original_frames'],
                    'total_evolved_frames': result['total_evolved_frames'],
                    'total_improvement': result['total_improvement'],
                    'improvement_percentage': result['improvement_percentage']
                }
            )
            print("="*70)


if __name__ == "__main__":
    main()
