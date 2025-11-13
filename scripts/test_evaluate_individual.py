"""
Test evaluation of a single individual on any segment.

This script validates that:
1. Segment data loads correctly
2. Individual can be created from recorded controls
3. Game integration works (Simulator + DirectController)
4. Checkpoint detection works
5. Fitness calculation returns expected score

Expected result: Recorded controls should cross checkpoint with fitness > 10000
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genetics_algo.simulator import setup_game_for_simulation, evaluate_individual
from genetics_algo.lap_data import get_segment_info
from genetics_algo.individual import Individual
from genetics_algo.utils import Logger


# ============ CONFIGURATION ============
SEGMENT_ID = 0  # Change this to test different segments (0, 1, 2, ...)
WAIT_TIME_SECONDS = 2.0  # Time to wait after game setup before starting controls
# =======================================


def main():
    # Create logger - auto-increments run number
    with Logger(log_dir="logs", prefix="test_individual", console_echo=True) as logger:

        logger.log("=" * 60)
        logger.log("TESTING SINGLE INDIVIDUAL EVALUATION")
        logger.log("=" * 60)
        logger.log()

        logger.log(f"Configuration:")
        logger.log(f"  Segment ID: {SEGMENT_ID}")
        logger.log(f"  Wait time: {WAIT_TIME_SECONDS}s")
        logger.log()

        # Load segment data
        logger.log(f"Step 1: Loading segment {SEGMENT_ID} data...")
        segment_info = get_segment_info(SEGMENT_ID)
        logger.log(f"✓ Segment {SEGMENT_ID} loaded: {segment_info['length']} frames")
        logger.log(f"  Start frame: {segment_info['start_frame']}")
        logger.log(f"  End frame: {segment_info['end_frame']}")
        logger.log(f"  Start state: speed={segment_info['start_state']['speed']:.2f}, "
                   f"angle={segment_info['start_state']['angle']:.1f}")
        logger.log()

        # Create individual from recorded controls
        logger.log(f"Step 2: Creating individual from recorded controls (segment {SEGMENT_ID})...")
        individual = Individual.from_segment(SEGMENT_ID)
        logger.log(f"✓ Individual created: {individual}")
        logger.log()

        # Setup game
        logger.log("Step 3: Setting up game environment...")
        app, car, controller, simulator = setup_game_for_simulation()
        logger.log()

        # Evaluate individual
        logger.log("Step 4: Evaluating individual...")
        fitness = evaluate_individual(individual, segment_info, simulator, app,
                                      wait_time_seconds=WAIT_TIME_SECONDS)
        logger.log()

        # Final results
        logger.log("=" * 60)
        logger.log("TEST RESULTS")
        logger.log("=" * 60)
        logger.log(f"Final fitness: {fitness:.2f}")

        if fitness > 10000:
            logger.log("✓ SUCCESS: Checkpoint crossed! (fitness > 10000)")
        else:
            logger.log("✗ FAILURE: Checkpoint not crossed (fitness < 10000)")

        logger.log()
        logger.log(f"Log saved to: {logger.log_path}")
        logger.log("Closing game window...")


if __name__ == "__main__":
    main()