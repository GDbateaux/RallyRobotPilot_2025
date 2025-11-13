"""
GENETIC ALGORITHM CONFIGURATION
================================
ALL parameters for GA training centralized in ONE place.
Edit these values to change GA behavior - no need to edit code!

Quick Start:
- For testing: MAX_SEGMENTS=1, POPULATION_SIZE=5, NUM_GENERATIONS_INITIAL=5
- For full lap: MAX_SEGMENTS=20, POPULATION_SIZE=10, NUM_GENERATIONS_INITIAL=10
"""

# ============================================================================
# TRACK CONFIGURATION
# ============================================================================
# Path to recorded lap data
RECORDED_LAP_PATH = "runs/simpletrack_recorded_1.npz"

# Track geometry
NUM_CHECKPOINTS = 30          # Total checkpoints in the track (for lap subdivision - DO NOT CHANGE)
                              # This determines how the recorded lap is subdivided into checkpoints
FINISH_LINE_CHECKPOINT_ID = NUM_CHECKPOINTS - 1   # Checkpoint ID for the finish line
                              # Note: Checkpoint 29 was removed (too close to start), so finish line is renumbered from 30→29
CHECKPOINT_WIDTH = 50.0       # Width of checkpoint lines (meters)

# ============================================================================
# TRAINING SCOPE
# ============================================================================
MAX_SEGMENTS = 2              # How many segments to train
                              # 1 = first segment only
                              # 20 = full lap (all checkpoints)
                              # 3 = first 3 segments (recommended for testing)

# ============================================================================
# GENOME SETTINGS
# ============================================================================
GENOME_LENGTH_MULTIPLIER = 2  # Genome length = multiplier * recorded_lap_length
                              # Gives individuals extra time to complete the lap

# Random initialization probabilities for genome generation
# Throttle (axis_fb) probabilities - should sum to 1.0
INIT_PROB_FORWARD = 1.0     # Probability of forward (axis_fb = 1)
INIT_PROB_COAST = 0.0       # Probability of coast (axis_fb = 0)
INIT_PROB_BACKWARD = 0.0     # Probability of backward (axis_fb = -1)

# Steering (axis_lr) probabilities - equal by default
INIT_PROB_LEFT = 0          # Probability of left (axis_lr = -1)
INIT_PROB_STRAIGHT = 1.0      # Probability of straight (axis_lr = 0)
INIT_PROB_RIGHT = 0         # Probability of right (axis_lr = 1)

# ============================================================================
# FITNESS FUNCTION PARAMETERS
# ============================================================================
# Rewards
CHECKPOINT_BONUS = 1000        # Points awarded per checkpoint crossed (in order)
SURVIVAL_POINTS_PER_FRAME = 0.0 # Points per frame survived

# Penalties
COLLISION_PENALTY = -500    # Large penalty for collision
OUT_OF_ORDER_PENALTY = -10000  # Penalty for crossing checkpoints out of order
OUT_OF_BOUNDS_PENALTY = -100000  # Penalty for being too far from last checkpoint (illegal wall crossing)
OUT_OF_BOUNDS_DISTANCE = 150    # Distance threshold (units) - if car is farther than this from last checkpoint, apply penalty
SPEED_PENALTY_FACTOR = 1.0    # Multiplier for speed penalty
                              # Penalty = checkpoints_crossed * frames_to_last * factor
                              # Lower frames = lower penalty = reward for speed

FINISH_LINE_SPEED_BONUS_FACTOR = 10.0  # Bonus for completing lap faster
                              # Bonus = (genome_size - finish_frame) * factor
                              # Faster lap = more frames saved = higher bonus

ENABLE_SURVIVAL_BONUS = False # Enable/disable survival points bonus
                              # True: Rewards longer runs (can favor slow individuals)
                              # False: Focus purely on checkpoints + speed (better for racing)

# Finish line validation
FINISH_LINE_MIN_FRAME = 50    # Minimum frame before finish line can be crossed
                              # Prevents immediate validation at start

# ============================================================================
# POPULATION SETTINGS
# ============================================================================
POPULATION_SIZE = 40          # Number of individuals per generation
                              # Small (3-5): Fast, good for testing
                              # Medium (8-12): Balanced
                              # Large (15-20): Slower but more thorough

# ============================================================================
# GENERATION SETTINGS
# ============================================================================
NUM_GENERATIONS_INITIAL = 100   # Initial generations to try
                              # Training stops early if solution found

EXTRA_GENERATIONS_AFTER_FINISH = 5  # Continue for N generations after finish line crossed
                              # Allows optimization after completing the lap
                              # Set to 0 to stop immediately when finish line reached

NUM_GENERATIONS_MAX = 10      # Maximum generations if no solution found
                              # Continues until solution or this limit

# LEGACY (not used, kept for backwards compatibility)
NUM_GENERATIONS = 30          # Old parameter, use NUM_GENERATIONS_INITIAL instead

# ============================================================================
# EVOLUTION STRATEGY
# ============================================================================
# Selection
ELITE_PERCENTAGE = 0.10       # Percentage of population to keep as elite
                              # 0.10 = Top 10% go directly to next generation

TOURNAMENT_SIZE = 5           # Number of individuals in each tournament
                              # 3-5: Balanced selection pressure
                              # 7-10: Strong selection (favors best)

# Mutation
MUTATION_WINDOW_SIZE = 15     # Frames before crash to mutate
                              # Mutation happens in [crash_frame - window, crash_frame]
                              # Smaller = more focused, Larger = more exploration

MUTATION_RATE_NO_CRASH = 0.2  # Probability to mutate individuals that didn't crash
                              # 0.0 = Clone them unchanged
                              # 0.1 = 10% c
# hance to mutate anyway

MUTATION_DURATION_MIN = 3     # Minimum consecutive frames to mutate
                              # Prevents tiny single-frame changes
                              # 5 = At least 5 frames affected (sustained behaviors)

MUTATION_DURATION_MAX = 20    # Maximum consecutive frames to mutate
                              # Random duration between MIN and MAX
                              # Example: MIN=5, MAX=20 → durations of 5-20 frames
                              # Higher = more sustained behaviors (better for turns and straights)
                              # Lower = more fine-grained control

BRAKE_MUTATION_DURATION_MAX = 5  # Maximum frames for BRAKE mutations specifically
                              # Brake mutations use shorter duration (MIN to this value)
                              # Rationale: Full braking to speed=0 takes 9 frames, rarely optimal for racing
                              # Short brake bursts (3-5 frames) better for speed control without killing momentum

NUM_MUTATIONS_PER_EVENT = 1   # Number of mutations to apply per event (1-3)
                              # 1 = Single mutation (conservative)
                              # 2-3 = Multiple mutations (more exploration)
                              # Higher = more variation per generation

# Mutation probabilities for throttle (axis_fb) - should sum to 1.0
MUTATION_PROB_FORWARD = 0.60  # Probability of forward throttle mutation
MUTATION_PROB_COAST = 0.10    # Probability of no throttle (coast) mutation
MUTATION_PROB_BACKWARD = 0.30 # Probability of backward throttle mutation

# Mutation probabilities for steering (axis_lr) - equal by default
MUTATION_PROB_LEFT = 1/3      # Probability of left steering mutation
MUTATION_PROB_STRAIGHT = 1/3  # Probability of straight steering mutation
MUTATION_PROB_RIGHT = 1/3     # Probability of right steering mutation

# # Legacy mutation rate (for old mutation strategy, if needed)
# MUTATION_RATE = 0.15          # Probability of mutating each control frame
#                               # Low (0.05-0.15): Conservative, slow improvement
#                               # Medium (0.15-0.30): Balanced (recommended)
#                               # High (0.30-0.50): Aggressive, more exploration
#
# USE_HEURISTIC = False          # Enable direction-based mutation guidance
#                               # True: Mutations biased toward checkpoint
#                               # False: Pure random mutations
#
# HEURISTIC_BIAS = 0.15         # Strength of heuristic guidance (if enabled)
#                               # 0.10: Subtle bias (~25% toward checkpoint)
#                               # 0.15: Medium bias (~43% toward checkpoint)
#                               # 0.20: Strong bias (~56% toward checkpoint)

# Elite selection
ELITE_COUNT = 1               # Number of best individuals to keep unchanged
                              # 1 = Keep only the best
                              # 2-3 = Keep top 2-3 for more stability

# Crossover (Sexual Reproduction)
ENABLE_CROSSOVER = True       # Enable crossover (mixing genes from 2 parents)
                              # True: Sexual reproduction (tournament + crossover)
                              # False: Asexual reproduction (only mutations)

CROSSOVER_RATE = 0.7          # Probability of crossover vs cloning
                              # 0.7 = 70% crossover, 30% cloning
                              # Higher = more gene mixing

CROSSOVER_TYPE = 'single_point'  # Type of crossover
                              # 'single_point': Cut at one point, swap tails
                              # 'two_point': Cut at two points, swap middle
                              # 'uniform': Each gene randomly from either parent

TOURNAMENT_SIZE = 3           # Tournament selection size
                              # 3 = Pick 3 random individuals, select best
                              # Larger = more selection pressure (favors best)
                              # Smaller = more diversity (gives weaker a chance)

# Collision detection
ENABLE_COLLISION_DETECTION = True  # Enable collision detection and penalties

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# Segment settings
SEGMENT_OVERLAP_FRAMES = 2    # Frames to overlap between segments
                              # Example: Segment 0 ends at frame 24
                              #          Segment 1 starts at frame 22 (24 - 2)
                              # This creates a 2-frame overlap for smoother transitions

SEGMENT_TIMEOUT_BUFFER = 1   # Extra frames beyond segment length
                              # sExample: If segment is 24 frames, timeout = 24 + 10 = 34 frames
                              # This gives individuals some exploration time beyond the
                              # recorded optimal path (useful for trying different strategies)

# Physics
DT = 0.1                      # Physics timestep (seconds)
                              # 0.1 = 10 FPS physics simulation

# Timeouts
MAX_TIMEOUT_FRAMES = 500      # Global maximum simulation frames (absolute safety limit)
                              # Prevents infinite loops if individual gets stuck
                              # Per-segment timeout = segment_length + SEGMENT_TIMEOUT_BUFFER
                              # Example timeline for a 24-frame segment:
                              #   - Frame 0-23: Normal controls
                              #   - Frame 24-25: Buffer frames after checkpoint crossing (visualization)
                              #   - Frame 26-33: Extra exploration time (SEGMENT_TIMEOUT_BUFFER)
                              #   - Frame 34: Timeout triggered if checkpoint not crossed
                              # If timeout reached without crossing checkpoint:
                              #   → Simulation stops
                              #   → Fitness calculated based on distance to checkpoint (negative)
                              #   → Individual marked as failed

WAIT_TIME_SECONDS = 0.0       # Wait before first evaluation
                              # 0.0: No wait (headless mode)
                              # 1.0-2.0: Wait for visual mode

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_DURING_TRAINING = True   # Generate plots after each segment
PLOT_AFTER_TRAINING = True    # Generate summary plots at end
SAVE_PLOTS = True             # Save plots to disk
SAVE_ALL_PLOTS = True        # Save plot for every generation (True) or only final (False)
                              # True: Creates plots/track_gen1.png, track_gen2.png, ... track_genN.png
                              # False: Only creates final plot after all generations
VERBOSE_OUTPUT = True         # Print detailed progress information

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
USE_HEADLESS = True           # Use fast headless physics engine
                              # True: ~10x faster, no visualization
                              # False: Slower, but you can see the car

# NUM_WORKERS = 20              # Number of parallel workers for simulation
NUM_WORKERS = 20            # Number of parallel workers for simulation
                              # Set to number of CPU cores available
                              # 1 = Sequential (no parallelization)
                              # 20 = Use 20 cores in parallel
                              # Set to -1 to use all available cores

# ============================================================================
# NETWORK SETTINGS (for remote game connection)
# ============================================================================
GAME_HOST = "127.0.0.1"       # Game server host
GAME_PORT = 5000              # Game server port

# ============================================================================
# QUICK PRESETS
# ============================================================================
# Uncomment one of these blocks to quickly change settings:

# PRESET: Quick Test (1 segment, fast)
# MAX_SEGMENTS = 1
# POPULATION_SIZE = 3
# NUM_GENERATIONS_INITIAL = 3
# NUM_GENERATIONS_MAX = 20
# MUTATION_RATE = 0.30

# PRESET: Balanced (3 segments, good quality)
# MAX_SEGMENTS = 3
# POPULATION_SIZE = 8
# NUM_GENERATIONS_INITIAL = 8
# NUM_GENERATIONS_MAX = 50
# MUTATION_RATE = 0.25

# PRESET: Full Lap (all segments, high quality)
# MAX_SEGMENTS = 20
# POPULATION_SIZE = 12
# NUM_GENERATIONS_INITIAL = 10
# NUM_GENERATIONS_MAX = 100
# MUTATION_RATE = 0.20
