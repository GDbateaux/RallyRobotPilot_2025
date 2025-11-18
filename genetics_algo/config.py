# ============================================================================
# TRACK CONFIGURATION
# ============================================================================
# Track name (corresponds to directory in assets/)
# Options: "SimpleTrack", "NotSoSimpleTrack", "SlightlyHarder", etc.
TRACK_NAME = "SlightlyHarder"

# Path to recorded lap data
RECORDED_LAP_PATH = "runs/slightlyharder_recorded_1.npz"

# Track geometry
NUM_CHECKPOINTS = 40          # Total checkpoints in the track (for lap subdivision - DO NOT CHANGE)
                              # This determines how the recorded lap is subdivided into checkpoints
FINISH_LINE_CHECKPOINT_ID = NUM_CHECKPOINTS - 1   # Checkpoint ID for the finish line
                              # Note: Checkpoint 29 was removed (too close to start), so finish line is renumbered from 30→29
CHECKPOINT_WIDTH = 50.0       # Width of checkpoint lines (meters)

# Checkpoint removal configuration
CHECKPOINTS_TO_REMOVE = [9, 10]    # List of checkpoint IDs to remove from the track
                              # Example: [17, 30, 39] removes checkpoints 17, 30, and 39
                              # Remaining checkpoints keep their physical positions but are renumbered sequentially
                              # [] = No checkpoints removed (default)

# ============================================================================
# GENOME SETTINGS
# ============================================================================
GENOME_LENGTH_MULTIPLIER = 1.0  # Genome length = multiplier * recorded_lap_length
                              # Gives individuals extra time to complete the lap

# Bootstrap from existing solution
START_AT = None               # Frame number to start random generation from
                              # None = Generate entire genome randomly (default)
                              # 30 = Use latest ga_runs/*.npz genome for frames 0-30, then random
                              # This allows bootstrapping from a known good partial solution

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
SPEED_PENALTY_FACTOR = 2.0    # Multiplier for speed penalty
                              # Penalty = frames_to_last * factor
                              # Lower frames = lower penalty = reward for speed
                              # Try values: 1.0 (gentle), 2.0 (moderate), 3.0 (strong)

FINISH_LINE_SPEED_BONUS_FACTOR = 10.0  # Bonus for completing lap faster
                              # Bonus = (genome_size - finish_frame) * factor
                              # Faster lap = more frames saved = higher bonus

ENABLE_SURVIVAL_BONUS = False # Enable/disable survival points bonus
                              # True: Rewards longer runs (can favor slow individuals)
                              # False: Focus purely on checkpoints + speed (better for racing)

# Progress bonus (distance traveled AFTER last checkpoint)
PROGRESS_BONUS_PER_DISTANCE = 10.0  # Points per unit of distance traveled after crossing last checkpoint
                              # This rewards actual forward progress after checkpoints, not just survival time
                              # Prevents cars from gaming the system by slowing down or circling
                              # 0.0 = Disabled (no progress bonus)
                              # 0.5 = Gentle (slight exploration incentive)
                              # 1.0 = Moderate (balanced - recommended starting point)
                              # 2.0 = Strong (heavily favors distance exploration)
                              # Distance is 3D Euclidean from checkpoint position to final position

# Finish line validation
FINISH_LINE_MIN_FRAME = 50    # Minimum frame before finish line can be crossed
                              # Prevents immediate validation at start

# ============================================================================
# POPULATION SETTINGS
# ============================================================================
POPULATION_SIZE = 100          # Number of individuals per generation
                              # Small (3-5): Fast, good for testing
                              # Medium (8-12): Balanced
                              # Large (15-20): Slower but more thorough

# ============================================================================
# GENERATION SETTINGS
# ============================================================================
NUM_GENERATIONS_INITIAL = 40   # Initial generations to try
                              # Training stops early if solution found

EXTRA_GENERATIONS_AFTER_FINISH = 5  # Continue for N generations after finish line crossed
                              # Allows optimization after completing the lap
                              # Set to 0 to stop immediately when finish line reached

# ============================================================================
# EVOLUTION STRATEGY
# ============================================================================
# Selection
ELITE_PERCENTAGE = 0.10       # Percentage of population to keep as elite
                              # These individuals go UNCHANGED to next generation
                              # 0.10 = Top 10% preserved
                              # 0.20 = Top 20% preserved

ELITE_OFFSPRING_MULTIPLIER = 6  # Number of mutated children each elite produces
                              # These are GUARANTEED offspring from top performers
                              # Example: 5 elite × 4 offspring = 20 children from elite
                              # 0 = Disabled (no guaranteed offspring)
                              # 2-3 = Moderate exploitation
                              # 4-5 = Strong exploitation (recommended for breakthrough detection)
                              # 6+ = Very strong (may reduce diversity)

TOURNAMENT_SIZE = 8          # Number of individuals in each tournament
                              # Used to select parents for remaining population slots
                              # 3-5: Balanced selection pressure
                              # 7-10: Strong selection (favors best)
                              # 12-15: Very strong (highly favors best individuals)

# Mutation
MUTATION_WINDOW_SIZE = 20    # Frames before crash to mutate
                              # Mutation happens in [crash_frame - window, crash_frame]
                              # Smaller = more focused, Larger = more exploration

MUTATION_DISTRIBUTION_TYPE = "gaussian"  # Distribution type for mutations
                              # "power" = 1/distance^power (biased toward collision point)
                              # "gaussian" = Normal distribution centered before collision (RECOMMENDED)

# Power distribution parameters (used when MUTATION_DISTRIBUTION_TYPE = "power")
MUTATION_DISTANCE_POWER = 0.5  # Power for distance-based mutation probability (1/distance^power)
                              # Controls how mutations are distributed within the window
                              # 0.0 = Uniform distribution (equal probability everywhere)
                              # 0.3 = Very flat (lots of exploration far from collision)
                              # 0.5 = Square root (balanced - RECOMMENDED)
                              # 0.7 = Moderate focusing near collision
                              # 1.0 = Linear (original, very steep - most mutations right before crash)
                              # Higher = more focus near collision, Lower = more spread out
                              # Example with collision at frame 150, window=20:
                              #   power=1.0: 95% mutations in frames 145-150 (very focused)
                              #   power=0.5: 60% mutations in frames 145-150 (balanced)
                              #   power=0.3: 40% mutations in frames 145-150 (exploratory)

# Gaussian distribution parameters (used when MUTATION_DISTRIBUTION_TYPE = "gaussian")
MUTATION_CENTER_OFFSET = 9    # Frames before collision to center the mutation distribution
                              # 8 = Mutations peak at 8 frames before collision (RECOMMENDED)
                              # This is typically when corrective actions should start
                              # Higher = earlier corrections, Lower = later corrections

MUTATION_GAUSSIAN_STD = 4.0   # Standard deviation for gaussian distribution (frames)
                              # Controls spread of mutations around the center
                              # 2.0 = Very focused (68% within ±2 frames, 95% within ±4 frames)
                              # 3.0 = Balanced (68% within ±3 frames, 95% within ±6 frames)
                              # 3.5 = Moderate spread (68% within ±3.5 frames, 95% within ±7 frames) - RECOMMENDED
                              # 4.0 = Wide (68% within ±4 frames, 95% within ±8 frames)
                              # 5.0 = Very wide (68% within ±5 frames, 95% within ±10 frames)

MUTATION_RATE_NO_CRASH = 0.2  # Probability to mutate individuals that didn't crash
                              # 0.0 = Clone them unchanged
                              # 0.1 = 10% chance to muta<zte anyway

MUTATION_DURATION_MIN = 3     # Minimum consecutive frames to mutate
                              # Prevents tiny single-frame changes
                              # 5 = At least 5 frames affected (sustained behaviors)

MUTATION_DURATION_MAX = 20    # Maximum consecutive frames to mutate
                              # Random duration between MIN and MAX
                              # Example: MIN=5, MAX=20 → durations of 5-20 frames
                              # Higher = more sustained behaviors (better for turns and straights)
                              # Lower = more fine-grained control

BRAKE_MUTATION_DURATION_MAX = 8  # Maximum frames for BRAKE mutations specifically
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

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
SEGMENT_OVERLAP_FRAMES = 2    # Frames to overlap between segments
                              # Example: Segment 0 ends at frame 24
                              #          Segment 1 starts at frame 22 (24 - 2)
                              # This creates a 2-frame overlap for smoother transitions

SEGMENT_TIMEOUT_BUFFER = 1   # Extra frames beyond segment length
                              # Example: If segment is 24 frames, timeout = 24 + 10 = 34 frames
                              # This gives individuals some exploration time beyond the
                              # recorded optimal path (useful for trying different strategies)

# Collision detection safety buffer
COLLISION_SAFETY_BUFFER = 0.0  # Extra distance for collision detection (units)
                              # Adds safety margin to prevent rare missed collisions
                              # 0.0 = exact match to real game (but may miss ~0.08% of collisions)
                              # 0.05 = 5cm buffer (minimal, catches most misses)
                              # 0.1 = 10cm buffer (recommended, very safe)
                              # 0.2 = 20cm buffer (very conservative)
                              # Higher = safer but GA learns more cautious paths

# Car width for multi-ray collision detection
CAR_WIDTH = 2.0              # Width of car for side collision detection (units)
                              # Used to offset left/right raycasts during turns
                              # Half-width offset on each side (1.0 left, 1.0 right)
                              # 1.5 = Narrow (tight cornering, may clip walls in real game)
                              # 2.0 = Standard (recommended, matches typical car width)
                              # 2.5 = Wide (very safe, more conservative paths)
                              # Higher = GA learns to give walls more clearance

# Physics sub-stepping for collision detection
PHYSICS_SUBSTEPS = 2          # Number of physics updates per frame
                              # Prevents high-speed tunneling through thin walls
                              # 1 = No sub-stepping (can tunnel at max speed, fastest)
                              # 2 = 2 checks per frame (catches walls > 2.5 units, 2× slower)
                              # 3 = 3 checks per frame (catches walls > 1.67 units, 3× slower) ← RECOMMENDED
                              # 4 = 4 checks per frame (catches walls > 1.25 units, 4× slower, very safe)
                              # Higher = more accurate but slower simulation

# Proximity collision detection (prevents getting too close to obstacles)
MIN_SAFE_DISTANCE = 2.5       # Minimum safe distance from obstacles (units)
                              # If any proximity sensor detects obstacle closer than this, treat as collision
                              # 2.0 = Very tight (risky, may graze walls)
                              # 3.0 = Standard (recommended, good safety margin)
                              # 4.0 = Conservative (wide berth, safer paths)
                              # 5.0 = Very safe (may limit optimal racing lines)

NUM_PROXIMITY_RAYS = 20       # Number of proximity sensor rays to cast
                              # Rays are spread in an arc in front of the car
                              # 15 = Standard (good coverage, matches original game)

PROXIMITY_HALF_ANGLE = 100     # Half angle for proximity sensor coverage (degrees)
                              # Total coverage = 2 * half_angle
                              # 90 = 180° total coverage (front hemisphere)
                              # 60 = 120° total coverage (narrower, front-focused)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
SAVE_ALL_PLOTS = True        # Save plot for every generation (True) or only final (False)
                              # True: Creates plots/track_gen1.png, track_gen2.png, ... track_genN.png
                              # False: Only creates final plot after all generations

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
NUM_WORKERS = 20            # Number of parallel workers for simulation
                              # Set to number of CPU cores available
                              # 1 = Sequential (no parallelization)
                              # 20 = Use 20 cores in parallel
                              # Set to -1 to use all available cores
