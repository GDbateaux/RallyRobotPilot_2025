#!/bin/bash

# Create genetics_algo directory
mkdir -p genetics_algo

# Create all Python files with minimal structure
cat > genetics_algo/config.py << 'EOF'
"""
Configuration parameters for genetic algorithm training.
All hyperparameters and paths centralized here for easy modification.
"""

# Data paths
RECORDED_LAP_PATH = "old_data.bak"

# Track configuration
NUM_CHECKPOINTS = 5
CHECKPOINT_WIDTH = 10.0
SEGMENT_OVERLAP_FRAMES = 10

# GA hyperparameters
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
MUTATION_RATE = 0.03
ELITE_PERCENTAGE = 0.05

# Simulation parameters
MAX_TIMEOUT_FRAMES = 500
DT = 0.1  # Time step (10 FPS)

# Game connection
GAME_HOST = "127.0.0.1"
GAME_PORT = 5000
EOF

cat > genetics_algo/lap_data.py << 'EOF'
"""
Load and provide access to recorded lap data.
Handles all data extraction from recorded lap file.
"""

import pickle
import lzma
from config import RECORDED_LAP_PATH, NUM_CHECKPOINTS, CHECKPOINT_WIDTH


def load_recorded_lap():
    """
    Load the recorded lap data from file.

    Returns:
        list: List of SensingSnapshot objects
    """
    pass


def get_total_frames():
    """
    Get total number of frames in recorded lap.

    Returns:
        int: Total frame count
    """
    pass


def get_checkpoint_crossing_frames():
    """
    Calculate which frames the car crosses each checkpoint.

    Returns:
        list: Frame indices where checkpoints are crossed
    """
    pass


def get_segment_info(segment_id):
    """
    Get information about a specific segment.

    Args:
        segment_id (int): Segment identifier (0 to NUM_CHECKPOINTS-1)

    Returns:
        dict: {
            'start_frame': int,
            'end_frame': int,
            'start_state': dict,
            'target_checkpoint': dict
        }
    """
    pass


def get_recorded_controls(segment_id):
    """
    Extract control sequence for a specific segment.

    Args:
        segment_id (int): Segment identifier

    Returns:
        list: List of [forward/back, left/right] control pairs
    """
    pass
EOF

cat > genetics_algo/checkpoint.py << 'EOF'
"""
Checkpoint geometry and crossing detection logic.
Handles all checkpoint-related calculations.
"""

import numpy as np


def calculate_checkpoint_line(position, angle, width):
    """
    Calculate checkpoint line endpoints from car position and orientation.

    Args:
        position (tuple): (x, y, z) car position
        angle (float): Car angle in degrees
        width (float): Checkpoint width

    Returns:
        dict: {
            'center': (x, y, z),
            'point_a': (x, y, z),
            'point_b': (x, y, z),
            'width': float,
            'orientation': float
        }
    """
    pass


def check_line_crossing(pos1, pos2, checkpoint_line):
    """
    Check if movement from pos1 to pos2 crosses the checkpoint line.
    Uses 2D line intersection in X-Z plane (ignoring Y).

    Args:
        pos1 (tuple): Starting position (x, y, z)
        pos2 (tuple): Ending position (x, y, z)
        checkpoint_line (dict): Checkpoint definition

    Returns:
        bool: True if line was crossed
    """
    pass


def get_distance_to_checkpoint(position, checkpoint_line):
    """
    Calculate perpendicular distance from position to checkpoint line.

    Args:
        position (tuple): (x, y, z)
        checkpoint_line (dict): Checkpoint definition

    Returns:
        float: Distance to checkpoint
    """
    pass
EOF

cat > genetics_algo/individual.py << 'EOF'
"""
Individual representation for genetic algorithm.
Each individual is a sequence of controls (genome).
"""

import random
import copy


class Individual:
    """
    Represents one solution candidate (control sequence).
    """

    def __init__(self, genome):
        """
        Args:
            genome (list): List of [bool, bool] control pairs
        """
        self.genome = genome
        self.fitness = None

    @classmethod
    def random(cls, length):
        """
        Create individual with random controls.

        Args:
            length (int): Number of control frames

        Returns:
            Individual: New random individual
        """
        pass

    @classmethod
    def from_controls(cls, controls):
        """
        Create individual from existing control sequence.

        Args:
            controls (list): List of control pairs

        Returns:
            Individual: New individual with given controls
        """
        pass

    def mutate(self, rate):
        """
        Randomly flip control bits based on mutation rate.

        Args:
            rate (float): Probability of flipping each bit
        """
        pass

    def copy(self):
        """
        Create deep copy of this individual.

        Returns:
            Individual: Copy of this individual
        """
        pass
EOF

cat > genetics_algo/population.py << 'EOF'
"""
Population management and genetic operators.
Handles selection, crossover, and population evolution.
"""

import random
from individual import Individual


class Population:
    """
    Manages a population of individuals and evolution operations.
    """

    def __init__(self, individuals):
        """
        Args:
            individuals (list): List of Individual objects
        """
        self.individuals = individuals

    def sort_by_fitness(self):
        """
        Sort population by fitness (descending - best first).
        """
        pass

    def get_elite(self, percentage):
        """
        Get top percentage of individuals.

        Args:
            percentage (float): Fraction of population to keep (0.0 to 1.0)

        Returns:
            list: Elite individuals
        """
        pass

    def tournament_selection(self, k=3):
        """
        Select one individual using tournament selection.

        Args:
            k (int): Tournament size

        Returns:
            Individual: Selected individual
        """
        pass

    @staticmethod
    def crossover(parent1, parent2):
        """
        Create offspring by combining two parents.

        Args:
            parent1 (Individual): First parent
            parent2 (Individual): Second parent

        Returns:
            Individual: Offspring
        """
        pass

    def create_initial_population(recorded_controls, pop_size, elite_copies=5):
        """
        Create initial population seeded from recorded run.

        Args:
            recorded_controls (list): Controls from recorded lap
            pop_size (int): Population size
            elite_copies (int): Number of exact copies of recorded run

        Returns:
            Population: Initial population
        """
        pass
EOF

cat > genetics_algo/game_interface.py << 'EOF'
"""
Interface for communicating with the game.
Handles all game state manipulation and control sending.
"""

import requests
from config import GAME_HOST, GAME_PORT


def reset_car(position, angle, speed):
    """
    Reset car to specific state.

    Args:
        position (tuple): (x, y, z)
        angle (float): Rotation in degrees
        speed (float): Initial speed
    """
    pass


def send_controls(controls):
    """
    Send control sequence to game.

    Args:
        controls (list): List of [forward/back, left/right] pairs
    """
    pass


def get_car_state():
    """
    Get current car state from game.

    Returns:
        dict: {
            'position': (x, y, z),
            'speed': float,
            'angle': float
        }
    """
    pass


def send_command(command):
    """
    Send raw command string to game.

    Args:
        command (str): Command string (e.g., "push forward;")

    Returns:
        bool: Success status
    """
    pass
EOF

cat > genetics_algo/simulator.py << 'EOF'
"""
Simulation execution and result collection.
Runs individual through game and returns performance data.
"""

from game_interface import reset_car, send_controls, get_car_state
from checkpoint import check_line_crossing
from config import MAX_TIMEOUT_FRAMES


def simulate_individual(individual, start_state, target_checkpoint, max_frames=None):
    """
    Execute individual's controls in game and measure performance.

    Args:
        individual (Individual): Individual to evaluate
        start_state (dict): Initial car state
        target_checkpoint (dict): Target checkpoint definition
        max_frames (int): Maximum frames before timeout

    Returns:
        dict: {
            'checkpoint_crossed': bool,
            'frames_taken': int,
            'final_position': tuple,
            'crashed': bool
        }
    """
    pass


def detect_crash(speed, position, previous_position):
    """
    Detect if car has crashed based on state changes.

    Args:
        speed (float): Current speed
        position (tuple): Current position
        previous_position (tuple): Previous position

    Returns:
        bool: True if crash detected
    """
    pass
EOF

cat > genetics_algo/fitness.py << 'EOF'
"""
Fitness calculation from simulation results.
Determines how good an individual performed.
"""

from config import MAX_TIMEOUT_FRAMES


def calculate_fitness(simulation_result):
    """
    Calculate fitness score from simulation result.

    For POC:
    - If checkpoint crossed: High score based on speed (fewer frames = better)
    - If not crossed: Very negative score (effectively eliminated)

    Args:
        simulation_result (dict): Result from simulator

    Returns:
        float: Fitness score
    """
    pass
EOF

cat > genetics_algo/train_segment.py << 'EOF'
"""
Main training loop for one segment.
Orchestrates the entire GA process for a single segment.
"""

from config import POPULATION_SIZE, NUM_GENERATIONS, ELITE_PERCENTAGE, MUTATION_RATE
from lap_data import get_segment_info, get_recorded_controls
from population import Population
from simulator import simulate_individual
from fitness import calculate_fitness


def train_segment(segment_id):
    """
    Train GA on a specific segment.

    Args:
        segment_id (int): Which segment to train (0 to NUM_CHECKPOINTS-1)

    Returns:
        Individual: Best individual found
    """

    print(f"=== Training Segment {segment_id} ===")

    # TODO: Load segment data
    # TODO: Create initial population
    # TODO: Evolution loop
    # TODO: Return best individual

    pass


def evolution_loop(population, segment_info):
    """
    Run one generation of evolution.

    Args:
        population (Population): Current population
        segment_info (dict): Segment configuration

    Returns:
        Population: Next generation
    """
    pass


if __name__ == "__main__":
    # Train first segment as POC
    best_individual = train_segment(segment_id=0)
    print(f"Best fitness: {best_individual.fitness}")
EOF

cat > genetics_algo/utils.py << 'EOF'
"""
Utility functions and helpers.
"""


def format_time(seconds):
    """
    Format seconds into human-readable time.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    pass


def save_individual(individual, filepath):
    """
    Save individual to file.

    Args:
        individual (Individual): Individual to save
        filepath (str): Path to save file
    """
    pass


def load_individual(filepath):
    """
    Load individual from file.

    Args:
        filepath (str): Path to load from

    Returns:
        Individual: Loaded individual
    """
    pass
EOF

cat > genetics_algo/README.md << 'EOF'
# Genetic Algorithm Training for Rally Game

## Structure

- **config.py**: All configuration parameters
- **lap_data.py**: Load recorded lap data
- **checkpoint.py**: Checkpoint geometry and detection
- **individual.py**: Individual (genome) representation
- **population.py**: Population and genetic operators
- **game_interface.py**: Communication with game
- **simulator.py**: Execute simulations
- **fitness.py**: Fitness calculation
- **train_segment.py**: Main training loop
- **utils.py**: Helper functions

## Usage
```bash
python train_segment.py
```

## Development Plan

1. Implement lap_data.py (load and parse recorded data)
2. Implement checkpoint.py (copy from notebook)
3. Implement individual.py (basic genome operations)
4. Implement game_interface.py (HTTP communication)
5. Implement simulator.py (run one individual)
6. Implement fitness.py (simple scoring)
7. Implement population.py (genetic operators)
8. Implement train_segment.py (tie it all together)
9. Test and iterate

## Notes

- Start with segment 0 (first checkpoint) as POC
- All files are minimal - flesh out later
- Focus on clean structure over features
EOF

# Create __init__.py to make it a package
touch genetics_algo/__init__.py

echo "✓ Created genetics_algo/ directory structure"
echo "✓ All files created with minimal structure"
echo ""
echo "Next steps:"
echo "1. Review the structure in genetics_algo/"
echo "2. Decide which file to implement first"
echo "3. Implement one file at a time, test, then move to next"