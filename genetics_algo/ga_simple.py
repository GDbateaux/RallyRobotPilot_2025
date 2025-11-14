"""
Simple Genetic Algorithm Core - Built from scratch.
Clean implementation for learning and understanding.
"""

from genetics_algo.individual import generate_random_genome
from genetics_algo.config import GENOME_LENGTH_MULTIPLIER, POPULATION_SIZE


def calculate_genome_length(recorded_lap_data):
    """
    Calculate genome length based on recorded lap.
    Genome length = GENOME_LENGTH_MULTIPLIER * len(recorded_lap_data)

    Args:
        recorded_lap_data (list): List of snapshots from recorded lap

    Returns:
        int: Length of genome (number of control tuples)
    """
    recorded_length = len(recorded_lap_data)
    genome_length = GENOME_LENGTH_MULTIPLIER * recorded_length
    return genome_length


def create_individual(genome_length):
    """
    Create one random individual.

    Args:
        genome_length (int): Number of control tuples in genome

    Returns:
        dict: Individual with 'genome' key containing list of control tuples
    """
    genome = generate_random_genome(genome_length)
    return {
        'genome': genome,
        'fitness': None  # Will be calculated later
    }


def create_initial_population(genome_length, population_size):
    """
    Create initial population of random individuals.

    Args:
        genome_length (int): Number of control tuples in each genome
        population_size (int): Number of individuals in population

    Returns:
        list: List of individuals (dicts with 'genome' and 'fitness')
    """
    population = []
    for i in range(population_size):
        individual = create_individual(genome_length)
        population.append(individual)

    return population