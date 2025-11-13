"""
Evolution pipeline for genetic algorithm.
Combines selection, mutation, and population management.
"""

from genetics_algo.selection import select_elite, select_parents, sort_population_by_fitness
from genetics_algo.mutation import create_children
from genetics_algo.config import ELITE_PERCENTAGE, TOURNAMENT_SIZE, MUTATION_WINDOW_SIZE, MUTATION_RATE_NO_CRASH


def augment_population_with_results(population, simulation_results, fitness_results):
    """
    Augment population with simulation and fitness results.
    Adds collision info and fitness to each individual.

    Args:
        population (list): List of individuals
        simulation_results (list): List of simulation results (same order)
        fitness_results (list): List of fitness results (same order)

    Returns:
        list: Population with added fields (modifies in place)
    """
    for ind, sim_result, fit_result in zip(population, simulation_results, fitness_results):
        ind['collision_detected'] = sim_result['collision_detected']
        ind['collision_frame'] = sim_result.get('collision_frame')
        ind['fitness'] = fit_result['fitness']
        ind['checkpoints_crossed'] = fit_result['checkpoints_crossed']
        ind['checkpoints_crossed_count'] = fit_result['checkpoints_crossed_count']

    return population


def create_next_generation(population,
                          elite_percentage=None,
                          tournament_size=None,
                          mutation_window_size=None,
                          mutation_rate_no_crash=None):
    """
    Create next generation via elitism + tournament selection + mutation.

    Pipeline:
    1. Sort population by fitness
    2. Select elite (top percentage)
    3. Select parents via tournament selection
    4. Create children via mutation
    5. Combine elite + children

    Args:
        population (list): Current population with 'fitness' and collision info
        elite_percentage (float): Percentage of elite to keep (default from config)
        tournament_size (int): Tournament size (default from config)
        mutation_window_size (int): Mutation window (default from config)
        mutation_rate_no_crash (float): Mutation rate for non-crashed (default from config)

    Returns:
        list: Next generation population (same size as input)
    """
    if elite_percentage is None:
        elite_percentage = ELITE_PERCENTAGE
    if tournament_size is None:
        tournament_size = TOURNAMENT_SIZE
    if mutation_window_size is None:
        mutation_window_size = MUTATION_WINDOW_SIZE
    if mutation_rate_no_crash is None:
        mutation_rate_no_crash = MUTATION_RATE_NO_CRASH

    population_size = len(population)

    # Step 1: Sort population by fitness (best first)
    sorted_population = sort_population_by_fitness(population)

    # Step 2: Select elite
    elite = select_elite(sorted_population, elite_percentage)
    elite_count = len(elite)

    # Step 3: Determine how many children to create
    num_children = population_size - elite_count

    # Step 4: Select parents via tournament selection
    parents = select_parents(sorted_population, num_children, tournament_size)

    # Step 5: Create children via mutation
    children = create_children(parents, mutation_window_size, mutation_rate_no_crash)

    # Step 6: Combine elite + children
    next_generation = elite + children

    return next_generation


def get_best_individual(population):
    """
    Get the best individual from population.

    Args:
        population (list): List of individuals with 'fitness' key

    Returns:
        dict: Best individual
    """
    return max(population, key=lambda ind: ind['fitness'])


def get_population_stats(population):
    """
    Calculate population statistics.

    Args:
        population (list): List of individuals with 'fitness' key

    Returns:
        dict: Statistics (best_fitness, worst_fitness, avg_fitness, etc.)
    """
    fitness_scores = [ind['fitness'] for ind in population]

    stats = {
        'best_fitness': max(fitness_scores),
        'worst_fitness': min(fitness_scores),
        'avg_fitness': sum(fitness_scores) / len(fitness_scores),
        'best_individual': max(population, key=lambda ind: ind['fitness']),
        'collisions': sum(1 for ind in population if ind.get('collision_detected', False)),
        'max_checkpoints': max(ind.get('checkpoints_crossed_count', 0) for ind in population)
    }

    return stats