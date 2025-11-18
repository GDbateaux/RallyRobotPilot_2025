
from genetics_algo.selection import select_elite, select_parents, sort_population_by_fitness
from genetics_algo.mutation import create_children
from genetics_algo.config import (
    ELITE_PERCENTAGE, ELITE_OFFSPRING_MULTIPLIER, TOURNAMENT_SIZE,
    MUTATION_WINDOW_SIZE, MUTATION_RATE_NO_CRASH
)


def augment_population_with_results(population, simulation_results, fitness_results):
    for ind, sim_result, fit_result in zip(population, simulation_results, fitness_results):
        ind['collision_detected'] = sim_result['collision_detected']
        ind['collision_frame'] = sim_result.get('collision_frame')
        ind['fitness'] = fit_result['fitness']
        ind['checkpoints_crossed'] = fit_result['checkpoints_crossed']
        ind['checkpoints_crossed_count'] = fit_result['checkpoints_crossed_count']

    return population


def create_next_generation(population,
                          elite_percentage=None,
                          elite_offspring_multiplier=None,
                          tournament_size=None,
                          mutation_window_size=None,
                          mutation_rate_no_crash=None):

    if elite_percentage is None:
        elite_percentage = ELITE_PERCENTAGE
    if elite_offspring_multiplier is None:
        elite_offspring_multiplier = ELITE_OFFSPRING_MULTIPLIER
    if tournament_size is None:
        tournament_size = TOURNAMENT_SIZE
    if mutation_window_size is None:
        mutation_window_size = MUTATION_WINDOW_SIZE
    if mutation_rate_no_crash is None:
        mutation_rate_no_crash = MUTATION_RATE_NO_CRASH

    population_size = len(population)

    # Step 1: Sort population by fitness (best first)
    sorted_population = sort_population_by_fitness(population)

    # Step 2: Select elite (unchanged copies)
    elite = select_elite(sorted_population, elite_percentage)
    elite_count = len(elite)

    # Step 3: Create guaranteed offspring from elite
    elite_offspring = []
    if elite_offspring_multiplier > 0:
        for _ in range(elite_offspring_multiplier):
            # Each elite individual creates one child per iteration
            elite_children = create_children(elite, mutation_window_size, mutation_rate_no_crash)
            elite_offspring.extend(elite_children)

    # Step 4: Calculate remaining slots to fill via tournament
    slots_filled = elite_count + len(elite_offspring)
    remaining_slots = population_size - slots_filled

    # Step 5: Fill remaining slots via tournament selection
    tournament_children = []
    if remaining_slots > 0:
        parents = select_parents(sorted_population, remaining_slots, tournament_size)
        tournament_children = create_children(parents, mutation_window_size, mutation_rate_no_crash)

    # Step 6: Combine all three groups
    next_generation = elite + elite_offspring + tournament_children

    return next_generation


def get_population_stats(population):
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