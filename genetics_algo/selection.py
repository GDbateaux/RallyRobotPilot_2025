import random
import copy
from genetics_algo.config import ELITE_PERCENTAGE, TOURNAMENT_SIZE


def select_elite(population, elite_percentage=None):
    if elite_percentage is None:
        elite_percentage = ELITE_PERCENTAGE

    elite_count = max(1, int(len(population) * elite_percentage))

    # Deep copy elite individuals to prevent mutation
    elite = [copy.deepcopy(ind) for ind in population[:elite_count]]

    return elite


def tournament_selection(population, tournament_size=None):
    if tournament_size is None:
        tournament_size = TOURNAMENT_SIZE

    # Ensure tournament size doesn't exceed population
    tournament_size = min(tournament_size, len(population))

    # Sample WITH replacement (same individual can appear multiple times)
    contestants = random.sample(population, tournament_size)

    # Select best by fitness
    winner = max(contestants, key=lambda ind: ind['fitness'])

    return winner


def select_parents(population, num_parents, tournament_size=None):
    parents = []
    for _ in range(num_parents):
        parent = tournament_selection(population, tournament_size)
        parents.append(parent)

    return parents


def sort_population_by_fitness(population):
    return sorted(population, key=lambda ind: ind['fitness'], reverse=True)