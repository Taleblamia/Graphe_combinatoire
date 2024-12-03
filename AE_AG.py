# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:29:57 2024

@author: ouiam
"""

import numpy as np
import random


# Fonction pour évaluer un individu
def fobj(M, P, tol=1e-14):
    """Calcule les objectifs pour une matrice donnée."""
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)
    ind_nonzero = np.where(sing_values > tol)[0]
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]


def compare_individuals(M, P1, P2):
    """Compare deux individus de manière lexicographique."""
    r1, s1 = fobj(M, P1)
    r2, s2 = fobj(M, P2)
    if r1 != r2:
        return r1 < r2  # Comparaison sur le rang
    return s1 < s2  # Comparaison sur la plus petite valeur singulière


def generate_random_pattern(M):
    """Génère un individu aléatoire."""
    return np.random.choice([-1, 1], size=M.shape)


def initialize_population(pop_size, M):
    """Initialise une population."""
    return [generate_random_pattern(M) for _ in range(pop_size)]


def tournament_selection(M, population, taille_selection):
    """Effectue une sélection par tournoi."""
    selected = []
    while len(selected) < taille_selection:
        candidates = random.sample(population, 2)
        best = candidates[0] if compare_individuals(M, candidates[0], candidates[1]) else candidates[1]
        selected.append(best)
    return selected


def crossover(parent1, parent2, crossover_rate):
    """Effectue un croisement entre deux parents."""
    if np.random.rand() < crossover_rate:
        mask = np.random.randint(0, 2, size=parent1.shape, dtype=bool)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
    return parent1.copy(), parent2.copy()


def mutate(individual, mutation_rate, mutation_fraction):
    """Applique une mutation à un individu."""
    num_mutations = int(mutation_fraction * individual.size)
    for _ in range(num_mutations):
        if np.random.rand() < mutation_rate:
            i, j = np.random.randint(0, individual.shape[0]), np.random.randint(0, individual.shape[1])
            individual[i, j] *= -1
    return individual


def genetic_algorithm(M, max_iter=500, population_size=500, taille_selection=250, 
                       crossover_rate=0.8, mutation_rate=0.1, mutation_fraction=0.05):
    """
    Algorithme évolutionnaire pour optimiser une matrice P.
    """
    # Initialisation
    population = initialize_population(population_size, M)
    best_pattern = None
    best_fitness = None

    for iteration in range(max_iter):
        # Évaluation de la population
        population = sorted(population, key=lambda P: fobj(M, P))
        current_best = population[0]
        
        # Mise à jour du meilleur individu
        if best_pattern is None or compare_individuals(M, current_best, best_pattern):
            best_pattern = current_best
            best_fitness = fobj(M, best_pattern)

        # Sélection
        parents = tournament_selection(M, population, taille_selection)

        # Croisement
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_generation.extend([child1, child2])

        # Mutation
        next_generation = [mutate(child, mutation_rate, mutation_fraction) for child in next_generation]

        # Nouvelle population
        population = next_generation[:population_size]

        # Affichage des progrès
        print(f"Iteration {iteration+1}/{max_iter}, Best Fitness: {best_fitness}")

    return best_pattern


def lecture_fichier(path):
    """Lit une matrice à partir d'un fichier."""
    with open(path, 'r') as fin:
        m, n = map(int, fin.readline().rstrip().split())
        data = [list(map(float, fin.readline().rstrip().split())) for _ in range(m)]
    return np.array(data)


# Exemple d'utilisation
if __name__ == "__main__":
    M = lecture_fichier("correl5_matrice.txt")
    best_pattern = genetic_algorithm(M)
    print(f"Meilleur individu trouvé: {fobj(M, best_pattern)}")
