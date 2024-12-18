# -*- coding: utf-8 -*-
import numpy as np

# Fonction objectif
def fobj(M, P, tol=1e-14):
    singular_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)
    nonzero_singular = singular_values[singular_values > tol]
    return len(nonzero_singular), nonzero_singular[-1] if len(nonzero_singular) > 0 else 0


# Comparaison de deux motifs
def compareP1betterthanP2(M, P1, P2):
    r1, s1 = fobj(M, P1)
    r2, s2 = fobj(M, P2)
    return (r1 < r2) or (r1 == r2 and s1 < s2)


# Génération de nouveaux motifs
def generate_new_pattern(P):
    new_P = P.copy()
    num_changes = np.random.randint(1, np.prod(P.shape) // 4)
    indices = np.random.randint(0, P.size, num_changes)
    new_P.flat[indices] *= -1
   
    return new_P


# Métaheuristique pour optimiser le motif
def metaheuristic(M, initial_temperature=100, cooling_rate=0.9, iterations_per_palier=100, final_temperature=1e-3):
    #P = construction_heuristique(M)
    P = np.ones(M.shape)
    current_rank, current_singular = fobj(M, P)
    best_P, best_rank, best_singular = P, current_rank, current_singular
    temperature = initial_temperature
    rank_history = [best_rank]
    singular_history = [best_singular]

    while temperature > final_temperature:
        for _ in range(iterations_per_palier):
            new_P = generate_new_pattern(P)
            new_rank, new_singular = fobj(M, new_P)
            
            # Critère d'acceptation
            if compareP1betterthanP2(M, new_P, P) or \
                    np.random.rand() < np.exp(-abs(new_rank - current_rank) / temperature):
                P, current_rank, current_singular = new_P, new_rank, new_singular
                
                
                # Mise à jour du meilleur motif
                if compareP1betterthanP2(M, P, best_P):
                    best_P, best_rank, best_singular = P, current_rank, current_singular
                    rank_history.append(best_rank)
                    singular_history.append(best_singular)

        temperature *= cooling_rate


    return best_P, best_rank, singular_history

