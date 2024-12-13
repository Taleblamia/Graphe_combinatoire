# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant
import grasp
import time


#%%


# Matrice de distances euclidiennes
def matrices1_ledm(n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (i - j) ** 2
    return M


# Matrice slackngon
def matrices2_slackngon(n):
    M = circulant(np.cos(np.pi / n) - np.cos(np.pi / n + 2 * np.pi * np.arange(0, n) / n))
    M /= M[0, 2]
    M = np.maximum(M, 0)
    np.fill_diagonal(M, 0)
    M[np.arange(n - 1), np.arange(1, n)] = 0
    M[n - 1, 0] = 0
    return M


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
    #new_P = P.copy()
    # num_changes = np.random.randint(1, np.prod(P.shape) // 4)
    # indices = np.random.randint(0, P.size, num_changes)
    # new_P.flat[indices] *= -1
    indice =np.random.randint(0, P.shape[1])
    new_P= grasp.graspP(M,P,indice,num_candidats=1000)
    return new_P


# Construction heuristique initiale
def construction_heuristique(M):
    m, n = M.shape
    P = -np.ones((m, n))
    for i in range(m):
        P[i, i] = 1
        if i*2 < n:
            P[i, i + 1] = 1
    return P

# Métaheuristique pour optimiser le motif
def metaheuristic(M, initial_temperature=100, cooling_rate=0.9, iterations_per_palier=100, final_temperature=1e-3):
    #P = construction_heuristique(M)
    P = grasp.grasp(M,1000)
    current_rank, current_singular = fobj(M, P)
    best_P, best_rank, best_singular = P, current_rank, current_singular
    print(best_rank)
    temperature = initial_temperature
    rank_history = [best_rank]
    singular_history = [best_singular]

    while temperature > final_temperature:
        for _ in range(iterations_per_palier):
            new_P = generate_new_pattern(P)
            new_rank, new_singular = fobj(M, new_P)
            #print(new_rank)
            # Critère d'acceptation
            if compareP1betterthanP2(M, new_P, P) or \
                    np.random.rand() < np.exp(-abs(new_rank - current_rank) / temperature):
                P, current_rank, current_singular = new_P, new_rank, new_singular
                print(current_rank)
                
                # Mise à jour du meilleur motif
                if compareP1betterthanP2(M, P, best_P):
                    best_P, best_rank, best_singular = P, current_rank, current_singular
                    rank_history.append(best_rank)
                    singular_history.append(best_singular)

        temperature *= cooling_rate
    
    
    # Affichage des résultats
    # plt.figure(figsize=(10, 6))
    # plt.plot(rank_history, label="Rang")
    # #lt.plot(singular_history, label="Plus petite valeur singulière")
    # plt.title("Évolution des métriques en fonction des itérations")
    # plt.xlabel("Étapes")
    # plt.ylabel("Valeurs")
    # plt.legend()
    # plt.grid()
    # plt.show()

    return best_P, best_rank, best_singular


# Lecture d'une matrice depuis un fichier
def lecture_fichier(path):
    with open(path, 'r') as fin:
        m, n = map(int, fin.readline().split())
        data = [list(map(float, fin.readline().split())) for _ in range(m)]
    return np.array(data)


# Exemple d'utilisation
if __name__ == "__main__":
    M = lecture_fichier("correl5_matrice.txt")
    best_P, best_rank, best_singular = metaheuristic(M)
    print("Meilleur motif trouvé :\n", best_P)
    print("Rang minimal :", best_rank)
    print("Plus petite valeur singulière non-nulle :", best_singular)
    start_time = time.time()
    end_time = time.time()

    # Calculer la durée
    elapsed_time = end_time - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    # Afficher les résultats
    print("\nTemps d'exécution : {:02d}h {:02d}m {:.2f}s".format(hours, minutes, seconds))
    #%%
# start_time = time.time()
# P = grasp.grasp(M,100000)
# print(fobj(M,P))
# print(time.time()-start_time)
