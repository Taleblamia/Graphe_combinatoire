# -*- coding: utf-8 -*-
"""opti combi - projet square root rank.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZXNhyCQTIiKr94WOZSZOJ12rXZs8HsnT
"""

import numpy as np

def matrices1_ledm(n):
  M  = np.zeros((n,n))
  for i in range(n):
    for j in range(n):
      M[i,j]=(i-j)**2
  return M

from scipy.linalg import circulant
def matrices2_slackngon(n):
  M  = circulant(np.cos(np.pi/n)-np.cos(np.pi/n + 2*np.pi*np.arange(0,n,1)/n))
  M /= M[0,2]
  M  = np.maximum(M,0)
  for i in range(n):
    M[i,i] = 0
    if i<n-1:
      M[i,i+1] = 0
    else:
      M[i,0] = 0
  return M

def fobj(M,P,tol=1e-14):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False) # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                # indices des valeurs > tolérance donnée
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]       # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def metaheuristic(M):
  bestPattern = np.ones(M.shape) #pattern initial

  ... #votre méthode

  return bestPattern

M = np.array([[4,0,1],[1,1,1],[1,1,0]])
P1 = np.array([[1,1,-1],[-1,1,1],[1,-1,-1]])
P2 = np.array([[-1,1,-1],[-1,-1,1],[1,1,-1]])
print(compareP1betterthanP2(M,P1,P2))
print(np.linalg.svd(P1*np.sqrt(M), compute_uv=False))

M = matrices2_slackngon(7)
P = np.array([[1,1,1,1,1,-1,1],[1,1,1,-1,1,-1,1],[1,1,1,1,1,1,-1],[1,-1,1,1,1,-1,-1],[1,1,-1,1,1,1,1],[1,-1,1,-1,-1,1,1],[1,1,1,1,1,1,1]])
print(fobj(M,P))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from scipy.linalg import circulant

def matrices1_ledm(n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (i - j) ** 2
    return M

def matrices2_slackngon(n):
    M = circulant(np.cos(np.pi / n) - np.cos(np.pi / n + 2 * np.pi * np.arange(0, n, 1) / n))
    M /= M[0, 2]
    M = np.maximum(M, 0)
    for i in range(n):
        M[i, i] = 0
        if i < n - 1:
            M[i, i + 1] = 0
        else:
            M[i, 0] = 0
    return M

def fobj(M, P, tol=1e-14):
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)  # Valeurs singulières
    ind_nonzero = np.where(sing_values > tol)[0]                  # Indices > tolérance
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]         # Retourne rang et petite valeur singulière

def compareP1betterthanP2(M, P1, P2):
    r1, s1 = fobj(M, P1)  # Objectifs pour P1
    r2, s2 = fobj(M, P2)  # Objectifs pour P2
    if r1 != r2:
        return r1 < r2    # Priorité au rang
    return s1 < s2        # Ensuite à la plus petite valeur singulière

def generate_neighbors(P):
    """
    Génère des voisins en inversant les signes de certains éléments de P.
    """
    neighbors = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            neighbor = P.copy()
            neighbor[i, j] *= -1
            neighbors.append(neighbor)
    return neighbors

def metaheuristic(M, max_iter=100, tabu_size=10):
    """
    Métaheuristique basée sur l'algorithme Tabou pour résoudre le problème du square root rank.
    """
    bestPattern = np.ones(M.shape)  # Pattern initial
    tabu_list = []                 # Liste tabou
    best_objective = fobj(M, bestPattern)  # Objectif initial
    currentPattern = bestPattern.copy()

    for iteration in range(max_iter):
        neighbors = generate_neighbors(currentPattern)  # Génère des voisins
        best_neighbor = None
        best_neighbor_objective = None

        for neighbor in neighbors:
            if any(np.array_equal(neighbor, tabu) for tabu in tabu_list):
                continue  # Ignore les voisins dans la liste tabou

            neighbor_objective = fobj(M, neighbor)
            if best_neighbor is None or compareP1betterthanP2(M, neighbor, best_neighbor):
                best_neighbor = neighbor
                best_neighbor_objective = neighbor_objective

        if best_neighbor is None:
            break  # Aucun voisin valide trouvé

        currentPattern = best_neighbor

        # Mise à jour du meilleur résultat
        if compareP1betterthanP2(M, currentPattern, bestPattern):
            bestPattern = currentPattern
            best_objective = best_neighbor_objective

        # Mise à jour de la liste tabou
        tabu_list.append(currentPattern)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        print(f"Iteration {iteration + 1}: Best rank so far: {best_objective[0]}")

    return bestPattern

# Exemple d'utilisation
M = np.array([[4, 0, 1], [1, 1, 1], [1, 1, 0]])
M = matrices2_slackngon(7)
best_pattern = metaheuristic(M)
print("Best pattern found:")
print(best_pattern)
print(fobj(M, best_pattern))
#%%
import numpy as np

def matrices1_ledm(n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = (i - j) ** 2
    return M

from scipy.linalg import circulant
def matrices2_slackngon(n):
    M = circulant(np.cos(np.pi / n) - np.cos(np.pi / n + 2 * np.pi * np.arange(0, n, 1) / n))
    M /= M[0, 2]
    M = np.maximum(M, 0)
    for i in range(n):
        M[i, i] = 0
        if i < n - 1:
            M[i, i + 1] = 0
        else:
            M[i, 0] = 0
    return M

def fobj(M, P, tol=1e-14):
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)  # Calcul des valeurs singulières de la matrice P.*sqrt(M)
    ind_nonzero = np.where(sing_values > tol)[0]  # indices des valeurs > tolérance donnée
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]  # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M, P1, P2):
    r1, s1 = fobj(M, P1)  # on récupère les deux objectifs pour le pattern P1
    r2, s2 = fobj(M, P2)  # on récupère les deux objectifs pour le pattern P2
    if r1 != r2:  # on traite les objectifs de façon lexicographique :
        return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
    return s1 < s2  # alors on prend en compte la valeur de la plus petite valeur singulière

def construire_solution(M, alpha=0.5):
    m, n = M.shape  # m et n sont les dimensions de la matrice M
    candidats = []
    scores = []

    # Générer un ensemble de candidats
    for _ in range(100):  # Nombre de candidats à générer
        pattern = np.random.choice([1, -1], size=(m, n))  # Matrix de taille (m, n)
        rang, val = fobj(M, pattern)
        candidats.append(pattern)
        scores.append((rang, val))

    # Déterminer cmin et cmax
    rangs, valeurs = zip(*scores)
    cmin = min(rangs)
    cmax = max(rangs)
    seuil = cmin + alpha * (cmax - cmin)

    # Construire la LCR
    LCR = [c for c, (r, _) in zip(candidats, scores) if r <= seuil]

    # Vérifier que la LCR n'est pas vide
    if not LCR:
        raise ValueError("La liste des candidats restreinte (LCR) est vide. Vérifiez les paramètres de alpha.")
    
    # Sélectionner une matrice dans la LCR
    index = np.random.randint(len(LCR))  # Générer un indice aléatoire
    return LCR[index]

# Recherche locale (simple inversion des coefficients pour améliorer)
def recherche_locale(M, P):
    m, n = P.shape
    meilleur_P = P.copy()
    meilleur_rang, meilleure_val = fobj(M, meilleur_P)

    # Tester les voisins (une inversion à la fois)
    for i in range(m):
        for j in range(n):
            voisin = meilleur_P.copy()
            voisin[i, j] *= -1  # Inversion du coefficient
            rang, val = fobj(M, voisin)
            if rang < meilleur_rang or (rang == meilleur_rang and val < meilleure_val):
                meilleur_P = voisin
                meilleur_rang, meilleure_val = rang, val

    return meilleur_P

def metaheuristic(M, iterations, alpha_init=0.3, alpha_step=0.1, alpha_bounds=(0.1, 0.9)):
    m, n = M.shape
    meilleur_pattern = np.ones((m, n))  # Pattern initial
    meilleur_rang, meilleure_val = fobj(M, meilleur_pattern)
    alpha = alpha_init

    for it in range(iterations):
        # Phase constructive : générer une solution avec une LCR
        pattern_initial = construire_solution(M, alpha)
        
        # Phase de recherche locale : améliorer la solution
        pattern_ameliore = recherche_locale(M, pattern_initial)
        
        # Comparer avec la meilleure solution actuelle
        if compareP1betterthanP2(M, pattern_ameliore, meilleur_pattern):
            meilleur_pattern = pattern_ameliore
            meilleur_rang, meilleure_val = fobj(M, meilleur_pattern)
            print(f"Iter {it+1}: Nouvelle meilleure solution avec rang={meilleur_rang}, val={meilleure_val:.6f}")

        # Mise à jour dynamique de alpha
        # Si une meilleure solution est trouvée, intensifier (réduire alpha)
        # Sinon, diversifier (augmenter alpha)
        if compareP1betterthanP2(M, pattern_ameliore, meilleur_pattern):
            alpha = max(alpha_bounds[0], alpha - alpha_step)  # Réduction de alpha
        else:
            alpha = min(alpha_bounds[1], alpha + alpha_step)  # Augmentation de alpha

    return meilleur_pattern

def lecture_fichier(path):
    with open(path, 'r') as fin:  # ouverture du fichier en mode lecture
        m, n = map(int, fin.readline().rstrip().split())  # lecture des dimensions m et n
        data = []  # initialisation d'une liste pour stocker les matrices
        
        # Lecture des lignes suivantes contenant les éléments de la matrice
        for _ in range(m):
            ligne = list(map(float, fin.readline().rstrip().split()))  
            data.append(ligne)  
        
    return np.array(data)   # Renvoie la matrice sous forme de tableau numpy

# Test de l'algorithme
if __name__ == "__main__":
    M = lecture_fichier('correl5_matrice.txt')
    print(M.shape)
    print(M)
    #M = matrices2_slackngon(7)  # Générer la matrice du problème

    best_pattern = metaheuristic(M, iterations=1000, alpha_init=0.8, alpha_step=0.05)
    print("Meilleur pattern trouvé :")
    print(best_pattern)
    print(fobj(M, best_pattern))

