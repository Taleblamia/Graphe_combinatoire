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

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  valeurs_non_nulles = [val for val in sing_values if val > tol]  # Filtrer les valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]], valeurs_non_nulles      # outputs: objectif1=rang, objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M, P1, P2):
    r1, s1, sv1 = fobj(M, P1)  # on récupère les deux objectifs pour le pattern P1
    r2, s2, sv2 = fobj(M, P2)  # on récupère les deux objectifs pour le pattern P2
    if r1 != r2:  # on traite les objectifs de façon lexicographique :
        return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
    return s1 < s2  # alors on prend en compte la valeur de la plus petite valeur singulière

def construire_solution(M, alpha=0.3):
    m, n = M.shape  # m et n sont les dimensions de la matrice M
    candidats = []
    scores = []

    # Générer un ensemble de candidats
    for _ in range(20):  # Nombre de candidats à générer
        pattern = np.random.choice([1, -1], size=(m, n))  # Matrix de taille (m, n)
        rang, val, sing_values = fobj(M, pattern)
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

def recherche_locale(M, P, max_voisins=50):
    m, n = P.shape
    meilleur_P = P.copy()
    meilleur_rang, meilleure_val, _ = fobj(M, meilleur_P)
    
    voisins_a_tester = [(i, j) for i in range(m) for j in range(n)]
    np.random.shuffle(voisins_a_tester)  # Mélange des indices
    voisins_a_tester = voisins_a_tester[:max_voisins]  # Sous-échantillon

    for i, j in voisins_a_tester:
        voisin = meilleur_P.copy()
        voisin[i, j] *= -1  # Inversion du coefficient
        rang, val, _ = fobj(M, voisin)
        if rang < meilleur_rang or (rang == meilleur_rang and val < meilleure_val):
            meilleur_P = voisin
            meilleur_rang, meilleure_val = rang, val

    return meilleur_P


def construction_heuristique(M):
    m, n = M.shape
    P = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i == j or i + j == n - 1:
                P[i, j] = 1
            else:
                P[i, j] = -1
    return P


def metaheuristic(M, iterations, alpha_init=0.3, alpha_step=0.1, alpha_bounds=(0.1, 0.9)):
    m, n = M.shape
    meilleur_pattern = np.ones((m, n))  # Pattern initial
    meilleur_rang, meilleure_val , sing_values = fobj(M, meilleur_pattern)
    alpha = alpha_init

    for it in range(iterations):
        # Phase constructive : générer une solution avec une LCR
        pattern_initial = construire_solution(M, alpha)
        
        # Phase de recherche locale : améliorer la solution
        pattern_ameliore = recherche_locale(M, pattern_initial)
        
        # Comparer avec la meilleure solution actuelle
        if compareP1betterthanP2(M, pattern_ameliore, meilleur_pattern):
            meilleur_pattern = pattern_ameliore
            meilleur_rang, meilleure_val, sing_values = fobj(M, meilleur_pattern)
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

def ecriture_fichier(path, P, valeurs_sing):
    with open(path, 'w') as fout:
        # Écrire le pattern ligne par ligne
        for ligne in P:
            fout.write(' '.join(map(str, ligne)) + '\n')

        # Écrire les valeurs singulières non nulles
        for val in valeurs_sing:
            fout.write(f"{val}\n")
def random_matrix(m, n, r):
    return ((np.random.rand(m, r) * 10) @ (np.random.rand(r, n) * 10)) ** 2
import time
import numpy as np

# Test de l'algorithme
if __name__ == "__main__":
    m = 30
    n = 30
    r = 2

    # Mesurer le temps de début
    start_time = time.time()

    # Générer la matrice ou lire depuis un fichier
    M = np.random.rand(m, n) * r  # Exemple de matrice aléatoire
    #M = lecture_fichier('correl5_matrice.txt')
    #M = lecture_fichier('exempleslide_matrice.txt')

    # Exécuter l'algorithme métaheuristique
    best_pattern = metaheuristic(M, iterations=200, alpha_init=0.8, alpha_step=0.1)

    # Mesurer le temps de fin
    end_time = time.time()

    # Calculer la durée
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # Afficher les résultats
    print("\nTemps d'exécution : {:02d}h {:02d}m {:.2f}s".format(hours, minutes, seconds))
    
    print("\nMeilleur pattern trouvé :")
    #print(best_pattern)
    print("Résultat fobj : \n", fobj(M, best_pattern)[0], fobj(M, best_pattern)[1])

    # Écrire les résultats dans un fichier
    ecriture_fichier('resultat_pattern.txt', best_pattern, fobj(M, best_pattern)[2])
#%% Version 2
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

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  valeurs_non_nulles = [val for val in sing_values if val > tol]  # Filtrer les valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]], valeurs_non_nulles      # outputs: objectif1=rang, objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M, P1, P2):
    r1, s1, sv1 = fobj(M, P1)  # on récupère les deux objectifs pour le pattern P1
    r2, s2, sv2 = fobj(M, P2)  # on récupère les deux objectifs pour le pattern P2
    if r1 != r2:  # on traite les objectifs de façon lexicographique :
        return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
    return s1 < s2  # alors on prend en compte la valeur de la plus petite valeur singulière

def construire_solution_guidée(M, alpha=0.3):
    m, n = M.shape
    candidats = []
    scores = []

    # Générer des candidats en favorisant les zones à fort poids dans M
    for _ in range(20):  # Nombre de candidats à générer
        pattern = construction_heuristique(M)  # Initialiser un pattern avec des 1
        # Introduire des inversions (-1) en priorité dans les zones où M est dense
        densite = M / M.max()  # Normaliser les valeurs de M pour avoir une densité relative
        for i in range(m):
            for j in range(n):
                if np.random.rand() < densite[i, j]:  # Probabilité liée à la densité
                    pattern[i, j] *= -1
        rang, val, sing_values = fobj(M, pattern)
        candidats.append(pattern)
        scores.append((rang, val))

    # Déterminer cmin et cmax
    rangs, valeurs = zip(*scores)
    cmin = min(rangs)
    cmax = max(rangs)
    seuil = cmin + alpha * (cmax - cmin)

    # Construire la LCR
    LCR = [c for c, (r, _) in zip(candidats, scores) if r <= seuil]
    if not LCR:
        raise ValueError("La liste des candidats restreinte (LCR) est vide. Vérifiez les paramètres de alpha.")
    
    # Sélectionner une matrice dans la LCR
    index = np.random.randint(len(LCR))
    return LCR[index]


def recherche_locale_guidée(M, P, max_voisins=100):
    m, n = P.shape
    meilleur_P = P.copy()
    meilleur_rang, meilleure_val, _ = fobj(M, meilleur_P)

    # Trier les voisins en fonction des poids de M
    voisins = [(i, j) for i in range(m) for j in range(n)]
    voisins = sorted(voisins, key=lambda x: -M[x[0], x[1]])  # Priorité aux indices avec valeurs élevées dans M

    voisins_a_tester = voisins[:max_voisins]  # Limiter aux voisins les plus prometteurs

    for i, j in voisins_a_tester:
        voisin = meilleur_P.copy()
        voisin[i, j] *= -1  # Inversion du coefficient
        rang, val, _ = fobj(M, voisin)
        if rang < meilleur_rang or (rang == meilleur_rang and val < meilleure_val):
            meilleur_P = voisin
            meilleur_rang, meilleure_val = rang, val

    return meilleur_P



def construction_heuristique(M):
    m, n = M.shape
    P = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i == j or i + j == n - 1:
                P[i, j] = 1
            else:
                P[i, j] = -1
    return P


def metaheuristic_guidée(M, iterations, alpha_init=0.3, alpha_step=0.1, alpha_bounds=(0.1, 0.9)):
    m, n = M.shape
    meilleur_pattern = construction_heuristique(M) # Pattern initial
    meilleur_rang, meilleure_val, sing_values = fobj(M, meilleur_pattern)
    alpha = alpha_init

    for it in range(iterations):
        # Phase constructive guidée
        pattern_initial = construire_solution_guidée(M, alpha)

        # Phase de recherche locale guidée
        pattern_ameliore = recherche_locale_guidée(M, pattern_initial)

        # Comparer avec la meilleure solution actuelle
        if compareP1betterthanP2(M, pattern_ameliore, meilleur_pattern):
            meilleur_pattern = pattern_ameliore
            meilleur_rang, meilleure_val, sing_values = fobj(M, meilleur_pattern)
            print(f"Iter {it+1}: Nouvelle meilleure solution avec rang={meilleur_rang}, val={meilleure_val:.6f}")

        # Mise à jour dynamique de alpha
        if compareP1betterthanP2(M, pattern_ameliore, meilleur_pattern):
            alpha = max(alpha_bounds[0], alpha - alpha_step)
        else:
            alpha = min(alpha_bounds[1], alpha + alpha_step)

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

def ecriture_fichier(path, P, valeurs_sing):
    with open(path, 'w') as fout:
        # Écrire le pattern ligne par ligne
        for ligne in P:
            fout.write(' '.join(map(str, ligne)) + '\n')

        # Écrire les valeurs singulières non nulles
        for val in valeurs_sing:
            fout.write(f"{val}\n")
def random_matrix(m, n, r):
    return ((np.random.rand(m, r) * 10) @ (np.random.rand(r, n) * 10)) ** 2

import time


# Test de l'algorithme
if __name__ == "__main__":
    m = 30
    n = 30
    r = 2

    # Mesurer le temps de début
    start_time = time.time()

    # Générer la matrice ou lire depuis un fichier
    M = random_matrix(m,n,r) # Exemple de matrice aléatoire
    M = lecture_fichier('correl5_matrice.txt')
    #M = lecture_fichier('exempleslide_matrice.txt')

    # Exécuter l'algorithme métaheuristique
    best_pattern = metaheuristic_guidée(M, iterations=100, alpha_init=0.8, alpha_step=0.1)

    # Mesurer le temps de fin
    end_time = time.time()

    # Calculer la durée
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # Afficher les résultats
    print("\nTemps d'exécution : {:02d}h {:02d}m {:.2f}s".format(hours, minutes, seconds))
    
    print("\nMeilleur pattern trouvé :")
    #print(best_pattern)
    print("Résultat fobj : \n", fobj(M, best_pattern)[0], fobj(M, best_pattern)[1])

    # Écrire les résultats dans un fichier
    ecriture_fichier('resultat_pattern.txt', best_pattern, fobj(M, best_pattern)[2])
#%%
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

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  valeurs_non_nulles = [val for val in sing_values if val > tol]  # Filtrer les valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]], valeurs_non_nulles      # outputs: objectif1=rang, objectif2=plus petite val sing. non-nulle

def compareP1betterthanP2(M, P1, P2):
    r1, s1, sv1 = fobj(M, P1)  # on récupère les deux objectifs pour le pattern P1
    r2, s2, sv2 = fobj(M, P2)  # on récupère les deux objectifs pour le pattern P2
    if r1 != r2:  # on traite les objectifs de façon lexicographique :
        return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
    return s1 < s2  # alors on prend en compte la valeur de la plus petite valeur singulière

def construire_solution_guidée(M, alpha=0.3):
    m, n = M.shape
    candidats = []
    scores = []

    # Générer des candidats en favorisant les zones à fort poids dans M
    for _ in range(20):  # Nombre de candidats à générer
        pattern = construction_heuristique(M)  # Initialiser un pattern avec des 1
        # Introduire des inversions (-1) en priorité dans les zones où M est dense
        densite = M / M.max()  # Normaliser les valeurs de M pour avoir une densité relative
        for i in range(m):
            for j in range(n):
                if np.random.rand() < densite[i, j]:  # Probabilité liée à la densité
                    pattern[i, j] *= -1
        rang, val, sing_values = fobj(M, pattern)
        candidats.append(pattern)
        scores.append((rang, val))

    # Déterminer cmin et cmax
    rangs, valeurs = zip(*scores)
    cmin = min(rangs)
    cmax = max(rangs)
    seuil = cmin + alpha * (cmax - cmin)

    # Construire la LCR
    LCR = [c for c, (r, _) in zip(candidats, scores) if r <= seuil]
    if not LCR:
        raise ValueError("La liste des candidats restreinte (LCR) est vide. Vérifiez les paramètres de alpha.")
    
    # Sélectionner une matrice dans la LCR
    index = np.random.randint(len(LCR))
    return LCR[index]


def recherche_locale_guidée(M, P, max_voisins=100):
    m, n = P.shape
    meilleur_P = P.copy()
    meilleur_rang, meilleure_val, _ = fobj(M, meilleur_P)

    # Trier les voisins en fonction des poids de M
    voisins = [(i, j) for i in range(m) for j in range(n)]
    voisins = sorted(voisins, key=lambda x: -M[x[0], x[1]])  # Priorité aux indices avec valeurs élevées dans M

    voisins_a_tester = voisins[:max_voisins]  # Limiter aux voisins les plus prometteurs

    for i, j in voisins_a_tester:
        voisin = meilleur_P.copy()
        voisin[i, j] *= -1  # Inversion du coefficient
        rang, val, _ = fobj(M, voisin)
        if rang < meilleur_rang or (rang == meilleur_rang and val < meilleure_val):
            meilleur_P = voisin
            meilleur_rang, meilleure_val = rang, val

    return meilleur_P

import random

def construction_heuristique(M):
    m, n = M.shape
    P = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i == j or i + j == n - 1:
                P[i, j] = 1
            else:
                P[i, j] = -1
    return P


def grasp(M, num_candidats=10, X=10):
    """
    Construit le pattern (matrice P) colonne par colonne, en gardant chaque colonne fixée après sa construction.
    Sélection des X% meilleurs candidats en tenant compte du rang et de la plus petite valeur singulière.
    
    Arguments :
    - M : Matrice de base à transformer
    - num_candidats : Nombre de colonnes candidates générées à chaque itération
    - X : Pourcentage pour sélectionner les meilleurs candidats (diversification)
    
    Retourne :
    - P : Matrice pattern optimisée
    """
    m, n = M.shape
    P = np.zeros((m, n))  # Initialise une matrice vide pour construire progressivement

    for col in range(n):
        candidats_colonnes = []  # Liste pour stocker les colonnes candidates
        scores = []  # Scores associés aux colonnes candidates (rang, val_sing)

        # Générer plusieurs colonnes candidates
        for _ in range(num_candidats):
            col_candidat = np.random.choice([-1, 1], size=m)  # Colonne aléatoire avec -1 et 1
            P_temp = P.copy()
            P_temp[:, col] = col_candidat  # Intégrer la colonne au pattern temporaire
            rang, val_sing, _ = fobj(M[:,:col+1], P_temp[:,:col+1])  # Évaluer la fonction objectif
            candidats_colonnes.append(col_candidat)
            scores.append((rang, val_sing))  # Stocker le rang et la plus petite valeur singulière

        # Trier les candidats selon le rang, puis par la valeur singulière
        scores_sorted = sorted(zip(candidats_colonnes, scores), key=lambda x: (x[1][0], x[1][1]))

        # Étape 1 : Sélectionner les X% meilleurs candidats en fonction du rang et de la valeur singulière
        seuil = int(len(scores_sorted) * X / 100)  # Calcul du nombre de meilleurs candidats à prendre
        LCR = scores_sorted[:seuil]  # Liste des X% meilleurs candidats
        
        # Étape 2 : Sélectionner un candidat de manière aléatoire parmi les X% meilleurs candidats
        #meilleure_colonne, _ = random.choice(LCR)
        meilleure_colonne, _ = LCR[0]

        # Fixer la colonne sélectionnée
        P[:, col] = meilleure_colonne

        # Afficher l'état intermédiaire pour le suivi
        print(f"{col + 1}/{n} fixée : Rang minimal = {_[0]}, Val_sing = {_[1]}")

    return P


def lecture_fichier(path):
    with open(path, 'r') as fin:  # ouverture du fichier en mode lecture
        m, n = map(int, fin.readline().rstrip().split())  # lecture des dimensions m et n
        data = []  # initialisation d'une liste pour stocker les matrices
        
        # Lecture des lignes suivantes contenant les éléments de la matrice
        for _ in range(m):
            ligne = list(map(float, fin.readline().rstrip().split()))  
            data.append(ligne)  
        
    return np.array(data)   # Renvoie la matrice sous forme de tableau numpy

def ecriture_fichier(path, P, valeurs_sing):
    with open(path, 'w') as fout:
        # Écrire le pattern ligne par ligne
        for ligne in P:
            fout.write(' '.join(map(str, ligne)) + '\n')

        # Écrire les valeurs singulières non nulles
        for val in valeurs_sing:
            fout.write(f"{val}\n")
def random_matrix(m, n, r):
    return ((np.random.rand(m, r) * 10) @ (np.random.rand(r, n) * 10)) ** 2

import time


# Test de l'algorithme
if __name__ == "__main__":
    m = 30
    n = 30
    r = 2

    # Mesurer le temps de début
    start_time = time.time()

    # Générer la matrice ou lire depuis un fichier
    #M = random_matrix(m,n,r) # Exemple de matrice aléatoire
    M = lecture_fichier('correl5_matrice.txt')
    #M = lecture_fichier('exempleslide_matrice.txt')

    # Exécuter l'algorithme métaheuristique
    best_pattern = grasp(M, num_candidats=100000)

    # Mesurer le temps de fin
    end_time = time.time()

    # Calculer la durée
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # Afficher les résultats
    print("\nTemps d'exécution : {:02d}h {:02d}m {:.2f}s".format(hours, minutes, seconds))
    
    print("\nMeilleur pattern trouvé :")
    #print(best_pattern)
    print("Résultat fobj : \n", fobj(M, best_pattern)[0], fobj(M, best_pattern)[1])

    # Écrire les résultats dans un fichier
    ecriture_fichier('resultat_pattern.txt', best_pattern, fobj(M, best_pattern)[2])
#%%
