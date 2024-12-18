"""
Groupe 6

"""
import numpy as np
import random


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


def recherche_locale_colonne(M, P_temp, col, candidats_colonnes, scores, num_modifications=2):
    """
    Applique une recherche locale sur les candidats de la LCR.
    Modifie les colonnes en essayant d'obtenir une meilleure solution, en faisant X essais.
    
    Arguments :
    - M : Matrice originale
    - P_temp : Pattern temporaire
    - col : Indice de la colonne à modifier
    - candidats_colonnes : Liste des colonnes candidates
    - scores : Scores associés (rang, val_sing)
    - num_modifications : Nombre maximum d'inversions simultanées
    
    Retourne :
    - meilleure_colonne : Colonne optimisée après la recherche locale
    - meilleur_score : Score associé à la meilleure colonne
    """
    meilleur_score = min(scores, key=lambda x: (x[0], x[1]))  # Rang et val_sing minimaux initiaux
    meilleure_colonne = candidats_colonnes[scores.index(meilleur_score)]
    
    for i, col_candidat in enumerate(candidats_colonnes):
        for _ in range(100):  # Effectuer les 50 essais pour chaque colonne
            candidat_modifié = col_candidat.copy()
            indices_modif = np.random.choice(len(candidat_modifié), size=num_modifications, replace=False)
            candidat_modifié[indices_modif] *= -1  # Inverser les coefficients sélectionnés
            
            P_temp[:, col] = candidat_modifié
            rang, val_sing, _ = fobj(M[:, :col+1], P_temp[:, :col+1])
            
            # Comparer avec la meilleure solution trouvée
            if rang < meilleur_score[0] or (rang == meilleur_score[0] and val_sing < meilleur_score[1]):
                print(f"Collone {col+1} : Amélioration: rang={rang}, val_sing={val_sing:.4f}")
                meilleure_colonne = candidat_modifié
                meilleur_score = (rang, val_sing)
    
    return meilleure_colonne, meilleur_score

def recherche_locale_ligne(M, P_temp, line, candidats_lines, scores, num_modifications):
    """
    Applique une recherche locale sur les candidats de la LCR (pour une ligne donnée).
    Modifie les lignes en essayant d'obtenir une meilleure solution, en faisant X essais.

    Arguments :
    - M : Matrice originale
    - P_temp : Pattern temporaire
    - line : Indice de la ligne à modifier
    - candidats_lines : Liste des lignes candidates
    - scores : Scores associés (rang, val_sing)
    - num_modifications : Nombre maximum d'inversions simultanées

    Retourne :
    - meilleure_line : Ligne optimisée après la recherche locale
    - meilleur_score : Score associé à la meilleure ligne
    """
    meilleur_score = min(scores, key=lambda x: (x[0], x[1]))  # Rang et val_sing minimaux initiaux
    meilleure_line = candidats_lines[scores.index(meilleur_score)]
    amelioration = False
    for i, line_candidat in enumerate(candidats_lines):
        for _ in range(50):  # Effectuer les 50 essais pour chaque ligne
            
            candidat_modifié = line_candidat.copy()
            indices_modif = np.random.choice(len(candidat_modifié), size=num_modifications, replace=False)
            candidat_modifié[indices_modif] *= -1  # Inverser les coefficients sélectionnés
            print(candidat_modifié)

            P_temp[line, :] = candidat_modifié
            rang, val_sing, _ = fobj(M, P_temp)

            # Comparer avec la meilleure solution trouvée
            if rang < meilleur_score[0] or (rang == meilleur_score[0] and val_sing < meilleur_score[1]):
                print(f"Amélioration {line} : rang={rang}, val_sing={val_sing:.4f}")
                meilleure_line = candidat_modifié
                meilleur_score = (rang, val_sing)
                amelioration = True

    return meilleure_line, meilleur_score, amelioration


def grasp(M, X=2):
    """
    Implémente l'algorithme GRASP avec recherche locale sur les éléments de la LCR.
    """
    m, n = M.shape
    num_candidats = 2 ** min(m, n) if min(m, n) < 20 else 1000
    P = np.zeros((m, n), dtype=int)
    rang_prec = 0
    grasp_data = []  # Collecte des métriques pour le suivi

    for col in range(n):
        candidats_colonnes_set = set()
        candidats_colonnes = []
        scores = []

        for _ in range(num_candidats):
            col_candidat = np.random.choice([-1, 1], size=m).astype(int)
            col_tuple = tuple(col_candidat)
            if  col_tuple not in candidats_colonnes_set:
                if_verif = True
                P_temp = P.copy()
                P_temp[:, col] = col_candidat
                rang, val_sing, _ = fobj(M[:, :col+1], P_temp[:, :col+1])
    
                if (rang - rang_prec) == 0:
                    print(f"YOUPIIII Colonne {col+1}/{n} :  rang = {rang}")
                    P[:, col] = col_candidat
                    rang_prec = rang
                    grasp_data.append((col, rang, val_sing))  # Collecte des métriques
                    if_verif = False
                    break
                else:
                    candidats_colonnes.append(col_candidat)
                    candidats_colonnes_set.add(col_tuple)
                    scores.append((rang, val_sing))
            else: 
                num_candidats +=1
            
            
        if if_verif:
            scores_sorted = sorted(zip(candidats_colonnes, scores), key=lambda x: (x[1][0], x[1][1]))
            seuil = int(len(scores_sorted) * X / 100)
            LCR = scores_sorted[:seuil]
            candidats_LCR, scores_LCR = zip(*LCR)  # Séparer les colonnes et leurs scores

            # Appliquer la recherche locale sur les éléments de la LCR
            meilleure_colonne, meilleur_score = recherche_locale_colonne(
                M, P.copy(), col, list(candidats_LCR), list(scores_LCR))

            P[:, col] = meilleure_colonne
            rang_prec = meilleur_score[0]
            print(f" Colonne {col+1 }/{n} :  rang = {meilleur_score[0]}")
            grasp_data.append((col, rang_prec, meilleur_score[1]))  # Collecte des métriques

    return P, rang_prec, grasp_data


def diversification_grasp(matrice, pattern, best_rank):
    
    m, n = M.shape
    num_modifications = np.random.choice(max(m,n))
    col_modif = np.random.choice(max(m,n), size=num_modifications, replace=False)
    num_candidats = 2 ** min(m, n) if min(m, n) < 20 else 1000
    P = np.zeros((m, n), dtype=int)
    rang_prec = 0
    grasp_data = []  # Collecte des métriques pour le suivi

    for col in col_modif:
        candidats_colonnes_set = set()
        candidats_colonnes = []
        scores = []

        for _ in range(num_candidats):
            col_candidat = np.random.choice([-1, 1], size=m).astype(int)
            col_tuple = tuple(col_candidat)
            if  col_tuple not in candidats_colonnes_set:
                if_verif = True
                P_temp = P.copy()
                P_temp[:, col] = col_candidat
                rang, val_sing, _ = fobj(M[:, :col+1], P_temp[:, :col+1])
    
                if (rang - rang_prec) == 0:
                    print(f"YOUPIIII Colonne {col+1}/{n} :  rang = {rang}")
                    P[:, col] = col_candidat
                    rang_prec = rang
                    grasp_data.append((col, rang, val_sing))  # Collecte des métriques
                    if_verif = False
                    break
                else:
                    candidats_colonnes.append(col_candidat)
                    candidats_colonnes_set.add(col_tuple)
                    scores.append((rang, val_sing))
            else: 
                num_candidats +=1
            
            
        if if_verif:
            scores_sorted = sorted(zip(candidats_colonnes, scores), key=lambda x: (x[1][0], x[1][1]))
            seuil = int(len(scores_sorted) * X / 100)
            LCR = scores_sorted[:seuil]
            candidats_LCR, scores_LCR = zip(*LCR)  # Séparer les colonnes et leurs scores

            # Appliquer la recherche locale sur les éléments de la LCR
            meilleure_colonne, meilleur_score = recherche_locale_colonne(
                M, P.copy(), col, list(candidats_LCR), list(scores_LCR))

            P[:, col] = meilleure_colonne
            rang_prec = meilleur_score[0]
            print(f" Colonne {col+1 }/{n} :  rang = {meilleur_score[0]}")
            grasp_data.append((col, rang_prec, meilleur_score[1]))  # Collecte des métriques

    

def search_ligne_indep(matrice, pattern, Transposee =False):
    if Transposee:
        matrice = matrice.T
        pattern = pattern.T
 
    liste_index = []
    rang_init = fobj(matrice,pattern)[0]
    for i in range(np.shape(matrice)[0]):
        lignes_M = []
        lignes_P = []
        for j in range(np.shape(matrice)[0]):
            if j != i:
                lignes_M.append(matrice[j])
                lignes_P.append(pattern[j])
        matrice_temp = np.array(lignes_M)
        pattern_temp = np.array(lignes_P)
        rang = fobj(matrice_temp,pattern_temp)[0]
        
        if rang<rang_init:
            liste_index.append(i)
    return liste_index


def use_search_line(M, P, line_list, rang, X=1, num_modifications=3):
    """
    Optimise les lignes indépendantes identifiées dans `line_list` avec recherche locale.

    Arguments :
    - M : Matrice originale
    - P : Pattern actuel
    - line_list : Liste des indices des lignes à optimiser
    - rang : Rang initial de la matrice
    - X : Pourcentage pour sélectionner les meilleurs candidats (diversification)
    - num_modifications : Nombre maximum d'inversions simultanées pour la recherche locale

    Retourne :
    - new_P : Nouveau pattern après optimisation
    - search_line_data : Données collectées pour suivi des métriques
    - rang_prec : Rang final après optimisation
    """
    m, n = M.shape
    num_candidats = 2 ** min(m, n) if min(m, n) < 15 else 50000
    rang_prec = rang
    new_P = P.copy()
    search_line_data = []  # Collecte des métriques
    amelioration = False
    
    if len(line_list):
        for line in line_list:
            candidats_lines_set = set()
            candidats_lines = []
            scores = []
            
            # Générer plusieurs lignes candidates
            for _ in range(num_candidats):
                line_candidat = np.random.choice([-1, 1], size=n).astype(int)
                line_tuple = tuple(line_candidat)
                
                if  line_tuple not in candidats_lines_set:
                    if_verif = True
                    P_temp = new_P.copy()
                    P_temp[line, :] = line_candidat
                    rang, val_sing, _ = fobj(M, P_temp)
        
                    if rang < rang_prec:
                        print(f"YOUPIIII Ligne {line}/{m} : rang = {rang}")
                        new_P[line, :] = line_candidat
                        rang_prec = rang
                        search_line_data.append((line, rang, val_sing))
                        if_verif = False
                        amelioration = True
                        break
                    else:
                        candidats_lines.append(line_candidat)
                        candidats_lines_set.add(line_tuple)
                        scores.append((rang, val_sing))
                else: 
                    num_candidats +=1
    
            if if_verif:
                # Trier les candidats selon le rang, puis par la valeur singulière
                scores_sorted = sorted(zip(candidats_lines, scores), key=lambda x: (x[1][0], x[1][1]))
                seuil = int(len(scores_sorted) * X / 100)
                LCR = scores_sorted[:seuil]
    
                # Recherche locale sur les candidats de la LCR
                P_temp = new_P.copy()
                meilleure_line, meilleur_score, amelioration = recherche_locale_ligne(
                    M, P_temp, line, [x[0] for x in LCR], [x[1] for x in LCR], num_modifications
                )
    
                # Appliquer la meilleure ligne trouvée
                new_P[line, :] = meilleure_line
                rang_prec = meilleur_score[0]
                print(f"Ligne {line}/{m} après recherche locale : rang = {rang_prec}")
                search_line_data.append((line, rang_prec, meilleur_score[1]))

    return new_P, search_line_data, rang_prec, amelioration


import matplotlib.pyplot as plt

def plot_grasp(grasp_data):
    cols, rangs, val_sings = zip(*grasp_data)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(cols, rangs, marker='o', label='Rang')
    plt.xlabel('Colonnes')
    plt.ylabel('Rang')
    plt.title('Évolution du Rang (GRASP)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(cols, val_sings, marker='o', color='orange', label='Valeur singulière')
    plt.xlabel('Colonnes')
    plt.ylabel('Dernière Valeur Singulière')
    plt.title('Évolution de la Valeur Singulière (GRASP)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_search_line(search_line_data):
    lines, rangs, val_sings = zip(*search_line_data)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(lines, rangs, marker='o', label='Rang')
    plt.xlabel('Lignes')
    plt.ylabel('Rang')
    plt.title('Évolution du Rang (Search Line)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lines, val_sings, marker='o', color='orange', label='Valeur singulière')
    plt.xlabel('Lignes')
    plt.ylabel('Dernière Valeur Singulière')
    plt.title('Évolution de la Valeur Singulière (Search Line)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()



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
    m = 120
    n = 120
    r = 8

    # Mesurer le temps de début
    start_time = time.time()

    # Générer la matrice ou lire depuis un fichier
    #M = random_matrix(m,n,r) # Exemple de matrice aléatoire
    M = lecture_fichier('correl5_matrice.txt')
    #M = lecture_fichier('exempleslide_matrice.txt')
    #M = matrices1_ledm(20)
    #M = lecture_fichier('120.txt')
   
    # Exécution des algorithmes
    best_pattern, rang, grasp_data = grasp(M)
    line_list = search_ligne_indep(M, best_pattern)
    print(f"This is the line liste : ", line_list)
    new_best_pattern_line, search_line_data, new_rang, amelioration = use_search_line(M, best_pattern, line_list, rang)
    
    for i in range(3):
        line_list_prec = line_list.copy()
        print(f"\nItération {i+1} : Liste des lignes indépendantes : {line_list}")
        new_best_pattern_line, search_line_data, new_rang, amelioration = use_search_line(M, new_best_pattern_line, line_list, new_rang)
        
        # Recalculer les lignes indépendantes
        line_list = search_ligne_indep(M, new_best_pattern_line)
        
        # Vérifier si l'état des lignes n'a pas changé
        if not amelioration:
            print("Aucune amélioration supplémentaire trouvée. Arrêt de la boucle.")
            break
        # Tracer les métriques
        plot_search_line(search_line_data)
    
        print(f"\nRésultat fobj {i} : \n", fobj(M, new_best_pattern_line)[0], fobj(M, new_best_pattern_line)[1])
        
    # Tracer les métriques
    plot_grasp(grasp_data)
    


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
    print("Résultat fobj 1 : \n", fobj(M, best_pattern)[0], fobj(M, best_pattern)[1])
    
    print("\nRésultat fobj 2 : \n", fobj(M, new_best_pattern_line)[0], fobj(M, new_best_pattern_line)[1])
    
    # Écrire les résultats dans un fichier
    ecriture_fichier('resultat_pattern1.txt', best_pattern, fobj(M, best_pattern)[2])

    ecriture_fichier('resultat_pattern2_lines.txt', new_best_pattern_line, fobj(M, new_best_pattern_line)[2])


