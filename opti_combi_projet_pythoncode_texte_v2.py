
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


def grasp(M, X=100, shuffle = True):
    
    m, n = M.shape
    
    liste = []
    
    for i in range(max(m,n)):
        liste.append(i)
    
    if shuffle:
        random.shuffle(liste)
        
        

    num_candidats = 2 ** min(m, n) if min(m, n) < 17 else 100000
    P = np.zeros((m, n), dtype=int)  # Initialise une matrice vide pour construire progressivement
    rang_prec = 0
    
    for col in liste:
        candidats_colonnes = []  # Liste pour stocker les colonnes candidates
        scores = []  # Scores associés aux colonnes candidates (rang, val_sing)
        if_verif= True
        
        # Générer plusieurs colonnes candidates
        for _ in range(num_candidats):
            col_candidat = np.random.choice([-1, 1], size=m).astype(int)  # Colonne aléatoire avec -1 et 1
            P_temp = P.copy()
            P_temp[:, col] = col_candidat  # Intégrer la colonne au pattern temporaire
            rang, val_sing, _ = fobj(M[:,:col+1], P_temp[:,:col+1])  # Évaluer la fonction objectif
            
            
            if (rang - rang_prec) == 0:
                print(f"Youpi je suis passé avec -1 rang = {rang}")

                P[:, col] = col_candidat
                rang_prec = rang
                if_verif= False
                break
                
            else:
                
                candidats_colonnes.append(col_candidat)
                scores.append((rang, val_sing))  # Stocker le rang et la plus petite valeur singulière
                
               
        if if_verif:
            
        
            # Trier les candidats selon le rang, puis par la valeur singulière
            scores_sorted = sorted(zip(candidats_colonnes, scores), key=lambda x: (x[1][0], x[1][1]))
    
            # Étape 1 : Sélectionner les X% meilleurs candidats en fonction du rang et de la valeur singulière
            seuil = int(len(scores_sorted) * X / 100)  # Calcul du nombre de meilleurs candidats à prendre
            
            LCR = scores_sorted[:seuil]  # Liste des X% meilleurs candidats
            
            rang_prec = rang
            
            # Étape 2 : Sélectionner un candidat de manière aléatoire parmi les X% meilleurs candidats
            print(f"Colonne {col + 1}/{n} :  rang = {rang}")

            #meilleure_colonne, _ = random.choice(LCR)
            meilleure_colonne, _ = LCR[0]
    
            # Fixer la colonne sélectionnée
            P[:, col] = meilleure_colonne


    return P,rang

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


def use_search_line (M,P, line_list,rang,  X = 50):
    m, n = M.shape
    num_candidats = 2 ** min(m, n) if min(m, n) < 17 else 100000  # Utiliser 1000 ou une autre valeur raisonnable comme limite

    rang_prec = rang
    new_P = P.copy()
    
    for line in line_list:
        candidats_lines = []  # Liste pour stocker les colonnes candidates
        scores = []  # Scores associés aux colonnes candidates (rang, val_sing)
        if_verif= True
        
        # Générer plusieurs colonnes candidates
        for _ in range(num_candidats):
            line_candidat = np.random.choice([-1, 1], size=n).astype(int)  # Colonne aléatoire avec -1 et 1
            P_temp = new_P.copy()
            P_temp[line, :] = line_candidat  # Intégrer la colonne au pattern temporaire
            
            rang, val_sing, _ = fobj(M, P_temp)  # Évaluer la fonction objectif
            
            
            if (rang - rang_prec) < 0:
                print(f"Line {line} YOUPIIII avec rang = {rang}")

                new_P[line, :] = line_candidat
                rang_prec = rang
                candidats_lines = []
                if_verif= False
                break
                
            else:
                
                candidats_lines.append(line_candidat)
                scores.append((rang, val_sing))  # Stocker le rang et la plus petite valeur singulière
                
               
        if if_verif:
            
        
            # Trier les candidats selon le rang, puis par la valeur singulière
            scores_sorted = sorted(zip(candidats_lines, scores), key=lambda x: (x[1][0], x[1][1]))
    
            # Étape 1 : Sélectionner les X% meilleurs candidats en fonction du rang et de la valeur singulière
            seuil = int(len(scores_sorted) * X / 100)  # Calcul du nombre de meilleurs candidats à prendre
            
            LCR = scores_sorted[:seuil]  # Liste des X% meilleurs candidats
            
            rang_prec = rang
            
            # Étape 2 : Sélectionner un candidat de manière aléatoire parmi les X% meilleurs candidats
            print(f"Ligne {line }/{m} :  rang = {rang}")

            #meilleure_line, _ = random.choice(LCR)
            meilleure_line, _ = LCR[0]
    
            # Fixer la ligne sélectionnée
            new_P[line, :] = meilleure_line


    return new_P


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
    M = random_matrix(m,n,r) # Exemple de matrice aléatoire
    #M = lecture_fichier('correl5_matrice.txt')
    #M = lecture_fichier('exempleslide_matrice.txt')

    # Exécuter l'algorithme métaheuristique
    best_pattern , rang = grasp(M )
    
    line_list = search_ligne_indep(M, best_pattern)
    
    print(f"This is the line liste : ", line_list)
    
    #Appel de la fonction pour trouver le nouveau pattern en retravaillant sur les lignes
    new_best_pattern_line = use_search_line(M, best_pattern, line_list, rang)
    
    

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


