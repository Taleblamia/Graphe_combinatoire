# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:35:53 2024

@author: Max
"""

import numpy as np

def fobj(M,P):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False)    # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  tol         = max(M.shape)*sing_values[0]*np.finfo(float).eps  # Calcul de la tolérance à utiliser pour la matrice P*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                   # indices des valeurs > tolérance
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]          # outputs: objectif1=rang, objectif2=plus petite val sing. non-nulle


# Liste pour accumuler les lignes
M = []

# Ajouter les lignes une par une
M.append([2, 4, 9])  # Première ligne
M.append([16, 25, 36])  # Deuxième ligne
M.append([25, 49, 81])  # Troisième ligne

P = []

# Ajouter les lignes une par une
P.append([1, 1, 1])  # Première ligne
P.append([1, 1, 1])  # Deuxième ligne
P.append([1, 1, 1])  # Troisième ligne

# Convertir la liste en matrice NumPy
matrice_init = np.array(M)
pattern_init = np.array(P)


def search_ligne_indep(matrice, pattern, Transposee):
    
    if Transposee:
        matrice = matrice.T
        #pattern = pattern.T
    
    print(matrice)

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
        print(rang)
        
        if rang<rang_init:
            liste_index.append(i)
        
    return liste_index


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


M2 = lecture_fichier('correl5_matrice.txt')
P2 = lecture_fichier('resultat_pattern.txt')


x = search_ligne_indep(M2, P2, False)

print(x)


