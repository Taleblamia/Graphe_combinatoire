# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:00:05 2024

@author: Mounir
"""
import numpy as np

def construction_gradient(M):
    """
    Construit un pattern initial basé sur les gradients locaux.
    Les grandes variations locales dans M sont favorisées.
    
    :param M: Matrice d'entrée (numpy array)
    :return: Pattern initial P (numpy array)
    """
    m, n = M.shape
    P = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            voisins = []
            if i > 0: voisins.append(M[i - 1, j])  # voisin haut
            if i < m - 1: voisins.append(M[i + 1, j])  # voisin bas
            if j > 0: voisins.append(M[i, j - 1])  # voisin gauche
            if j < n - 1: voisins.append(M[i, j + 1])  # voisin droit
            
            # Gradient local : valeur comparée à la moyenne des voisins
            moyenne_voisins = np.mean(voisins) if voisins else 0
            P[i, j] = 1 if M[i, j] >= moyenne_voisins else -1
    
    return P


def construction_diagonale(M):
    """
    Construit un pattern initial en favorisant les diagonales.
    Les diagonales principales reçoivent +1 et les autres -1.
    
    :param M: Matrice d'entrée (numpy array)
    :return: Pattern initial P (numpy array)
    """
    m, n = M.shape
    P = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            if i == j or i + j == n - 1:  # diagonale principale ou secondaire
                P[i, j] = 1
            else:
                P[i, j] = -1
    
    return P

from sklearn.cluster import KMeans

def construction_clustering(M, n_clusters=2):
    """
    Construit un pattern initial basé sur le clustering des valeurs de M.
    Les clusters sont utilisés pour attribuer +1 ou -1.
    
    :param M: Matrice d'entrée (numpy array)
    :param n_clusters: Nombre de clusters pour K-means (par défaut 2).
    :return: Pattern initial P (numpy array)
    """
    m, n = M.shape
    M_flat = M.flatten().reshape(-1, 1)  # Transforme M en vecteur pour le clustering
    
    # Appliquer K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(M_flat)
    labels = kmeans.labels_.reshape(m, n)  # Reconstruire la matrice des labels
    
    # Attribuer +1 ou -1 selon les clusters
    P = np.where(labels == labels.max(), 1, -1)  # Le plus grand cluster reçoit +1
    
    return P

def construction_svd(M):
    """
    Construit un pattern initial basé sur les valeurs propres de la décomposition SVD.
    Utilise les vecteurs propres dominants pour structurer P.
    
    :param M: Matrice d'entrée (numpy array)
    :return: Pattern initial P (numpy array)
    """
    U, S, Vt = np.linalg.svd(M)  # Décomposition SVD
    dominant_pattern = np.outer(U[:, 0], Vt[0, :])  # Utilise les vecteurs propres dominants
    
    # Construction de P : +1 ou -1 selon le signe des valeurs dans dominant_pattern
    P = np.where(dominant_pattern >= 0, 1, -1)
    
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

def fobj(M, P, tol=1e-14):
    sing_values = np.linalg.svd(P * np.sqrt(M), compute_uv=False)  # Calcul des valeurs singulières de la matrice P.*sqrt(M)
    ind_nonzero = np.where(sing_values > tol)[0]  # indices des valeurs > tolérance donnée
    valeurs_non_nulles = [val for val in sing_values if val > tol]  # Filtrer les valeurs > tolérance
    return len(ind_nonzero), sing_values[ind_nonzero[-1]]  # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle



if __name__ == "__main__":
    # Lecture de la matrice
    M = lecture_fichier("ledm6_matrice.txt")

    # Gradient local
    P_gradient = construction_gradient(M)
    print("Pattern basé sur le gradient local :")
    print(P_gradient)
    print("fobj (gradient local) :", fobj(M, P_gradient))

    # Diagonale dominante
    P_diagonale = construction_diagonale(M)
    print("Pattern basé sur les diagonales dominantes :")
    print(P_diagonale)
    print("fobj (diagonale dominante) :", fobj(M, P_diagonale))

    # Clustering
    P_clustering = construction_clustering(M)
    print("Pattern basé sur le clustering :")
    print(P_clustering)
    print("fobj (clustering) :", fobj(M, P_clustering))

    # SVD
    P_svd = construction_svd(M)
    print("Pattern basé sur les vecteurs propres (SVD) :")
    print(P_svd)
    print("fobj (SVD) :", fobj(M, P_svd))
