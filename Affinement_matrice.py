# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:45:59 2024

@author: Max
"""

import numpy as np


# Exemple de matrice
matrice = [
    [1, 1, 1],
    [1, 0, 3],
    [2, 0, 6],
    [4, 0, 12],
    [3, 3, 3],
    [4, 1, 12],
    [8, 2, 24]
]

def sont_colineaires(l1, l2):
    # Vérification que les deux lignes sont colinéaires
    # On compare les rapports des éléments de chaque ligne
    ratio = None
    
    for i in range(len(l1)):
        if l1[i] == 0 or l2[i] == 0 :
            if l1[i] != l2[i] :
                return False
            else : continue
        
        else :
            if ratio == None:
                ratio = l2[i] / l1[i]
            
            elif ratio != l2[i] / l1[i]:
                return False
    
    else : return True
    
def find_collinearity(matrice):
    # Trouver les lignes colinéaires
    n = len(matrice)
    colineaire_dico = {}
    colineaires = []
    
    for i in range(n):
        cond1 = True
        for x in colineaires:
            if x[1]==i:
                cond1 = False
        
        if cond1 :
            for j in range(i + 1, n): 
                    
                if i not in colineaire_dico:
                    colineaire_dico[i] = []
                    
                    if sont_colineaires(matrice[i], matrice[j]):
                        colineaires.append((i, j))  # Les indices des lignes colinéaires
                        colineaire_dico[i].append(j)
                
                elif sont_colineaires(matrice[i], matrice[j]):
                    colineaires.append((i, j))  # Les indices des lignes colinéaires:
                    colineaire_dico[i].append(j)
    
    return colineaire_dico
                    

def matrice_reduite(matrice, dictionnaire):
    
    new_matrice = []
    
    for s in dictionnaire.keys():
        
        new_matrice.append(matrice[s])
    
    return new_matrice

dico = find_collinearity(matrice)

new_mat = matrice_reduite(matrice, dico)

print(matrice)        
print(new_mat) 
print(dico)     


