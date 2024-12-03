# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:01:50 2024

@author: ouiam
"""

import numpy as np
import random

def compareP1betterthanP2(M,P1,P2):
  r1, s1 = fobj(M,P1) #on récupère les deux objectifs pour le pattern P1
  r2, s2 = fobj(M,P2) #on récupère les deux objectifs pour le pattern P2
  if r1 != r2:        #on traite les objectifs de façon lexicographique :
      return r1 < r2  # d'abord les valeurs du rang, et si celles-ci sont égales
  return s1 < s2      # alors on prend en compte la valeur de la + petite valeur singulière

def fobj(M,P,tol=1e-14):
  sing_values = np.linalg.svd(P*np.sqrt(M), compute_uv=False) # Calcul des valeurs singulières de la matrice P.*sqrt(M)
  ind_nonzero = np.where(sing_values > tol)[0]                # indices des valeurs > tolérance donnée
  return len(ind_nonzero), sing_values[ind_nonzero[-1]]       # on retourne objectif1=rang et objectif2=plus petite val sing. non-nulle


def genetic_algorithm(M, taille_selectedparent=20,population_size= 500, num_generations=50, crossover_rate=0.8, mutation_rate=0.1, taux_croisement=0.8, proba_mutation = 0.5):
    # Fonction pour évaluer un individu
    def evaluate(M, P):
        rank, smallest_singular = fobj(M, P)
        return rank, smallest_singular

    # Générer une matrice \( P \) aléatoire
    def generate_random_pattern(M):
        return np.random.choice([-1, 1], size=M.shape)

    # Initialiser la population
    def initialize_population(pop_size, M):
        return [generate_random_pattern(M) for _ in range(pop_size)]

    # Sélection par tournoi 
    def tournament_selection(population, taille_selectedparent):
        infirst = np.random.randint(0,len(population))
        first = fobj(M, population[infirst])
        selected = [] # liste des parents 
        selected.append(population[infirst])
        best = population[infirst]
        
        for i,el in enumerate(population):
            if compareP1betterthanP2(M, el, best) and len(selected)< taille_selectedparent:      
                best =el
                ibest =i
                selected.append(el)
                population.pop(i)
        if len(selected) < taille_selectedparent :
            while len(selected) < taille_selectedparent :
                c = np.random.randint(0,len(population))
                if abs(fobj(M, population[c])[0]-fobj(M, population[ibest])[0]) < 8:
                    selected.append(population[c])
        return selected

    # Croisement (échanger des lignes entre deux matrices)
    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            # Générer un masque binaire aléatoire de la même forme que les parents
            mask = np.random.randint(0, 2, size=parent1.shape, dtype=bool)
            
            # Créer l'enfant en combinant les valeurs des deux parents selon le masque
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        
        # Si aucun croisement n'a lieu, retourner une copie du parent1
        return parent1.copy(), parent2.copy()

    # Mutation (changer aléatoirement le signe d'une entrée)
    
    def mutate(individual,m):
        # m est la proportion de nombres à changer 
        m = int (m*individual.shape[0]*individual.shape[1])
        for _ in range(m):
            if np.random.rand() < mutation_rate:
                i, j = np.random.randint(0, individual.shape[0]), np.random.randint(0, individual.shape[1])
                individual[i, j] *= -1
        return individual
    
    def create_couples(parents):
        # Mélanger les parents
        random.shuffle(parents)
        
        # Vérifier si le nombre de parents est impair
        if len(parents) % 2 != 0:
            # Ajouter un parent au hasard pour compléter
            random_parent = random.choice(parents)
            parents.append(random_parent)
        
        # Former les couples (associer deux par deux)
        couples = [(parents[i], parents[i + 1]) for i in range(0, len(parents), 2)]
        
        return couples
    
    def croisement(couples):
        children = []
        # Calculer le nombre de couples à croiser
        num_couples_to_cross = int(taux_croisement * len(couples))
        selected_couples = random.sample(couples, num_couples_to_cross)
        
        for couple in selected_couples:
            parent1, parent2 = couple
            # Croiser les deux parents pour produire deux enfants
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)  # Variante croisée
            children.append(child1)
            children.append(child2)
        
        return children
    
    def mutate_children(children):
        num_children_to_mutate = int(proba_mutation * len(children))
        selected_indices = random.sample(range(len(children)), num_children_to_mutate)
        
        # Copier les enfants pour éviter les effets de bord si nécessaire
        mutated_children = children.copy()
        
        for idx in selected_indices:
            mutated_children[idx] = mutate(mutated_children[idx])  # Remplacer avec le mutant
        
        return mutated_children
    
    # Initialiser la population
    population = initialize_population(population_size, M)
    parents = tournament_selection(population,taille_selectedparent)
    couples = create_couples(parents)
    children = croisement(couples)
    mutated_children = mutate_children(children)
    
    return True


def lecture_fichier(path):
    with open(path, 'r') as fin:  # ouverture du fichier en mode lecture
        m, n = map(int, fin.readline().rstrip().split())  # lecture des dimensions m et n
        data = []  # initialisation d'une liste pour stocker les matrices
        # Lecture des lignes suivantes contenant les éléments de la matrice
        for _ in range(m):
            ligne = list(map(float, fin.readline().rstrip().split()))  
            data.append(ligne)  
    return np.array(data)  

M = lecture_fichier("exempleslide_matrice.txt")
z =genetic_algorithm(M)

