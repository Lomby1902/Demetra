import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sys import stdout
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import random


class fixed_size_GA():

    def __init__(self,N:100,m:10, crossp=0.7,crossm = 0.7):
        '''
        Parameters:
        N (int) Number of individuals per population (population size).
        m (int): Number of fixed components.
        crossp (float) crossover probability.
        crossm (float) mutation probability.
        '''
        self._N = N
        self._m = m
        self._crossp = crossp
        self._crossm = crossm
        
        self._NEGNRMSEPiqrscorer = make_scorer(self._NEGNRMSEPiqr)
        
        df = pd.read_excel("datasets/Dataset_2.xlsx",sheet_name="preproc")
        self._X = (df.iloc[:,14:]).values
        self._Y = (df.iloc[:,7]).values
        self._n = self._X.shape[1]

        self._create_initial_population()

    def _create_initial_population(self):
        """
        This function create the initial pseudorandom population with
        a fixed size of 1s..

        Parameters:

        Returns:
        Initial population (list): list of bit string consisting of the initial population 
        """
        C = np.zeros((self._N,self._n),dtype=int)
        for i in range(self._N):
            I = np.random.permutation(np.arange(self._n))[:self._m]
            C[i][I] = 1
        self._population = C


    def get_population(self):
        return[individual for individual in self._population]
           
    def _check(self,X,P1,P2):
        '''
        Algorithm used to maintain a costant number of components selected
        after crossover or mutation

        Parameters:
        X (list): offspring index array
        P1 (list): first parent index array
        P2 (list): second parent index array

        Returns:
        X (list): checked offspring index array
        '''


        C = np.zeros(self._n, dtype=int)
        for h in range(len(X)):
            C[X[h]] = 1

        if sum(C) < self._m:
            X_new = np.sort(X)
            A = np.concatenate((P1,P2))
            D = np.setdiff1d(A ,X_new)
            I = np.random.permutation(np.arange(len(D)))
            k = 0
            for h in range(len(X) - 1):
                if X_new[h] == X_new[h+1]:
                    X_new[h] = D[I[k]]
                    k+=1
                    if k > len(D): k = 0

            X = X_new

        return X
    
    def _crossover(self,best_individuals:list):
        '''
        Algorithm for recombination using single point crossover.
        Note: The offspring will replace previous population

        Parameters:
        best_individuals (list): list of the selected individuals for the reproduction

        Returns:
        new_individuals (list): new list containing the new generated individuals
        '''

        new_individuals = np.zeros((len(best_individuals),self._n),dtype=int)
        i = 0
        random.shuffle(best_individuals)
        while i < len(best_individuals):
            O = []
            E = []
            prob = np.random.random()
            if  prob < self._crossp:
                for j in range(self._n):
                    if best_individuals[i][0][j] == 1:
                        O.append(j)
                    if best_individuals[i+1][0][j] == 1:
                        E.append(j)

                x = np.random.randint(self._m)
                part1 = O[:x]
                part2 = E[x:self._m]
                O_new = part1 + part2
                part1 = E[:x]
                part2 = O[x:self._m]
                E_new = part1 + part2
                O_new = self._check(O_new,O,E)
                E_new = self._check(E_new,O,E)

                if np.array_equal(O,O_new):
                    pass
                if np.array_equal(E,E_new):
                    pass
                for h in range(self._m):
                    new_individuals[i][O_new[h]] = 1
                    new_individuals[i+1][E_new[h]] = 1
            else:
                new_individuals[i] = best_individuals[i][0]
                new_individuals[i+1] = best_individuals[i+1][0]
            

            i+=2


        return new_individuals

    def _check_mutation(self,individual):
        '''
        This method check if after mutation the number 
        of 1s is still equal to m. If not it will remove or
        add 1 until the requirement will be respected.

        Paramater:
        individual(list) : individual to check

        Returns:
        new_individual(list) : individual checked
        '''

        somma = sum(individual)
        if np.sum(individual) > self._m:
            new_individual = np.zeros(self._n,dtype=int)
            indices = np.where(individual == 1)[0]
            indices = np.random.permutation(indices)[:self._m]
            new_individual[indices] = 1

        elif np.sum(individual) < self._m:
            new_individual = np.ones(self._n,dtype=int)
            indices = np.where(individual == 0)[0]
            indices = np.random.permutation(indices)[:self._n-self._m]
            new_individual[indices] = 0

        else:
            new_individual = individual
        return new_individual

    def _mutation(self,best_individuals):
        '''
        Algorithm to apply mutation to the new offspring.
        Note: This should be applied after crossover.

        Parameters:
        best_individuals (int): best individuals selected

        Return:
        new_individuals (list) : list of mutated offrsping
        '''

        new_individuals = np.zeros((len(best_individuals),self._n),dtype=int)
        for i in range(len(best_individuals)):
            for j in range(len(best_individuals[i])):
                if np.random.random() < self._crossm:
                     new_individuals[i][j] = not best_individuals[i][j]

            #check mutation
            new_individuals[i]= self._check_mutation(new_individuals[i])
        

        for individual in new_individuals:
            if sum(individual) != self._m:
                pass
        return new_individuals

    def _NEGNRMSEPiqr(self,observed_values, predicted_values):
        '''
        Function to evaluate error of prediction

        Parameters:
        observed_values (list): real values
        predicted_values (list): predicted values

        Returns:
        NEGNRMSEPiqr (float): negative NRMSPEiqr
        '''
        # Calculate RMSEP
        rmsep = np.sqrt(np.mean((observed_values - predicted_values) ** 2))
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = np.percentile(observed_values, 25)
        Q3 = np.percentile(observed_values, 75)

        # Calculate IQR
        IQR = Q3 - Q1


        return -rmsep/IQR
    
    def _fitness_function(self,individual):
        '''
        Fitness function. It is equal to negative NRMSEPiqr evaluated using a cross 
        validated Standard Scaler plus Ridge model with optimal alpha value.

        Parameters:
        individual (list): chromosome

        Returns:
        fitness (float): value of fitness of the selected chromosome
        '''
        X_selected = self._X[:, np.where(individual == 1)[0]]
        alpha = 0.00122457013067159 
        # Model
        ridge = make_pipeline(StandardScaler(),Ridge(alpha=alpha))
        scores = cross_val_score(ridge, X_selected, self._Y, cv=5, scoring=self._NEGNRMSEPiqrscorer)
        
        # Use the mean score as the fitness value (higher is better)
        return np.mean(scores)

    def _ordered_score(self):
        """
        Function to compute an ordered list containing individuals and associated fitness

        Returns:
        best_individuals (list): list of the nbest individuals with fitness score
        """

        fitness_scores = []
        for i,individual in enumerate(self._population):
            fitness_scores.append((i,self._fitness_function(individual)))

        fitness_scores.sort(reverse=True, key=lambda x: x[1])
        
        # Get best nbest individuals with realtive score
        best_individuals = [(self._population[individual[0]], individual[1]) for individual in fitness_scores]

        return best_individuals

    def _fix_population_size(self, new_individuals, best_individuals):
        '''
        This function is used if the size of the produced
        offspring is less than population size. When this happen,
        this function will use elitism, and so it will fill the next
        generation with the best individuals from the previous one
        to guarantee that population size will be constant.
        The population will be updated with the new values.

        Parameters:
        new_individuals(list) :individuals selected for reproduction
        best_individuals(list) : ordered list with best individuals
        '''
        self._population[:new_individuals.shape[0]] = new_individuals
        for i in range(new_individuals.shape[0],self._N):
            self._population[i] = best_individuals[i - new_individuals.shape[0]][0]
  
    def run(self,generations= 50,nbest=10):
        '''
        Function to run the genetic algorithm for the user.

        Parameters:
        new_individuals(list) :List of new generated individuals
        nbest(int) : Number of individuals to select for reproduction


        Returns:
        fitness_history(list) : list containing for each generation the 
                                fitness value of the best individual.
        '''
        self._fitness_history = []

        #Gen 0 score
        best_individuals = self._ordered_score()
        self._fitness_history.append(best_individuals[0])

        total_time = time.time()
        for gen in range(generations):
            start_time = time.time()
            new_individuals = self._crossover(best_individuals[:nbest])
            #new_individuals = self._mutation(new_individuals)
            self._fix_population_size(new_individuals,best_individuals)
            best_individuals = self._ordered_score()
            self._fitness_history.append(best_individuals[0])
            end_time = time.time() - start_time
            print(f"Generation {gen + 1} completed. Fitness: {best_individuals[0][1]}, elapsed time(sec): {end_time}.")

        total_time = start_time - total_time
        print("")
        print("------------------------")
        print("")  
        print(f"Total elapsed time(min): {total_time/60}")
        best = max(self._fitness_history, key= lambda x:x[1])
        print(f"Best individual: {np.where(best[0]==1)[0]} score = {best[1]}")
        plt.plot(np.arange(len(self._fitness_history)),[individual[1] for individual in self._fitness_history])
        plt.title("Fitness over generations")
        plt.ylabel("Fitness")
        plt.xlabel("Generations")
        plt.xticks(np.arange(len(self._fitness_history)))
        plt.show()

        return self._fitness_history


pop_size = 100
fixed_components = 100
crossp = 1
crossm = 0.2
n_generations = 50
n_best = 50
ga = fixed_size_GA(pop_size,fixed_components,crossp,crossm)

ga.run(n_generations ,n_best)



