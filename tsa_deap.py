
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import pandas as pd
import numpy
import statistics 

cities_letter = [0,1,2,3,4,5,6,7]
city_dist =pd.read_csv("TS_Distances_Between_Cities.csv") 
city_dist=city_dist.truncate(after=7)
#print(city_dist.columns)
city_dist=city_dist.drop(['Unnamed: 0'],axis=1)
#print(city_dist)

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("indices", numpy.random.permutation, cities_letter)
toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.indices)
toolbox.register("population", tools.initRepeat, list,toolbox.individual)



def evaluation(individual):
    '''Evaluates an individual by converting it into 
    a list of cities and passing that list to total_distance'''
    total_distance=[]
    for i in range(len(individual)-1):
    	each_distance = city_dist.iloc[individual[i],individual[i+1]]
    	total_distance.append(each_distance)
    #print(sum(total_distance))
    return sum(total_distance),


#def total_distance(tour):
    #return sum(distance(tour[i],tour[i-1])
  #             for i in range(len(tour)))
#def distance(A, B):

#	A=int(A)
#	B=int(B)
	#Do for all cities	
#	return city_dist[A][B]


def main():

    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=200)
    #print(pop)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    
    toolbox.register("evaluate", evaluation)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    CXPB, MUTPB = 0.5, 0.4
    
    print("Start of evolution")

    f1=open("Meenakshi_Anbukkarasu_GA_TS_Info.txt","w+")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("Populaton size %i individuals\r\n" % len(pop))

    

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    gen = 0

    # Begin the evolution
    while gen < 100 and min(fits)>10000:
        # A new generation
        gen = gen + 1

        #f1.write("Populaton Size: %i " % len(pop))       
        f1.write("-- Generation %i --\r\n" % gen)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        #subset=len(offspring)

        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #f1.write("  Evaluated %i individuals\r\n" % len(invalid_ind))
        # The population is entirely replaced by the offspring
        pop[:] = offspring

        subset = len(invalid_ind)
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
     
        f1.write("  Avg %s\r\n" % mean)
        f1.write("  Median %s\r\n" % statistics.median(fits))
        f1.write("  Std %s\r\n" % std)
        f1.write("  Size of the selected subset of the population: %i \r\n" %subset)
        
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s with Fitness %s" % (best_ind, best_ind.fitness.values))

    f1.close()

    f= open("Meenakshi_Anbukkarasu_GA_TS_Result.txt","w+")
    for i in range(len(best_ind)):
        if(best_ind[i] == 0):
            f.write("City %d / London is visited\r\n" %(i+1))
        if(best_ind[i] == 1):
            f.write("City %d / Venice is visited\r\n" %(i+1))
        if(best_ind[i] == 2):
            f.write("City %d / Dunedin is visited\r\n" %(i+1))
        if(best_ind[i] == 3):
            f.write("City %d / Singapore is visited\r\n" %(i+1))
        if(best_ind[i] == 4):
            f.write("City %d / Beijing is visited\r\n" %(i+1))
        if(best_ind[i] == 5):
            f.write("City %d / Phoenix is visited\r\n" %(i+1))
        if(best_ind[i] == 6):
            f.write("City %d / Tokyo is visited\r\n" %(i+1))
        if(best_ind[i] == 7):
            f.write("City %d / Victoria is visited\r\n" %(i+1))
    f.close()

if __name__ == "__main__":
    main()
