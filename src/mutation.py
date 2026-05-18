import random
from.utils import repairChromosome
from .solver import solveInstance

def mutateFixedSet(individual, dataStr, modelCode, freeCdsMin=0.05, freeCdsMax = 0.2, infraMax = 1, transpMax = 1, solver = "gurobi", relaxation = True):
    chromLenght = len(individual)
    numToUnfix = random.randint(round(chromLenght*freeCdsMin), round(chromLenght*freeCdsMax))
    unfixedCds = random.sample(range(chromLenght), numToUnfix)

    fixedCds = {}
    for i in range(chromLenght):
        if i not in unfixedCds:
            fixedCds[i] = individual["genes"][i]

    aux,aux, newGenes = solveInstance(dataStr, modelCode, relaxation = relaxation, fixedCds=fixedCds, maxInfra = infraMax, maxTransp = transpMax, alpha = individual["alpha"], solver = solver)
    newGenes = repairChromosome(newGenes)

    newIndividual = {
        "genes": newGenes, 
        "alpha": individual["alpha"]
    }

    return newIndividual

def mutate(individual, mutationRate=0.05): # Mutación básica
    newGenes = individual["genes"][:] 
    
    for i in range(len(newGenes)):
        if random.random() < mutationRate:
            newGenes[i] = 1 - newGenes[i] 
            
    if sum(newGenes) == 0:
         newGenes[random.randint(0, len(newGenes) - 1)] = 1
         
    newIndividual = {
        "genes": newGenes, 
        "alpha": individual["alpha"]
    }

    return newIndividual

def mutateRanked(individual, ranking, mutationRate=0.05): # Mutación básica
    newGenes = individual["genes"][:] 
    totalRatio = 0
    cdRatio = {}
    for item in ranking:
        totalRatio += item['ratio']
        cdRatio[item['cd']] = item['ratio']
        
    avgRanking = totalRatio / len(ranking)

    for i in range(len(newGenes)):
        if random.random() < mutationRate:
            if newGenes[i] == 1:
                if cdRatio[i] >= avgRanking:
                    if random.random() >= 0.5:
                        newGenes[i] = 0
                else:
                    newGenes[i] = 0
            else:
                newGenes[i] = 1
                
    if sum(newGenes) == 0:
         newGenes[random.randint(0, len(newGenes) - 1)] = 1
         
    newIndividual = {
        "genes": newGenes, 
        "alpha": individual["alpha"]
    }

    return newIndividual