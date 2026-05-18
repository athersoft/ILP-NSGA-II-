import random
from.utils import repairChromosome
from .solver import solveInstance

def crossoverFixedSet(parent1, parent2, dataStr, modelCode, infraMax = 1, transpMax = 1, solver = "gurobi", relaxation = True):
    point = random.randint(1, len(parent1["genes"]) - 1)

    fixedSet1 = {}
    for i in range(point, len(parent1["genes"])):
        fixedSet1[i] = parent2["genes"][i]

    fixedSet2 = {}
    for i in range(0, point):
        fixedSet2[i] = parent1["genes"][i]

    alpha1 = random.uniform(0.0, 1)
    alpha2 = random.uniform(0.0, 1)

    aux, aux, child1Genes = solveInstance(dataStr, modelCode, relaxation = relaxation, fixedCds=fixedSet1, maxInfra = infraMax, maxTransp = transpMax, alpha = alpha1, solver = solver)
    aux, aux, child2Genes = solveInstance(dataStr, modelCode, relaxation = relaxation, fixedCds=fixedSet2, maxInfra = infraMax, maxTransp = transpMax, alpha = alpha2, solver = solver)

    child1Genes = repairChromosome(child1Genes)
    child2Genes = repairChromosome(child2Genes)

    child1 = {"genes": child1Genes, "alpha": alpha1}
    child2 = {"genes": child2Genes, "alpha": alpha2}
    return child1, child2

def crossover(parent1, parent2): #El crossover más básico
    point = random.randint(1, len(parent1["genes"]) - 1)
    child1Genes = parent1["genes"][:point] + parent2["genes"][point:]
    child2Genes = parent2["genes"][:point] + parent1["genes"][point:]
    #childAlpha = (parent1["alpha"] + parent2["alpha"]) / 2.0
    child1 = {"genes": child1Genes, "alpha": random.uniform(0.0, 1)}
    child2 = {"genes": child2Genes, "alpha": random.uniform(0.0, 1)}
    return child1, child2