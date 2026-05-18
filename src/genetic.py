import random
import multiprocessing
import os
from .solver import initWorker, solveWorker, solveInstance, getEpsilon
from amplpy import AMPL
import math
from .iniPop import createBiasedChromosome, createRelaxedChromosome, createChromosome
import matplotlib.pyplot as plt
from src.model import mInfrastructure, mTransport
from .utils import repairChromosome, rankCds
from .mutation import mutateFixedSet, mutate, mutateRanked
from .crossover import crossover, crossoverFixedSet

def dominates(f1, f2):
    strongDominance = f1[0] < f2[0] and f1[1] < f2[1]
    weakDominance = (f1[0] <= f2[0] and f1[1] < f2[1]) or (f1[0] < f2[0] and f1[1] <= f2[1])
    return strongDominance or weakDominance

#Separación de frentes
def fastNonDominatedSort(fitnesses):
    fronts = [[]]
    dominationCount = {}  #Cuántos dominan al individuo
    dominatedSet = {}     #A quiénes domina el individuo 
    ranks = {}            #A qué rango pertenece el individuo 
    
    numIndividuals = len(fitnesses)

    #Rellenar dominationCount y dominatedSet
    for p in range(numIndividuals):
        dominationCount[p] = 0
        dominatedSet[p] = []
        
        for q in range(numIndividuals):
            if p != q:
                #Hace ambas comparaciones con toda la población
                if dominates(fitnesses[p], fitnesses[q]):
                    dominatedSet[p].append(q)
                elif dominates(fitnesses[q], fitnesses[p]):
                    dominationCount[p] += 1
                    
        #Si nadie domina a p, pertenece al primer frente
        if dominationCount[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)

    #Resto de frentes
    i = 0
    while len(fronts[i]) > 0:
        nextFront = []
        for p in fronts[i]:
            for q in dominatedSet[p]:
                dominationCount[q] -= 1
                # Si a q ya no lo domina nadie de los frentes restantes, pasa al siguiente
                if dominationCount[q] == 0:
                    ranks[q] = i + 1
                    nextFront.append(q)
        i += 1
        if len(nextFront) > 0:
            fronts.append(nextFront)
        else:
            break
            
    return fronts, ranks

def calculateCrowdingDistance(frontIndices, fitnesses):

    distances = {i: 0.0 for i in frontIndices}
    numIndividuals = len(frontIndices)
    
    if numIndividuals == 0:
        return distances
        
    if numIndividuals <= 2:
        for i in frontIndices:
            distances[i] = float('inf')
        return distances
    
    
    for m in range(2):
        sortedFront = sorted(frontIndices, key=lambda x: fitnesses[x][m])
        
        distances[sortedFront[0]] = float('inf')
        distances[sortedFront[-1]] = float('inf')
        
        #Se normaliza ,a distancia usando los máximos
        minObj = fitnesses[sortedFront[0]][m]
        maxObj = fitnesses[sortedFront[-1]][m]
        
        if maxObj - minObj == 0:
            continue
            
        # Calculamos la distancia normalizada para los puntos intermedios
        for i in range(1, numIndividuals - 1):
            prevIdx = sortedFront[i - 1]
            nextIdx = sortedFront[i + 1]
            currIdx = sortedFront[i]
            
            # Sumamos la distancia de este objetivo a la distancia total del individuo
            distances[currIdx] += (fitnesses[nextIdx][m] - fitnesses[prevIdx][m]) / (maxObj - minObj)
            
    return distances

#Función principal que conecta con el solver      
def evaluatePopulation(currentPop, fitnessCache, totalSolverCalls, fixCosts, pool):
    
    #Donde se guardarán los costos para reajustarlos en caso de que se repare el cromosoma
    costs = [None] * len(currentPop) 
    
    dataToEvaluate = [] 
    indicesToEvaluate = []

    for i, individual in enumerate(currentPop):
        ind = individual["genes"]
        alphaValue = individual["alpha"] 
        chromKey = tuple(ind)
        
        #Se usa la caché para reducir las llamadas
        if chromKey in fitnessCache:
            costs[i] = fitnessCache[chromKey]
        else:
            dataToEvaluate.append((ind, alphaValue)) 
            indicesToEvaluate.append(i)

    if dataToEvaluate:
        #Se utiliza la pool de procesos paralelos
        results = pool.map(solveWorker, dataToEvaluate)
        totalSolverCalls += len(dataToEvaluate)
        
        for idx, result in zip(indicesToEvaluate, results):
            cTransp, cInfra, demands = result
            ind = currentPop[idx]["genes"]
            
            #Cuando no se puede resolver o es infactible, las pool devuelven infinito
            if demands and cTransp != float('inf'):
                for i in range(len(ind)):
                    #Verificar si hay cd abiertos con 0 demanda asignada
                    if ind[i] == 1: 
                        demandaActual = demands[i][1]
                        if demandaActual == 0: 
                            ind[i] = 0
                            cInfra -= fixCosts[i][1]
                            
            chromKey = tuple(ind)
            fitnessCache[chromKey] = (cTransp, cInfra)
            costs[idx] = (cTransp, cInfra)

    return costs, totalSolverCalls

#Selección de torneo
def tournamentSelection(population, ranks, distances, k=2):
    #2 sujetos aleatorios
    selectedIndices = random.sample(range(len(population)), k)
    
    bestIdx = selectedIndices[0]
    for i in range(1, k):
        currIdx = selectedIndices[i]
        
        if ranks[currIdx] < ranks[bestIdx]:
            bestIdx = currIdx
        elif ranks[currIdx] == ranks[bestIdx]:
            if distances[currIdx] > distances[bestIdx]:
                bestIdx = currIdx
                
    return population[bestIdx]

def selectOperator(operatorConfig):
    methods = [item["method"] for item in operatorConfig]
    probabilities = [item["prob"] for item in operatorConfig]
    return random.choices(methods, weights=probabilities, k=1)[0]

def geneticAlgorithm(modelCode, dataStr, numCds, popSize=20, generations=10, mutationRate=0.1, nJobs=2, licenseUuid="", infraMax = 1, transportMax = 1, solver = "gurobi", relaxedMutation = True, relaxedCrossover = True, gaConfig=None):
    
    fitnessCache = {}
    totalSolverCalls = 0 
    paretoHistory = []
    failed = 0

    if gaConfig is None:
        gaConfig = {}

    gaConfig.setdefault("initialization", [{"method": "biased", "prob": 1.0}])
    gaConfig.setdefault("crossover", [{"method": "standard", "prob": 1.0}])
    gaConfig.setdefault("mutation", [{"method": "ranked", "prob": 1.0}])

    #Esto es para sacar los costos fijos, para poder realizar el ajuste de cromosomas
    tempAmpl = AMPL()
    tempAmpl.eval("reset;")
    tempAmpl.eval(modelCode)
    tempAmpl.eval(dataStr)
    fDict = tempAmpl.getParameter("F").getValues().toDict()
    capDict = tempAmpl.getParameter("Cap").getValues().toDict()
    dDict = tempAmpl.getParameter("d").getValues().toDict()
    tcDict = tempAmpl.getParameter("TC").getValues().toDict()

    numClients = len(dDict)
    totalDemand = sum(dDict.values())

    fixCostsFlat = [fDict[i] for i in range(numCds)]
    capacitiesFlat = [capDict[i] for i in range(numCds)]
    
    rankedCds = rankCds(fixCostsFlat, capacitiesFlat, tcDict, numCds, numClients)
    fixCosts = tempAmpl.getParameter("F").getValues().toList()

    #Inicializar el Pool del multiprocesing
    pool = multiprocessing.Pool(
        processes=nJobs,
        initializer=initWorker,
        initargs=(modelCode, dataStr, licenseUuid, "outlev=0", transportMax, infraMax, solver)
    )

    print("Creando y evaluando población inicial")

    population = []
    while len(population) < popSize:
        initMethod = selectOperator(gaConfig["initialization"])
        if initMethod == "biased":
            betaParam = random.uniform(0.2, 0.6) 
            population.append(createBiasedChromosome(rankedCds, capacitiesFlat, totalDemand, betaParameter=betaParam))
        elif initMethod == "relaxed":
            relaxedPopList = createRelaxedChromosome(dataStr, steps=10, workerPool=pool, fixedCosts=fixCosts)[0]
            population.extend(relaxedPopList)
        elif initMethod == "random":
            population.append(createChromosome(numCds))
    population = population[:popSize]

    fitnesses, totalSolverCalls = evaluatePopulation(population, fitnessCache, totalSolverCalls, fixCosts, pool)  

    #Se calcula tanto los frentes como las distancias para la población inicial
    fronts, ranks = fastNonDominatedSort(fitnesses)
    distances = {}
    for front in fronts:
        frontDistances = calculateCrowdingDistance(front, fitnesses)
        distances.update(frontDistances)

    #Ciclo principal
    for gen in range(generations):
        print(f"Generación {gen} de {generations}")
        offspringPopulation = [] #La población de hijos la guardo a parte

        #Se guarda la tupla para comprobar duplicados
        existingGenotypes = set()
        for ind in population:
            genes_tupla = tuple(ind["genes"])
            existingGenotypes.add(genes_tupla) 

        #Generación de hijos
        while len(offspringPopulation) < popSize:
            p1 = tournamentSelection(population, ranks, distances)
            p2 = tournamentSelection(population, ranks, distances)
            
            crossMethod = selectOperator(gaConfig["crossover"])
            if crossMethod == "fixedSet":
                c1, c2 = crossoverFixedSet(p1, p2, dataStr, modelCode, infraMax=infraMax, transpMax=transportMax, solver=solver, relaxation=relaxedCrossover)
            else:
                c1, c2 = crossover(p1, p2)

            mutMethod = selectOperator(gaConfig["mutation"])
            if mutMethod == "fixedSet":
                c1 = mutateFixedSet(c1, dataStr, modelCode, freeCdsMin=0.05, freeCdsMax=0.15, infraMax=infraMax, transpMax=transportMax, solver=solver, relaxation=relaxedMutation)
                c2 = mutateFixedSet(c2, dataStr, modelCode, freeCdsMin=0.05, freeCdsMax=0.15, infraMax=infraMax, transpMax=transportMax, solver=solver, relaxation=relaxedMutation)
            elif mutMethod == "ranked":
                c1 = mutateRanked(c1, rankedCds, mutationRate)
                c2 = mutateRanked(c2, rankedCds, mutationRate)
            else:
                c1 = mutate(c1, mutationRate)
                c2 = mutate(c2, mutationRate)

            #Evitar que salgan cromosomas ya existentes
            c1Tuple = tuple(c1["genes"])
            if c1Tuple not in existingGenotypes:
                offspringPopulation.append(c1)
                existingGenotypes.add(c1Tuple)
                failed = 0
            else:
                failed += 1

            if len(offspringPopulation) < popSize:
                c2_tuple = tuple(c2["genes"])
                if c2_tuple not in existingGenotypes:
                    offspringPopulation.append(c2)
                    existingGenotypes.add(c2_tuple)
                    failed = 0
                else:
                    failed += 1

            #Si más de 10 veces seguidas no sale un hijo nuevo, se genera un cromosoma aleatorio
            if failed > 10:
                randomChild = createChromosome(numCds)
                random_tuple = tuple(randomChild["genes"])
                if random_tuple not in existingGenotypes:
                    offspringPopulation.append(randomChild)
                    existingGenotypes.add(random_tuple)
                failed = 0
            
            offspringPopulation.extend([c1, c2])
            
        offspringPopulation = offspringPopulation[:popSize]
        #Se evalúan los hijos
        offspringFitnesses, totalSolverCalls = evaluatePopulation(offspringPopulation, fitnessCache, totalSolverCalls, fixCosts, pool)
        
        #Se unen todos
        combinedPopulation = population + offspringPopulation
        combinedFitnesses = fitnesses + offspringFitnesses
        
        #Se sacan los frentes para toda la población
        combinedFronts, combinedRanks = fastNonDominatedSort(combinedFitnesses)
        
        newPopulation = []
        newFitnesses = []
        
        for front in combinedFronts:
            frontDistances = calculateCrowdingDistance(front, combinedFitnesses)
            
            if len(newPopulation) + len(front) <= popSize:
                for idx in front:
                    newPopulation.append(combinedPopulation[idx])
                    newFitnesses.append(combinedFitnesses[idx])

            else:
                espacioRestante = popSize - len(newPopulation)
                #Se ordena la población según su distancia y así se rellena el espacio de población restante
                sortedFront = sorted(front, key=lambda idx: frontDistances[idx], reverse=True)
                
                for i in range(espacioRestante):
                    idx = sortedFront[i]
                    newPopulation.append(combinedPopulation[idx])
                    newFitnesses.append(combinedFitnesses[idx])
                break
                
        population = newPopulation
        fitnesses = newFitnesses
        
        #Se recalculan los frentes y distancias
        fronts, ranks = fastNonDominatedSort(fitnesses)
        distances = {}
        for front in fronts:
            frontDistances = calculateCrowdingDistance(front, fitnesses)
            distances.update(frontDistances)
            
        paretoFront = []
        for idx in fronts[0]:
            paretoFront.append((population[idx], fitnesses[idx][0], fitnesses[idx][1]))
        paretoHistory.append(paretoFront)

        print(f"Tamaño frente actual: {len(paretoFront)} \n")

    pool.close()
    pool.join()
    
    finalPareto = []
    seen = set()
    
    for idx in fronts[0]:
        costoTransp = fitnesses[idx][0]
        costoInfra = fitnesses[idx][1]
        
        coordenada = (round(costoTransp, 2), round(costoInfra, 2))
        
        if coordenada not in seen:
            seen.add(coordenada)
            finalPareto.append((population[idx], costoTransp, costoInfra))

    print(f"Tamaño frente actual: {len(finalPareto)} /n")

    return finalPareto, totalSolverCalls, paretoHistory
         
    #return finalPareto, totalSolverCalls, paretoHistory