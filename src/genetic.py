import random
import multiprocessing
import os
from .solver import initWorker, solveWorker, solveInstance, getEpsilon
from amplpy import AMPL
import math

import matplotlib.pyplot as plt
from src.model import mInfrastructure, mTransport

def createRelaxedChromosome(currentInstance, steps=10, workerPool=None, fixedCosts=None):
    
    # 1. Obtener el frente relajado
    transportMin, transportMax, infraMin, infraMax, relaxedParetoX, relaxedParetoY, relaxedPopulation = getEpsilon(
        currentInstance, mTransport, mInfrastructure, steps, relaxed=True
    )
    
    repairedPopulation = []
    print(f"Población relajada: \n {relaxedPopulation}")
    
    # Inicializar el gráfico y graficar los puntos relajados
    plt.figure(figsize=(10, 6))
    plt.scatter(relaxedParetoX, relaxedParetoY, color='blue', alpha=0.6, label='Frente Relajado (Continuo)')
    
    # 2. Reparar los cromosomas
    for individualGenes in relaxedPopulation:
        repairedGenes  = repairChromosome(individualGenes)
        newChromosome = {
            "genes": repairedGenes,
            "alpha": random.uniform(0.0, 1)
        }
        repairedPopulation.append(newChromosome)
        
    print(f"Población reparada: \n {repairedPopulation}")
    
    # 3. Evaluar la población reparada para obtener sus costos y graficarlos
    if workerPool and fixedCosts:
        dummyCache = {}
        # Llamamos a tu función evaluatePopulation para obtener los costos de Transporte e Infra
        repairedCosts, _ = evaluatePopulation(repairedPopulation, dummyCache, 0, fixedCosts, workerPool)
        
        repairedParetoX = []
        repairedParetoY = []
        
        for currentCost in repairedCosts:
            # Filtramos los que fallaron o dieron infinito
            if currentCost is not None and currentCost[0] != float('inf'):
                repairedParetoX.append(currentCost[0])
                repairedParetoY.append(currentCost[1])
                
        plt.scatter(repairedParetoX, repairedParetoY, color='red', marker='x', s=60, label='Frente Reparado (Entero)')
    else:
        print("Advertencia: No se entregó 'workerPool' o 'fixedCosts'. El frente reparado no será evaluado ni graficado.")

    # 4. Configurar y mostrar el gráfico final
    plt.title("Comparación de Frente de Pareto: Relajado vs Reparado")
    plt.xlabel("Costo de Transporte")
    plt.ylabel("Costo de Infraestructura e Inventario")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return repairedPopulation

def createBiasedChromosome(rankedCds, capacities, totalDemand, betaParameter=0.3):
    numCds = len(rankedCds)
    chrom = [0] * numCds
    currentCapacity = 0
    unselectedCds = rankedCds.copy()
    
    while currentCapacity < totalDemand:
        randVal = random.uniform(0.0001, 0.9999)
        selectedIndex = int(math.log(randVal) / math.log(1 - betaParameter))
        selectedIndex = min(selectedIndex, len(unselectedCds) - 1)
        
        chosenCdDict = unselectedCds.pop(selectedIndex)
        chosenCd = chosenCdDict['cd']
        
        chrom[chosenCd] = 1
        currentCapacity += capacities[chosenCd]
        
    #Diversidad
    if random.random() < 0.3:
        closedCds = [i for i in range(numCds) if chrom[i] == 0]
        if closedCds:
            chrom[random.choice(closedCds)] = 1
            
    return {
        "genes": chrom,
        "alpha": random.uniform(0.0, 1)
    }

def rankCds(fixedCosts, capacities, transportCosts, numCds, numClients):
    cdAttractiveness = []
    
    for i in range(numCds):
        avgTransportCost = sum(transportCosts[(i, j)] for j in range(numClients)) / numClients
        
        efficiencyRatio = capacities[i] / (fixedCosts[i] + (avgTransportCost * 50) + 0.0001)
        cdAttractiveness.append({'cd': i, 'ratio': efficiencyRatio})
        
    cdAttractiveness.sort(key=lambda x: x['ratio'], reverse=True)
    return cdAttractiveness

def repairChromosome(chromosome):
    return [1 if (item[1] if isinstance(item, tuple) else item) >= 0.15 else 0 for item in chromosome]

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

def createChromosome(numCds):
    #Se crea aleatoriamente un cromosoma con máximo el 30% de cd abiertos
    toOpen = random.randint(1, int(round(numCds*0.5)))
    #toOpen = random.randint(1, int(round(numCds*0.1)))
    chrom = []
    for i in range(numCds):
        chrom.append(0)
        
    positions = random.sample(range(numCds), toOpen)
    
    for pos in positions:
        chrom[pos] = 1
    
    #Alfa aleatorio
    alpha = random.uniform(0.0, 1)
    chromosoma = {
        "genes": chrom,
        "alpha": alpha
    }

    return chromosoma

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

def crossoverFixedSet(parent1, parent2, dataStr, modelCode, infraMax = 1, transpMax = 1):
    point = random.randint(1, len(parent1["genes"]) - 1)

    fixedSet1 = {}
    for i in range(point, len(parent1["genes"])):
        fixedSet1[i] = parent2["genes"][i]

    fixedSet2 = {}
    for i in range(0, point):
        fixedSet2[i] = parent1["genes"][i]

    alpha1 = random.uniform(0.0, 1)
    alpha2 = random.uniform(0.0, 1)

    aux, aux, child1Genes = solveInstance(dataStr, modelCode, relaxation = False, fixedCds=fixedSet1, maxInfra = infraMax, maxTransp = transpMax, alpha = alpha1)
    aux, aux, child2Genes = solveInstance(dataStr, modelCode, relaxation = False, fixedCds=fixedSet2, maxInfra = infraMax, maxTransp = transpMax, alpha = alpha2)

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

def mutateFixedSet(individual, dataStr, modelCode, freeCdsMin=0.05, freeCdsMax = 0.2, infraMax = 1, transpMax = 1):
    chromLenght = len(individual)
    numToUnfix = random.randint(round(chromLenght*freeCdsMin), round(chromLenght*freeCdsMax))
    unfixedCds = random.sample(range(chromLenght), numToUnfix)

    fixedCds = {}
    for i in range(chromLenght):
        if i not in unfixedCds:
            fixedCds[i] = individual["genes"][i]

    aux,aux, newGenes = solveInstance(dataStr, modelCode, relaxation = False, fixedCds=fixedCds, maxInfra = infraMax, maxTransp = transpMax, alpha = individual["alpha"])
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

def geneticAlgorithm(modelCode, dataStr, numCds, popSize=20, generations=10, mutationRate=0.1, nJobs=2, licenseUuid="", infraMax = 1, transportMax = 1):
    
    fitnessCache = {}
    totalSolverCalls = 0 
    paretoHistory = []
    failed = 0

    #Esto es para sacar los costos fijos, para poder realizar el ajuste de crommosomas
    from amplpy import AMPL
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
        initargs=(modelCode, dataStr, licenseUuid, "outlev=0", transportMax, infraMax)
    )

    print("Creando y evaluando población inicial")

    population = []
    for i in range(popSize):
        beta = random.uniform(0.2, 0.6) 
        population.append(createBiasedChromosome(rankedCds, capacitiesFlat, totalDemand, betaParameter=beta))

    #population = createRelaxedChromosome(dataStr, steps = 10, workerPool= pool,fixedCosts=fixCosts)
    #population = [createChromosome(numCds) for _ in range(popSize)]

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
            
            probabilidad = random.random()
            if probabilidad < 0.4:
                c1, c2 = crossoverFixedSet(p1, p2, dataStr, modelCode,infraMax=infraMax, transpMax=transportMax)
            else:
                c1, c2 = crossover(p1, p2)

            probabilidad = random.random()

            if probabilidad < 0.3:
                c1 = mutateFixedSet(c1, dataStr, modelCode, freeCdsMin=0.05, freeCdsMax = 0.15, infraMax=infraMax, transpMax=transportMax)
                c2 = mutateFixedSet(c2, dataStr, modelCode, freeCdsMin=0.05, freeCdsMax = 0.15, infraMax=infraMax, transpMax=transportMax)
            else:
                #c1 = mutate(c1, mutationRate)
                #c2 = mutate(c2, mutationRate)
                c1 = mutateRanked(c1, rankedCds, mutationRate)
                c2 = mutateRanked(c2, rankedCds, mutationRate)

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