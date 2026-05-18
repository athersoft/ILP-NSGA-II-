import random
import multiprocessing
import os
from .solver import initWorker, solveWorker, solveInstance, getEpsilon
from amplpy import AMPL
import math
import matplotlib.pyplot as plt
from src.model import mInfrastructure, mTransport
from .utils import repairChromosome

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