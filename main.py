import time
import numpy as np
from src.model import randomInstance, instanceToAmpl, mInfrastructure, mTransport, modelMultiObj
from src.genetic import geneticAlgorithm
from src.utils import  downloadData, parseNumCds, loadInstance, calculateHypervolume, characterizeInstance, generateAndSaveReport, loadEpsilonResults, loadConfig
from src.solver import getEpsilon
from src.graphics import plotAbsoluteFront, plotNormalizedFront
import statistics


if __name__ == "__main__":
    configData = loadConfig("config.json")
    numExperiments = configData["numExperiments"]
    gaParams = configData["gaParams"]
    instancesList = configData["instances"]
    
    licenseUuid = "b84215a6-2e17-4c6f-8d78-2019e0f3c0ff" 
    dataUrl = ""

    for instanceName in instancesList:
        print(f"Ejecución de la instancia: {instanceName}")

        currentInstance = loadInstance(instanceName)
            
        numCds = parseNumCds(currentInstance)
        print(f"Instancia cargada con {numCds} CDs candidatos.")

        previousResults = loadEpsilonResults(f"fronts/{instanceName}.json", instanceName)

        if previousResults is not None:
            print("Frente óptimo cargado")
            
            epsilonTime = previousResults['time']
            epsilonHypervolume = previousResults['hv']
            transportMin = previousResults['transMin']
            transportMax = previousResults['transMax']
            infraMin = previousResults['infraMin']
            infraMax = previousResults['infraMax']
            paretoX = previousResults['paretoX']
            paretoY = previousResults['paretoY']
            setPoints = True
        else:
            print("No hay frente óptimo para esta instancia")

            epsilonHypervolume = -1
            mogaHypervolume = -1
            transportMin = 0
            transportMax = 1
            infraMin = 0
            infraMax = 1
            epsilonTime = 1
            geneticTime = 0
            setPoints = False
            mogaQuality = 1
            paretoX = []
            paretoY = []
        
        if setPoints:
            epsilonPoints = []
            for i in range(len(paretoX)):
                epsilonPoints.append((paretoX[i], paretoY[i]))
                
            epsilonHypervolume = calculateHypervolume(epsilonPoints, transportMin, transportMax, infraMin, infraMax)

        mogaHypervolumeList = []
        mogaQualityList = []
        geneticTimeList = []
        totalCallsList = []
        
        bestParetoFront = []
        bestMogaX = []
        bestMogaY = []
        bestHypervolume = -1

        for expIndex in range(numExperiments):
            print(f"\n  -> Corriendo Experimento {expIndex + 1} de {numExperiments}...")
            
            timeZero = time.time()
            startTime = time.time()
            
            # --------------------------------------------Ejecución NSGA II-----------------------------------------------------#
            paretoFront, totalCalls, paretoHistory = geneticAlgorithm(
                modelCode=modelMultiObj,
                dataStr=currentInstance,
                numCds=numCds,
                popSize=gaParams.get("popSize", 50),
                generations=gaParams.get("generations", 15),
                nJobs=gaParams.get("nJobs", 15),
                licenseUuid=licenseUuid,
                mutationRate=gaParams.get("mutationRate", 0.4),
                transportMax=transportMax,
                infraMax=infraMax,
                solver=gaParams.get("solver", "knitro"),
                relaxedMutation=gaParams.get("relaxedMutation", True), 
                relaxedCrossover=gaParams.get("relaxedCrossover", True),
                gaConfig=gaParams
            )
            
            endTime = time.time()
            geneticTime = endTime - startTime
            
            print(f"  Tiempo NSGA II: {endTime-timeZero:.2f}s")
            print(f"  Frente de pareto: {paretoFront}")
            
            mogaX = [sol[1] for sol in paretoFront]
            mogaY = [sol[2] for sol in paretoFront]

            mogaPoints = []
            for solution in paretoFront:
                mogaPoints.append((solution[1], solution[2]))
            
            mogaHypervolumeCurrent = -1
            mogaQualityCurrent = 1

            if setPoints:
                mogaHypervolumeCurrent = calculateHypervolume(mogaPoints, transportMin, transportMax, infraMin, infraMax)
                if epsilonHypervolume > 0 and mogaHypervolumeCurrent > 0:
                    mogaQualityCurrent = (mogaHypervolumeCurrent / epsilonHypervolume) * 100
            
            if mogaHypervolumeCurrent > bestHypervolume or not bestParetoFront:
                bestHypervolume = mogaHypervolumeCurrent
                bestParetoFront = paretoFront
                bestMogaX = mogaX
                bestMogaY = mogaY
                
            mogaHypervolumeList.append(mogaHypervolumeCurrent)
            mogaQualityList.append(mogaQualityCurrent)
            geneticTimeList.append(geneticTime)
            totalCallsList.append(totalCalls)
            # ---------------------------------------------------------------------------------------------------------------------#

        avgGeneticTime = statistics.mean(geneticTimeList)
        avgTotalCalls = statistics.mean(totalCallsList)
        avgMogaHypervolume = statistics.mean(mogaHypervolumeList)
        avgMogaQuality = statistics.mean(mogaQualityList)

        characterizationStr = characterizeInstance(currentInstance)
        print(characterizationStr)

        print(f"\nHipervolumen Epsilon: {epsilonHypervolume}")
        print(f"Promedio Hipervolumen MOGA: {avgMogaHypervolume}")
            
        if setPoints and epsilonHypervolume > 0:
            print(f"Promedio Calidad pareto MOGA: {avgMogaQuality:.2f}%")

        generateAndSaveReport(
            instanceName=instanceName,
            dataUrl=dataUrl,
            numCds=numCds,
            characterizationStr=characterizationStr,
            useEpsilon=setPoints,
            setPoints=setPoints,
            epsilonTime=epsilonTime,
            epsilonHypervolume=epsilonHypervolume,
            transportMax=transportMax,
            infraMax=infraMax,
            transportMin=transportMin,
            infraMin=infraMin,
            paretoX=paretoX,
            paretoY=paretoY,
            popSize=gaParams.get("popSize", 50),
            generations=gaParams.get("generations", 15),
            mutationRate=gaParams.get("mutationRate", 0.4),
            totalCalls=avgTotalCalls,
            geneticTime=avgGeneticTime,
            mogaHypervolume=avgMogaHypervolume,
            paretoFront=bestParetoFront,
            mogaQuality=avgMogaQuality
        )
        
        if setPoints:
            plotNormalizedFront(paretoX, paretoY, bestMogaX, bestMogaY, transportMin, transportMax, infraMin, infraMax, setPoints, setPoints, numCds)
        else:
            plotAbsoluteFront(paretoX, paretoY, bestMogaX, bestMogaY, transportMin, transportMax, infraMin, infraMax, setPoints, setPoints)