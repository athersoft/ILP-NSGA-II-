import requests
import re
import matplotlib.pyplot as plt
import time
import os
import json
import re
import numpy as np

def downloadData(url):
    print(f"Descargando archivo .dat desde Gist...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error descargando: {e}")
        return None

def parseNumCds(ampl_data):
    match = re.search(r'set\s+I\s*:=\s*(.*?);', ampl_data, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        cdList_ids = content.split()
        return len(cdList_ids)
    return 0

def plot_convergence(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, marker='o')
    plt.title("Convergencia GA Híbrido")
    plt.xlabel("Generación")
    plt.ylabel("Costo")
    plt.grid(True)
    plt.show()

def repairChromosome(chromosome):
    return [1 if (item[1] if isinstance(item, tuple) else item) >= 0.15 else 0 for item in chromosome]

def calculateHypervolume(puntos, minX, maxX, minY, maxY):
    if len(puntos) == 0:
        return 0.0
        
    rangoX = maxX - minX if maxX > minX else 1.0
    rangoY = maxY - minY if maxY > minY else 1.0
    
    puntos_norm = []
    for p in puntos:
        nx = (p[0] - minX) / rangoX
        ny = (p[1] - minY) / rangoY
        puntos_norm.append((nx, ny))
        
    sortedPoints = sorted(puntos_norm, key=lambda p: p[0])
    
    hipervolumen = 0.0
    
    refX_norm = 1.0 
    refY_norm = 1.0
    
    for i in range(len(sortedPoints)):
        xx = sortedPoints[i][0]
        yy = sortedPoints[i][1]

        if xx <= refX_norm and yy <= refY_norm:
            if i + 1 < len(sortedPoints):
                nextX = sortedPoints[i + 1][0]
            else:
                nextX = refX_norm
                
            width = nextX - xx
            height = refY_norm - yy

            if width > 0 and height > 0:
                area = width * height
                hipervolumen += area
            
    return hipervolumen

def characterizeInstance(instanceData):
    reporte = []
    reporte.append("--- Análisis de la Instancia ---")
    
    # Función interna para extraer los valores finales de cada parámetro
    def extractValues(paramName):
        match = re.search(rf'param\s+{paramName}\s*:=\s*(.*?);', instanceData, re.DOTALL)
        if not match: return []
        
        tokens = match.group(1).split()
        valuesList = []
        
        # F, Cap, d (1D): id valor -> tomamos cada 2do token
        # TC (2D): i j valor -> tomamos cada 3er token
        step = 3 if paramName == 'TC' else 2
        
        for i in range(step - 1, len(tokens), step):
            try:
                valuesList.append(float(tokens[i]))
            except ValueError:
                pass
        return valuesList

    # Función interna para formatear estadísticas como texto
    def getStats(name, valuesList):
        if not valuesList:
            return f"{name}: No se encontraron datos."
            
        meanValue = np.mean(valuesList)
        stdValue = np.std(valuesList)
        cvValue = (stdValue / meanValue) * 100 if meanValue != 0 else 0
        
        return f"{name}:\n  - Promedio: {meanValue:.2f}\n  - Desv. Estándar: {stdValue:.2f}\n  - Coef. Variación (CV): {cvValue:.2f}%"

    # Extraemos y analizamos los parámetros clave
    fixedCosts = extractValues('F')
    capacities = extractValues('Cap')
    demands = extractValues('d')
    transportCosts = extractValues('TC')

    reporte.append(getStats("Costos Fijos (F)", fixedCosts))
    reporte.append(getStats("Capacidades (Cap)", capacities))
    reporte.append(getStats("Demandas (d)", demands))
    reporte.append(getStats("Costos de Transporte (TC)", transportCosts))
    
    # NUEVO: Análisis de Tensión de la Red (Capacidad Total vs Demanda Total)
    if capacities and demands:
        totalCap = sum(capacities)
        totalDem = sum(demands)
        ratio = totalCap / totalDem if totalDem > 0 else 0
        
        estado = "Muy Ajustada" if ratio < 1.5 else "Ajustada" if ratio < 3 else "Holgada"
        
        reporte.append("\nAnálisis de Tensión:")
        reporte.append(f"  - Demanda Total de la Red: {totalDem:.2f}")
        reporte.append(f"  - Capacidad Total de CDs : {totalCap:.2f}")
        reporte.append(f"  - Ratio (Cap/Dem)        : {ratio:.2f}x ({estado})")

    reporte.append("--------------------------------\n")
    
    # Devolvemos todo como un solo bloque de texto
    return "\n".join(reporte)

def rankCds(fixedCosts, capacities, transportCosts, numCds, numClients):
    cdAttractiveness = []
    
    for i in range(numCds):
        avgTransportCost = sum(transportCosts[(i, j)] for j in range(numClients)) / numClients
        
        efficiencyRatio = capacities[i] / (fixedCosts[i] + (avgTransportCost * 50) + 0.0001)
        cdAttractiveness.append({'cd': i, 'ratio': efficiencyRatio})
        
    cdAttractiveness.sort(key=lambda x: x['ratio'], reverse=True)
    return cdAttractiveness

def generateAndSaveReport(
    instanceName, dataUrl, numCds, characterizationStr,
    useEpsilon, setPoints, epsilonTime, epsilonHypervolume,
    transportMax, infraMax, transportMin, infraMin, paretoX, paretoY,
    popSize, generations, mutationRate, totalCalls, geneticTime,
    mogaHypervolume, paretoFront, mogaQuality
):
    reportLines = []
    reportLines.append("==================================================")
    reportLines.append("        REPORTE DE EJECUCIÓN MULTIOBJETIVO      ")
    reportLines.append("==================================================")
    reportLines.append(f"URL Instancia Evaluada: {dataUrl}")
    reportLines.append(f"Tamaño de Instancia: {numCds} CDs")
    
    reportLines.append("\nCARACTERIZACIÓN DE LA INSTANCIA")
    reportLines.append(characterizationStr.strip())

    reportLines.append("\nRESULTADOS EPSILON-CONSTRAINT")
    if useEpsilon or setPoints:
        reportLines.append(f"Tiempo de ejecución : {epsilonTime:.4f} segundos")
        reportLines.append(f"Hipervolumen        : {epsilonHypervolume:.4f}")
        reportLines.append(f"\nPuntos Lexicográficos (Extremos del Frente):")
        reportLines.append(f"  - Nadir: Transp={transportMax:.2f}, Infra={infraMax:.2f}")
        reportLines.append(f"  - Transp. Mín   : Transp={transportMin:.2f}, Infra={infraMax:.2f}")
        reportLines.append(f"  - Infra. Mín    : Transp={transportMax:.2f}, Infra={infraMin:.2f}")
        
        reportLines.append(f"\nPuntos del Frente ({len(paretoX)} steps):")
        for index in range(len(paretoX)):
            reportLines.append(f"  Punto {index+1}: Transp={paretoX[index]:.2f}, Infra={paretoY[index]:.2f}")
    else:
        reportLines.append("Ejecución omitida (useEpsilon = False).")

    reportLines.append("\nRESULTADOS ALGORITMO GENÉTICO (NSGA-II)")

    reportLines.append(f"Parámetros          : Población={popSize}, Generaciones={generations}, Mutación={mutationRate}")
    reportLines.append(f"Llamadas al Solver  : {totalCalls} evaluaciones exactas")
    reportLines.append(f"Tiempo de ejecución : {geneticTime:.4f} segundos")
    reportLines.append(f"Hipervolumen        : {mogaHypervolume:.4f}")
    
    reportLines.append(f"\nFrente de Pareto Final - {len(paretoFront)} puntos:")
    for index, solutionItem in enumerate(paretoFront):
        chromosomeStr = solutionItem[0]
        transportVal = solutionItem[1]
        infraVal = solutionItem[2]
        reportLines.append(f"  Punto {index+1}: Transp={transportVal:.2f}, Infra={infraVal:.2f} | Cromosoma: {chromosomeStr}")
            
    else:
        reportLines.append("Ejecución omitida (useMoga = False).")

    reportLines.append("\nCOMPARATIVA ESTADÍSTICA")
    if (useEpsilon or setPoints) and epsilonHypervolume > 0:
        reportLines.append(f"Calidad del MOGA vs Exacto : {mogaQuality:.2f}% (Cobertura del Hipervolumen)")
        if geneticTime > 0:
            timeAcceleration = epsilonTime / geneticTime
            reportLines.append(f"Aceleración de Tiempo      : El MOGA fue {timeAcceleration:.2f}x más rápido que Epsilon")
    else:
        reportLines.append("No hay datos suficientes de ambos métodos para comparar.")

    outputFolder = "resultados"
    os.makedirs(outputFolder, exist_ok=True)

    currentTime = time.time()
    fileName = f"Resultados_{instanceName}_{currentTime}.txt"
    filePath = os.path.join(outputFolder, fileName)
    
    with open(filePath, "w", encoding="utf-8") as fileObject:
        fileObject.write("\n".join(reportLines))
        
    print(f"\n*** Los resultados han sido guardados exitosamente en '{filePath}' ***")
    return filePath

def loadInstance(name):

    if not name.endswith(".dat"):
        fileName = f"{name}.dat"
    else:
        fileName = name
        
    folderName = "instances"
    filePath = os.path.join(folderName, fileName)
    
    try:
        with open(filePath, 'r', encoding='utf-8') as fileObject:
            fileContent = fileObject.read()
        return fileContent
    except FileNotFoundError:
        print(f"Error: El archivo '{fileName}' no se encontró en la carpeta '{folderName}'.")
        return None
    except Exception as errorObject:
        print(f"Error inesperado al leer la instancia: {errorObject}")
        return None
    
def loadEpsilonResults(filePath, instanceUrl):
    if not os.path.exists(filePath):
        return None

    try:
        with open(filePath, 'r', encoding='utf-8') as f:
            allResults = json.load(f)
            
        if instanceUrl in allResults:
            print(f" --- Resultados previos encontrados para: {instanceUrl} --- ")
            data = allResults[instanceUrl]
            
            return {
                'time': data['metadata']['executionTime'],
                'hv': data['metadata']['hypervolume'],
                'transMin': data['lexicographicPoints']['infraMax']['transp'],
                'transMax': data['lexicographicPoints']['infraMin']['transp'],
                'infraMin': data['lexicographicPoints']['infraMin']['infra'],
                'infraMax': data['lexicographicPoints']['infraMax']['infra'],
                'paretoX': data['paretoFront']['x'],
                'paretoY': data['paretoFront']['y']
            }
    except Exception as e:
        print(f"Error al cargar resultados previos: {e}")
        
    return None

def loadConfig(configPath):
    with open(configPath, 'r', encoding='utf-8') as file:
        return json.load(file)