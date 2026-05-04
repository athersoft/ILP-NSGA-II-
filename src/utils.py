import requests
import re
import matplotlib.pyplot as plt

def download_data(url):
    print(f"Descargando archivo .dat desde Gist...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error descargando: {e}")
        return None

def parse_num_cds(ampl_data):
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

def calcularHipervolumen(puntos, minX, maxX, minY, maxY):
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


import re
import numpy as np

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