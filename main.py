import time
import sys
import numpy as np
from src.model import AMPL_MODEL_CODE, randomInstance, instanceToAmpl, mInfrastructure, mTransport, modelMultiObj
from src.genetic import geneticAlgorithm
from src.utils import download_data, parse_num_cds, plot_convergence, calcularHipervolumen, characterizeInstance
from src.solver import getEpsilon
import matplotlib.pyplot as plt

#Instancias
#Generadas por diferentes modelos LLM 

instancesChatGPT = [
    "https://gist.githubusercontent.com/athersoft/c6baed29465f509c315c2f5fa7db93b4/raw/0934f2edc08cd275e8ac98872e6e1d4cc13cb003/80x40-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/60304f8af5b3dfc33cf62094a0cc78d6/raw/c90d6a6a62cb5313862852bc321267595ec10caf/100x50-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/6bd1bee9640084322d0f19f1764e4124/raw/fd4b2123efbb389f476ff5f2dc8bdf5583d92fdd/120x60-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/00335c96a7ff7e52a013910b5d657091/raw/ab6c7eb89824c03233493c0b5a4eb95d721062c5/140x70-chatgpt.dat",
    "https://gist.githubusercontent.com/athersoft/f45eca17f1a5f696d11725d08ed4fdaf/raw/94c125e1f576214c16e04a216d00b153c919584b/200x100-chatgpt"
]
instancesGrok = [
    "https://gist.githubusercontent.com/athersoft/3e4fdb3ee806d5cca5c2c1952e1de007/raw/acc503904e7972867591104ff55240c6ddb2dcdb/80x40-grok",
    "https://gist.githubusercontent.com/athersoft/39e316457aa8b8eb03b51ebae423f316/raw/97ea051b3c129ff0c6aeffad92ef8da34cd5693f/100x50-grok.dat",
    "https://gist.githubusercontent.com/athersoft/6daf6a7b4ac2062662601f83e1d2d2bd/raw/96468a3c756cb9d578e5e68c8b6c62d88c9addac/120x60-grok.dat",
    "https://gist.githubusercontent.com/athersoft/9e83d347c7c6516779da64f686573c18/raw/01f61ef6960ee5ab50d464aeb9a7e90b80a4227f/140x70-grok.dat"
]

instancesGemini = [
    "https://gist.github.com/athersoft/e0bfbcdc2bf4beda0ba81daeb87b8a2d/raw/ec7e78f81a177649f83e1fead4054adab51357a0/80x40-gemini.dat",
    "https://gist.github.com/athersoft/b3ce8c66ce3c51e174d81a7ca9eaefd9/raw/916f23718c276601df5a0631c1c59421470767b9/100x50-gemini.dat",
    "https://gist.githubusercontent.com/athersoft/3853f927779746cb3b8fb8650b8ff4d3/raw/c49a3b1d259d99d38b8c14daa33c6672d403924b/120x60-gemini.dat",
    "https://gist.githubusercontent.com/athersoft/1b2d3540308e38df4cd8cbaf28348593/raw/2f111b387f7d51e3d850ac1905e8d6f72b07d4b3/140x70-gemini.dat",
    "https://gist.github.com/athersoft/1a26d2dfe533bf2b31fcda682d1b82e7/raw/435db55d97f212e22d8a82bf1d4de3afec6fcd14/100x200-gemini.dat"
]

instancesDeepseek = [
    "https://gist.github.com/athersoft/da76049ae985f515cf3b9759083d6f6d/raw/b85800b5ff3c5a9754cd6af95113e49d2b6c98b9/80x40-deepseek.dat",
    "https://gist.github.com/athersoft/383e7ddf48dcf0d51af9ab5bec757eae/raw/a59a7a3fd7a7cc5b1b610a6b8a9cc5384928e791/100x50-deepseek.dat",
    "https://gist.github.com/athersoft/5544cef94c9382246010c575ff64e8d1/raw/dabe96f65671e4ac6d278d8fe6fe1f51c822b10d/120x60-deepseek.dat",
    "https://gist.github.com/athersoft/63415c7f2205b1c61129ebc1ee3cfcd8/raw/e1635522259d8076bad99dcc0865d11de6b01c96/140x70-deepseek.dat"
]

instancesSpecial = [
    "https://gist.githubusercontent.com/athersoft/ae222648b85aa417c53a841a3e39eac7/raw/6afa7e71baf4a75951bc1dab042cf89ce6ffbc4e/inventarioAbsurdo",
    "https://gist.githubusercontent.com/athersoft/61aa11e8d3cef6584417439e5fcc4808/raw/bc28fbcea9b088c0abec974fa5537a6e02d9da98/infraestructuraProhibitiva",
    "https://gist.githubusercontent.com/athersoft/d3a58f54fc61ad124e75884ae5595a32/raw/5fcbb5c1eb832f3d1ffee9fb85280e196f3da5b4/demandaExtrema",
    "https://gist.githubusercontent.com/athersoft/87326bd12029819eb826b9cd3db07808/raw/86c4880ef37dc7e11d2049ec5ccfdb85dfb8c650/capacidadRestringida",
    "https://gist.githubusercontent.com/athersoft/259c57976bd4ff835394be1b0f91aae0/raw/f10a2f6c1ba13520ecfebceab8478f03178e08b8/altaDispersion"
]

topologicos = [
    "https://gist.githubusercontent.com/athersoft/bf02498ff184b433148c77bdf18f8960/raw/10cf7213cb17f488affe614031ff4494892cf350/15x30_topologico",
    "https://gist.githubusercontent.com/athersoft/118f95592d497f953e4f8ef3ee8b9d8b/raw/f734c167ba3d9ddbe6a54c3c7add84d93508b96e/25x50_topologico",
    "https://gist.githubusercontent.com/athersoft/b6eb4abb0ea718ea6ad153d376764e32/raw/d7fea0bbb8efc61f8b533262c2175c5991a10cf5/40x80_topologico",
    "https://gist.githubusercontent.com/athersoft/b738c1d009151e4c881beca1a78bbcdb/raw/d46c9ab38337d3618b1e67eb8a2868c12592d3e3/50x100_topologico"
]


instanciaPaper = "https://gist.githubusercontent.com/athersoft/2dcb176d505a41cffdbcc568682576b5/raw/ac9331d7f6fcecf3fa9b97ca41b0e9d6b1f0b889/instanciaPaper"

#Pos 0: 80x40
#Pos 1: 100x50
#Pos 2: 120x60
#Pos 3: 140x70
#pos 4: 200x100 (Solo ChatGPT y Gemini)

# Configuración algoritmo genético
POP_SIZE = 50   #Tamaño de población    
GENERATIONS = 15 #Número de generaciones
N_CORES = 15       #Número de nucleos
MUTATION = 0.4
LICENSE_UUID = "b84215a6-2e17-4c6f-8d78-2019e0f3c0ff" 
ITERATIONS = 30
EPSILON = False
MOGA = True
STEPS = 10

if __name__ == "__main__":

    values = []
    times = []
    gaps = []
    solverCalls = []
    solutionList = []

    DATA_URL = topologicos[2] #Acá se selecciona la instancia que se evaluará
    
    currentInstance = download_data(DATA_URL)
    if not currentInstance:
        sys.exit(1) 
    nombreInstancia = DATA_URL.split('/')[-1]
    num_cds = parse_num_cds(currentInstance)
    print(f"Instancia cargada con {num_cds} CDs candidatos.")

    startTime = time.time()
    hipervolumenEpsilon = -1
    hipervolumenMoga = -1
    transportMax = 1
    infraMax = 1
    epsilonTime = 1
    geneticTime = 0
    setPoints = False
    calidadMoga = 1
    transportMin = 1
    transportMax = 1
    #transportMin, transportMax, infraMin, infraMax, paretoX, paretoY = 1678832000, 2029291000, 2504844.165146635, 5888023.670072044, [1678832000, 1679233000, 1682573000, 1686767000, 1692334000, 1701656000, 1717396000, 1721055000, 1756689000, 1760385000, 1773399000, 1794524000, 1831984000, 1873241000, 1912056000, 1949982000, 1990027000, 2029291000], [5888023.670072044, 5487517.776199848, 5136161.858950822, 4760297.168430241, 4384388.334549649, 4008375.9758833256, 3777715.1227779817, 3587222.813942907, 3299928.6447103084, 3256661.8329078364, 2876813.146142893, 2846777.3929949026, 2714025.6359970546, 2627904.091968918, 2622368.819028267, 2587474.410950005, 2543925.860084878, 2504844.165146635]
    
    #transportMin, transportMax, infraMin, infraMax, paretoX, paretoY = 1059580000, 1265074000, 1497981.4877616402, 3863439.838475446, [1059580000, 1059814000, 1060472000, 1062793000, 1072028000, 1075280000, 1081823000, 1085941000, 1091368000, 1104767000, 1125816000, 1150218000, 1169330000, 1172589000, 1174965000, 1265074000], [3863439.8384754476, 3600611.13284058, 3320672.074953132, 3047349.8478229414, 2812026.215677583, 2549296.310301116, 2394242.7364619314, 2271348.7657680884, 2023638.8990313748, 1864742.6237267565, 1846873.0888410378, 1830942.884754024, 1758639.8735143289, 1740278.378461626, 1524768.1310068604, 1497981.4877616533]    
    #epsilonTime = 66.8244

    plt.figure(figsize=(10, 6))
    if(EPSILON or setPoints):
        if(setPoints == False):
            transportMin, transportMax,infraMin, infraMax, paretoX, paretoY = getEpsilon(currentInstance,mTransport, mInfrastructure, STEPS)
            epsilonTime = time.time() - startTime
        
        plt.plot(paretoX, paretoY, color='green')
        plt.scatter(paretoX, paretoY, color='green')
        plt.scatter([transportMin, transportMax], [infraMax, infraMin], c=['blue', 'red'], zorder=5)
        epsilonPoints = []
        for i in range(len(paretoX)):
            epsilonPoints.append((paretoX[i], paretoY[i]))
            
        hipervolumenEpsilon = calcularHipervolumen(epsilonPoints, transportMin, transportMax, infraMin, infraMax)

    if(MOGA):
        t0 = time.time()
        paretoHistory = []
        
        startTime = time.time()
        paretoFront, totalCalls, paretoHistory = geneticAlgorithm(
            modelCode=modelMultiObj,
            dataStr=currentInstance,
            numCds=num_cds,
            popSize=POP_SIZE,
            generations=GENERATIONS,
            nJobs=N_CORES,
            licenseUuid=LICENSE_UUID,
            mutationRate = MUTATION,
            transportMax=transportMax,
            infraMax=infraMax
        )
        
        endTime = time.time()
        geneticTime = endTime - startTime
        print(f"Tiempo MOGA: {endTime-t0}")
        #print(f"Frente de pareto (Rank 2): {paretoRank2}")
        print(f"Frente de pareto: {paretoFront}")
        mogaX = [sol[1] for sol in paretoFront]
        mogaY = [sol[2] for sol in paretoFront]

        #rank2X = [sol[1] for sol in paretoRank2]
        #rank2Y = [sol[2] for sol in paretoRank2]

        for front in paretoHistory:
            xx = []
            yy = []
            for solucion in front:
                xx.append(solucion[1])
                yy.append(solucion[2])
            #plt.scatter(xx, yy, color='black', alpha=0.1)

        #plt.scatter(rank2X, rank2Y, color='orange', edgecolor='black', zorder=3, label='Frente Rank 2', alpha=0.5)
        plt.scatter(mogaX, mogaY, color='purple', edgecolor='black', zorder=4, label='Frente Final (MOGA)', alpha = 0.5)

        mogaPoints = []
        for solucion in paretoFront:
            mogaPoints.append((solucion[1], solucion[2]))
        
        if EPSILON or setPoints:
            hipervolumenMoga = calcularHipervolumen(mogaPoints, transportMin, transportMax, infraMin, infraMax)
    
    characterizeInstance(currentInstance)


    caracterizacionStr = characterizeInstance(currentInstance)

    print(caracterizacionStr)

    print(f"\nHipervolumen Epsilon: {hipervolumenEpsilon}")
    print(f"Hipervolumen MOGA: {hipervolumenMoga}")
        
    if hipervolumenEpsilon > 0 and hipervolumenMoga > 0:
        calidadMoga = (hipervolumenMoga/hipervolumenEpsilon) * 100
        print(f"Calidad pareto MOGA: {calidadMoga:.2f}%")



    print(f"\nHipervolumen Epsilon: {hipervolumenEpsilon}")
    print(f"Hipervolumen MOGA: {hipervolumenMoga}")
        
    if hipervolumenEpsilon > 0 and hipervolumenMoga > 0:
        print(f"Calidad pareto MOGA: {(hipervolumenMoga/hipervolumenEpsilon) * 100}")

    ###################################################################

    reporte = []
    reporte.append("==================================================")
    reporte.append("         REPORTE DE EJECUCIÓN MULTIOBJETIVO       ")
    reporte.append("==================================================")
    reporte.append(f"URL Instancia Evaluada: {DATA_URL}")
    reporte.append(f"Tamaño de Instancia: {num_cds} CDs")
    
    reporte.append("\nCARACTERIZACIÓN DE LA INSTANCIA")
    reporte.append(caracterizacionStr.strip())

    reporte.append("\nRESULTADOS EPSILON-CONSTRAINT")
    if EPSILON or setPoints:
        reporte.append(f"Tiempo de ejecución : {epsilonTime:.4f} segundos")
        reporte.append(f"Hipervolumen        : {hipervolumenEpsilon:.4f}")
        reporte.append(f"\nPuntos Lexicográficos (Extremos del Frente):")
        reporte.append(f"  - Nadir: Transp={transportMax:.2f}, Infra={infraMax:.2f}")
        reporte.append(f"  - Transp. Mín   : Transp={transportMin:.2f}, Infra={infraMax:.2f}")
        reporte.append(f"  - Infra. Mín    : Transp={transportMax:.2f}, Infra={infraMin:.2f}")
        
        reporte.append(f"\nPuntos del Frente ({len(paretoX)} steps):")
        for i in range(len(paretoX)):
            # Si logras extraer los cromosomas del getEpsilon, añádelos a este string:
            reporte.append(f"  Punto {i+1}: Transp={paretoX[i]:.2f}, Infra={paretoY[i]:.2f}")
    else:
        reporte.append("Ejecución omitida (EPSILON = False).")

    reporte.append("\nRESULTADOS ALGORITMO GENÉTICO (NSGA-II)")
    if MOGA:
        reporte.append(f"Parámetros          : Población={POP_SIZE}, Generaciones={GENERATIONS}, Mutación={MUTATION}")
        reporte.append(f"Llamadas al Solver  : {totalCalls} evaluaciones exactas")
        reporte.append(f"Tiempo de ejecución : {geneticTime:.4f} segundos")
        reporte.append(f"Hipervolumen        : {hipervolumenMoga:.4f}")
        
        reporte.append(f"\nFrente de Pareto Final - {len(paretoFront)} puntos:")
        for i, sol in enumerate(paretoFront):
            cromosoma = sol[0]
            t_val = sol[1]
            i_val = sol[2]
            reporte.append(f"  Punto {i+1}: Transp={t_val:.2f}, Infra={i_val:.2f} | Cromosoma: {cromosoma}")
            
    else:
        reporte.append("Ejecución omitida (MOGA = False).")

    reporte.append("\nCOMPARATIVA ESTADÍSTICA")
    if (EPSILON or setPoints) and MOGA and hipervolumenEpsilon > 0:
        reporte.append(f"Calidad del MOGA vs Exacto : {calidadMoga:.2f}% (Cobertura del Hipervolumen)")
        if geneticTime > 0:
            aceleracion = epsilonTime / geneticTime
            reporte.append(f"Aceleración de Tiempo      : El MOGA fue {aceleracion:.2f}x más rápido que Epsilon")
    else:
        reporte.append("No hay datos suficientes de ambos métodos para comparar.")

    # Guardar en archivo txt
    nombreArchivo = f"Resultados_{nombreInstancia}_{time.time()}.txt"
    with open(nombreArchivo, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))
    
    print(f"\n*** Los resultados han sido guardados exitosamente en '{nombreArchivo}' ***")
    ###############################################################



    plt.xlabel("Costo Transporte")
    plt.ylabel("Costo Infraestructura")
    
    plt.xlabel("Costo Transporte")
    plt.ylabel("Costo Infraestructura")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # =========================================================
    # NORMALIZACIÓN Y PLOTEO
    # =========================================================
    plt.figure(figsize=(10, 6))

    # Funciones lambda para normalizar Min-Max (escala 0 a 1)
    norm_X = lambda x: (x - transportMin) / (transportMax - transportMin) if transportMax > transportMin else 0
    norm_Y = lambda y: (y - infraMin) / (infraMax - infraMin) if infraMax > infraMin else 0

    if(EPSILON or setPoints):
        # Normalizamos los arreglos del Epsilon
        paretoX_norm = [norm_X(x) for x in paretoX]
        paretoY_norm = [norm_Y(y) for y in paretoY]
        
        plt.plot(paretoX_norm, paretoY_norm, color='green', label='Frente Exacto (Epsilon)')
        plt.scatter(paretoX_norm, paretoY_norm, color='green')
        
        # Extremos normalizados: (0, 1) y (1, 0)
        plt.scatter([0, 1], [1, 0], c=['blue', 'red'], zorder=5)

    if(MOGA):
        # Normalizamos los arreglos del MOGA
        mogaX_norm = [norm_X(x) for x in mogaX]
        mogaY_norm = [norm_Y(y) for y in mogaY]
        
        plt.scatter(mogaX_norm, mogaY_norm, color='purple', edgecolor='black', zorder=4, label='Frente Final (NSGA II)', alpha=0.5)

    plt.xlabel("Costo Transporte")
    plt.ylabel("Costo Infraestructura")
    plt.title(f"Frente de Pareto: {num_cds} CDs")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
