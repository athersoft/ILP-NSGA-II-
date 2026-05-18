import time
from amplpy import AMPL, OutputHandler
import numpy as np
import math
from .model import modelMultiObj
from .utils import repairChromosome

workerAmpl = None

def getEpsilon(currentInstance, mTransport, mInfrastructure, steps, relaxed = False):
    # Puntos lexicográficos
    transportMin, aux, _ = solveInstance(currentInstance, mTransport, relaxation = relaxed)
    aux, infraMax, _ = solveInstance(currentInstance, mInfrastructure, math.ceil(transportMin), relaxation = relaxed)

    aux, infraMin, _ = solveInstance(currentInstance, mInfrastructure, relaxation = relaxed)
    transportMax, aux, _ = solveInstance(currentInstance, mTransport, math.ceil(infraMin), relaxation = relaxed)

    print(f"Punto X lexicográfico de infraestructura e inventario: {transportMax}")
    print(f"Punto Y lexicográfico de infraestructura e inventario: {infraMin}")
    print(f"Punto X lexicográfico de transporte: {transportMin}")
    print(f"Punto Y lexicográfico de transporte: {infraMax}")

    paretoX = []
    paretoY = []
    cdList = []

    # 1er Barrido: Minimizar Transporte sujeto a Infraestructura
    print("\nCalculando Epsilon 1 (Min Transport)")
    epsilonSteps1 = np.linspace(infraMin, infraMax, steps)
    for step in epsilonSteps1:
        transportCost, infraCost, openCds = solveEpsilon(currentInstance, mTransport, step, relaxed= relaxed)
        if transportCost is not None:
            paretoX.append(transportCost)
            paretoY.append(infraCost)
            cdList.append(openCds)


    # 2do Barrido: Minimizar Infraestructura sujeto a Transporte
    print("Calculando Epsilon 2 (Min Infra)")
    epsilonSteps2 = np.linspace(transportMin, transportMax, steps)
    for step in epsilonSteps2:
        transportCost, infraCost, openCds = solveEpsilon(currentInstance, mInfrastructure, step, relaxed= relaxed)
        if infraCost is not None:
            paretoX.append(transportCost)
            paretoY.append(infraCost)
            cdList.append(openCds)

    # Limpieza: Unir, eliminar duplicados y ordenar por el eje X (Costo de Transporte)
    puntosUnicos = list(set(zip(paretoX, paretoY)))
    puntosOrdenados = sorted(puntosUnicos, key=lambda x: x[0])
    
    paretoX_clean = [p[0] for p in puntosOrdenados]
    paretoY_clean = [p[1] for p in puntosOrdenados]

    print(f"\nPareto X (Ordenado): {paretoX_clean}")
    print(f"Pareto Y (Ordenado): {paretoY_clean}")

    return transportMin, transportMax, infraMin, infraMax, paretoX_clean, paretoY_clean, cdList

def solveInstance(instance, model, epsilon = 1e20, relaxation = False, fixedCds = None, maxInfra = 1, maxTransp = 1, alpha = 0.5, solver = "gurobi", relaxedMutation = True, relaxedCrossover = True):
    t0 = time.time()
    ampl = AMPL()
    ampl.eval("reset;")
    ampl.eval(model)
    ampl.eval(instance)
    ampl.param["epsilon"] = epsilon
    ampl.setOption("solver", solver)
    #ampl.option["gurobi_options"] = "NonConvex=2 timelimit=1800"
    if relaxation:
        ampl.setOption("relax_integrality", 1)

    if solver == "gurobi":
        ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=0.05"
    if solver == "knitro":
        opciones = "outlev=0 mip_integral_gap_rel=0.05 opttol=1e-4 feastol=1e-4 mip_method=2 numthreads = 15"
        ampl.setOption("knitro_options", opciones)
    if solver == "snopt":
        ampl.setOption("relax_integrality", 1)
    #ampl.setOutputHandler(SilentOutputHandler())
    if model == modelMultiObj:
        ampl.param["maxTransp"] = maxTransp
        ampl.param["maxInfra"] = maxInfra
        ampl.param["alpha"] = alpha

    if fixedCds:
        indexList = list(ampl.getVariable("Z").getValues().toDict().keys())
        
        for cdIndex, state in fixedCds.items():
            if cdIndex in indexList:
                comando = f"let Z[{cdIndex}] := {state}; fix Z[{cdIndex}];"
                ampl.eval(comando)

    ampl.solve()
    print(f"Tiempo relaxation = {relaxation}: {time.time()-t0}")
    transp = ampl.getValue("CostoTransp")
    infra = ampl.getValue("CostoInfra")
    #print(f"Cds abiertos: {ampl.getData("Z")} ")
    openCds = ampl.getData("Z").toList()
    ampl.close()

    return transp, infra, openCds

def solveEpsilon(instance, model, epsilonValue, relaxed = False):
    ampl = AMPL()
    ampl.eval("reset;")
    ampl.eval(model)
    ampl.eval(instance)
    ampl.param["epsilon"] = epsilonValue
    ampl.setOption("solver", "gurobi")
    
    ampl.setOption("gurobi_options", "outlev=0") 
    ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=1e-8 FeasTol=1e-9 BarConvTol=1e-9"
    ampl.option["gurobi_options"] = "NonConvex=2 MIPGap=0.05 timelimit=1800"
    if relaxed:
        ampl.setOption("relax_integrality", 1)
    
    ampl.solve()
    
    solveResult = ampl.getValue("solve_result")
    if solveResult == "solved":
        transp = ampl.getValue("CostoTransp")
        infra = ampl.getValue("CostoInfra")
        openCds = ampl.getData("Z").toList()
        ampl.close()
        return transp, infra, openCds
    ampl.close()
    return None, None, None


################Funciones para parelelizar###################
class SilentOutputHandler(OutputHandler):
    def output(self, kind, msg):
        pass

def initWorker(modelCode, dataStr, licenseUuid, gurobiOptions, maxTransp, maxInfra, solver = "gurobi"):
    """Inicializa AMPL leyendo strings de texto directo."""
    global workerAmpl
    try:
        workerAmpl = AMPL()
        workerAmpl.setOutputHandler(SilentOutputHandler())
        
        workerAmpl.eval("reset;")
        workerAmpl.eval(modelCode)
        workerAmpl.eval(dataStr)

        workerAmpl.param["maxTransp"] = maxTransp
        workerAmpl.param["maxInfra"] = maxInfra
        
        # Configuración silenciosa para el worker
        workerAmpl.setOption("solver_msg", 0)
        workerAmpl.setOption("solver", solver)
        if solver == "gurobi":
            workerAmpl.setOption("gurobi_options", f"{gurobiOptions} outlev=0 MIPGap=0.05")
        if solver == "knitro":
            opciones = "outlev=0 mip_integral_gap_rel=0.05 opttol=1e-4 feastol=1e-4 mip_method=1"
            workerAmpl.setOption("knitro_options", opciones)
        if solver == "snopt":
            workerAmpl.setOption("presolve", 1)
            workerAmpl.setOption("relax_integrality", 1)
        
        workerAmpl.setOption("presolve", 0) 
        
    except Exception as e:
        print(f"FATAL ERROR en initWorker: {e}")
        workerAmpl = None


def solveWorker(dataInfo):
    chromosome, alphaValue = dataInfo 
    global workerAmpl
    
    if workerAmpl is None or sum(chromosome) == 0:
        return float('inf'), float('inf'), []
        
    try:
        fixCmd = "".join([f"fix Z[{i}]:={x};" for i, x in enumerate(chromosome)])
        workerAmpl.eval(fixCmd)
        
        # Asignamos el valor de alpha al modelo de AMPL
        workerAmpl.param["alpha"] = alphaValue
        
        workerAmpl.solve()
        res = workerAmpl.getValue("solve_result")
        
        if res == "solved" or res == "limit":
            cTransp = workerAmpl.getValue("CostoTransp")
            cInfra = workerAmpl.getValue("CostoInfra")
            demands = workerAmpl.getVariable("D").getValues().toList()
            
            return cTransp, cInfra, demands
        else:
            return float('inf'), float('inf'), []
            
    except Exception as e:
        return float('inf'), float('inf'), []