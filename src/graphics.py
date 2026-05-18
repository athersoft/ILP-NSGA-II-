import matplotlib.pyplot as plt

def plotAbsoluteFront(paretoX, paretoY, mogaX, mogaY, transportMin, transportMax, infraMin, infraMax, epsilon, setPoints):
    plt.figure(figsize=(10, 6))
    
    if epsilon or setPoints:
        plt.plot(paretoX, paretoY, color='green')
        plt.scatter(paretoX, paretoY, color='green')
        plt.scatter([transportMin, transportMax], [infraMax, infraMin], c=['blue', 'red'], zorder=5)

    plt.scatter(mogaX, mogaY, color='purple', edgecolor='black', zorder=4, label='Frente Final (MOGA)', alpha=0.5)

    plt.xlabel("Costo Transporte")
    plt.ylabel("Costo Infraestructura")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plotNormalizedFront(paretoX, paretoY, mogaX, mogaY, transportMin, transportMax, infraMin, infraMax, epsilon, setPoints, numCds):
    plt.figure(figsize=(10, 6))

    normX = lambda x: (x - transportMin) / (transportMax - transportMin) if transportMax > transportMin else 0
    normY = lambda y: (y - infraMin) / (infraMax - infraMin) if infraMax > infraMin else 0

    if epsilon or setPoints:
        paretoXNorm = [normX(x) for x in paretoX]
        paretoYNorm = [normY(y) for y in paretoY]
        
        plt.plot(paretoXNorm, paretoYNorm, color='green', label='Frente Exacto (Epsilon)')
        plt.scatter(paretoXNorm, paretoYNorm, color='green')
        plt.scatter([0, 1], [1, 0], c=['blue', 'red'], zorder=5)


    mogaXNorm = [normX(x) for x in mogaX]
    mogaYNorm = [normY(y) for y in mogaY]
    
    plt.scatter(mogaXNorm, mogaYNorm, color='purple', edgecolor='black', zorder=4, label='Frente Final (NSGA II)', alpha=0.5)

    plt.xlabel("Costo Transporte")
    plt.ylabel("Costo Infraestructura")
    plt.title(f"Frente de Pareto: {numCds} CDs")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()