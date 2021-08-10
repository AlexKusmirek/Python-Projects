  
"""Initial version of a generic 4-variable genetic algorithm."""

import numpy as np
import matplotlib.pyplot as plt

# Problem definition
def fitness(chromosome):

    [incs, planes, per_plane, raan] = chromosome
    fit = planes * per_plane + (incs / 2) - (raan / 2)

    return fit

def mutation(mu):

    import random

    return random.random() < mu

# ALso need to include constraints on number of satellites, max performance score, etc.
problem = {
    "satmin": 1,
    "satmax": 11,
    "planesmin": 1,
    "planesmax": 11,
    "raanmin": 0,
    "raanmax": 91,
    "incmin": 0,
    "incmax": 91

}

# GA parameters
params = {
    "maxit": 200,
    "npop": 20,
    "mu": 0.1
}

npop = params.get("npop")
gamma = params.get("gamma")

# Initial individual constellation template
"""The constellation variables that need to be iterated are n_planes, n_sats, inc, first_raan and walker_factor"""
incmax = problem.get("incmax")
incmin = problem.get("incmin")
satmax = problem.get("satmax")  # Sats and planes need to be ints
satmin = problem.get("satmin")
planesmax = problem.get("planesmax")
planesmin = problem.get("planesmin")
raanmax = problem.get("raanmax")
raanmin = problem.get("raanmin")
maxit = params.get("maxit")
mu = params.get("mu")

# Initialise first population
pop = [None] * npop
cost = [None] * npop
popsol = [None] * npop
k = 0
for k in range(0, npop):
    planes = np.random.randint(planesmin, planesmax)
    per_plane = np.random.randint(satmin, satmax)
    incs = np.random.randint(incmin, incmax)
    raan = np.random.randint(raanmin, raanmax)

    chromosome = [incs, planes, per_plane, raan]

    pop[k] = chromosome
    # placeholder for cost function
    fit = fitness(chromosome)
    # Need some constraint to limit the amount of satellites - perhaps an if function?
    cost[k] = fit
    sol = [fit, chromosome]
    popsol[k] = sol

    print "Solution ", k+1
    print chromosome
    print "Performance = ", fit
    print "--------------------------"

# Sort initial parent pool by performance before initiating Main Loop
sortpop = sorted(pop, key=fitness, reverse=True)
popsolsort = sorted(popsol, reverse=True)

# Generation's best solution
bestcost = max(cost)
print "Highest performance of initial generation = ", bestcost

"""With the initial random population generated
the code now progresses to the main loop of the GA:"""

# Empty lists for storing values of each generation to plot
x = []
y = []
mean = []
# Include IF statement to save "perfect" results?

# Main Loop of GA
for it in range(maxit):

    popc = []
    npar = npop // 2
    # Loop for selecting parents, producing offspring, and filling new population
    for _ in range(npar):

        # Selection of fittest solutions as parents
        sortpop = sorted(pop, key=fitness, reverse=True)
        parents = sortpop[0:npar]
        bestsol = sortpop[0]
        # parent population only needs to hold chromosomes once sorted, not fitness values

        # Crossover of selected parents to produce offspring
        q = np.random.permutation(parents)
        p1 = q[0]
        p2 = q[1]

        xpoint = np.random.randint(1, len(p1))

        c1a = list(p1[:xpoint])
        c1b = list(p2[xpoint:])
        c2a = list(p2[:xpoint])
        c2b = list(p1[xpoint:])
        c1 = c1a + c1b
        c2 = c2a + c2b

        # Random mutation of children
        mutate = mutation(mu)
        if mutate:

            mpoint = np.random.randint(0, len(c1))
            if mpoint == 0:
                c1[mpoint] = np.random.randint(incmin, incmax)
            elif mpoint == 1:
                c1[mpoint] = np.random.randint(planesmin, planesmax)
            elif mpoint == 2:
                c1[mpoint] = np.random.randint(satmin, satmax)
            elif mpoint == 3:
                c1[mpoint] = np.random.randint(raanmin, raanmax)

        # Add offspring to new population
        popc += c1, c2

    # Merge, sort and select next generation of parents
    pop = sorted(popc, key=fitness, reverse=True)

    # Add average performance of generation to empty list
    print pop
    aver = []
    for con in pop:

        a = fitness(con)
        aver.append(a)

    genaverage = sum(aver) / len(aver)
    mean.append(genaverage)

    # Need to include a break condition so it doesn't iterate away from optimal solutions
    # Including an average performance per generation would be very useful, including as a break condition

    # Print and record results of generation
    genbest = fitness(pop[0])
    x.append(it+1)
    y.append(genbest)
    print "Generation ", it+1
    print "Best performer :", pop[0]
    print "Best performance :", genbest
    print "------------------------------------"

print "Final optimised population: ",
print pop

# Print optimisation process
plt.plot(x, y)
plt.xlabel("Generation")
plt.ylabel("Best Performance")
plt.title("Best Performance of each Generation")
plt.show()

# Prints to check code is functioning


"""Need to ensure the functions for crossover and mutation are defined
and suitable for the data structure that each generation produces."""
