import numpy as np

# === Step 1: Define the Problem (Optimization Function) ===
def rastrigin(x):
    """Rastrigin function — a common multimodal optimization test."""
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# === Step 2: Initialize Parameters ===
GRID_SIZE = (10, 10)          # 10x10 grid of cells
DIMENSIONS = 2                # Solution space dimension
ITERATIONS = 100              # Number of iterations
NEIGHBORHOOD_RADIUS = 1       # How far neighbors influence each other
BOUNDS = (-5.12, 5.12)        # Bounds for Rastrigin function

# === Step 3: Initialize Population ===
population = np.random.uniform(BOUNDS[0], BOUNDS[1], size=(GRID_SIZE[0], GRID_SIZE[1], DIMENSIONS))

# === Step 4: Evaluate Fitness Function ===
def evaluate_population(pop):
    fitness = np.zeros(GRID_SIZE)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            fitness[i, j] = rastrigin(pop[i, j])
    return fitness

# === Helper: Get Neighborhood ===
def get_neighbors(i, j):
    neighbors = []
    for di in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
        for dj in range(-NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_RADIUS + 1):
            if di == 0 and dj == 0:
                continue
            ni = (i + di) % GRID_SIZE[0]
            nj = (j + dj) % GRID_SIZE[1]
            neighbors.append((ni, nj))
    return neighbors

# === Step 5: Update States ===
def update_population(pop, fitness):
    new_pop = np.copy(pop)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            neighbors = get_neighbors(i, j)
            best_neighbor = min(neighbors, key=lambda n: fitness[n[0], n[1]])
            if fitness[best_neighbor] < fitness[i, j]:
                # Move slightly toward the best neighbor
                new_pop[i, j] = pop[i, j] + 0.5 * (pop[best_neighbor] - pop[i, j])
                # Add small random mutation
                new_pop[i, j] += np.random.uniform(-0.1, 0.1, DIMENSIONS)
                # Enforce bounds
                new_pop[i, j] = np.clip(new_pop[i, j], BOUNDS[0], BOUNDS[1])
    return new_pop

# === Step 6 & 7: Iterate and Output Best Solution ===
best_solution = None
best_fitness = float('inf')

for iteration in range(ITERATIONS):
    fitness = evaluate_population(population)
    current_best_index = np.unravel_index(np.argmin(fitness), GRID_SIZE)
    current_best = population[current_best_index]
    current_best_fitness = fitness[current_best_index]
    
    # Track the best found solution
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        best_solution = current_best.copy()
    
    # Print progress
    print(f"Iteration {iteration + 1}/{ITERATIONS} | Best Fitness: {best_fitness:.6f}")
    
    # Update population for next iteration
    population = update_population(population, fitness)

# === Final Output ===
print("\n=== Best Solution Found ===")
print(f"Best Solution (x): {best_solution}")
print(f"Best Fitness Value: {best_fitness:.6f}")
