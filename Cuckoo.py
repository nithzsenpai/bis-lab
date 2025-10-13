import numpy as np

# Objective function
def objective_function(x):
    return x**2

# Lévy flight
def levy_flight(Lambda=1.5):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    step = u / (abs(v) ** (1 / Lambda))
    return step

# Cuckoo Search Algorithm
def cuckoo_search(n=15, max_iter=50, pa=0.25, alpha=0.01, bounds=(-10, 10)):
    # Initialize nests randomly within bounds
    nests = np.random.uniform(bounds[0], bounds[1], n)
    fitness = np.array([objective_function(x) for x in nests])
    best_index = np.argmin(fitness)
    best = nests[best_index]

    for t in range(max_iter):
        # Generate new solution by Lévy flight
        for i in range(n):
            step = alpha * levy_flight() * (nests[i] - best)
            new_solution = nests[i] + step
            new_solution = np.clip(new_solution, bounds[0], bounds[1])  # keep in bounds
            new_fitness = objective_function(new_solution)

            # Replace if better
            if new_fitness < fitness[i]:
                nests[i] = new_solution
                fitness[i] = new_fitness

        # Abandon a fraction of nests
        abandon = np.random.rand(n) < pa
        nests[abandon] = np.random.uniform(bounds[0], bounds[1], np.sum(abandon))
        fitness[abandon] = [objective_function(x) for x in nests[abandon]]

        # Update best
        best_index = np.argmin(fitness)
        if fitness[best_index] < objective_function(best):
            best = nests[best_index]

        # Print progress
        print(f"Iteration {t+1}: Best x = {best:.4f}, f(x) = {objective_function(best):.6f}")

    return best, objective_function(best)

# Run CSA
best_x, best_fx = cuckoo_search()
print("\nFinal Best Solution:")
print(f"x = {best_x:.4f}, f(x) = {best_fx:.6f}")
