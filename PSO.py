import random
import math

# Fitness function (we want to minimize this)
def fitness(x, y):
    return x**2 + y**2

# Particle class
class Particle:
    def __init__(self, x, y):
        self.position = [x, y]
        self.velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.best_position = list(self.position)
        self.best_fitness = fitness(x, y)

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        for i in range(2):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self):
        for i in range(2):
            self.position[i] += self.velocity[i]

        current_fitness = fitness(self.position[0], self.position[1])
        if current_fitness < self.best_fitness:  # minimization
            self.best_position = list(self.position)
            self.best_fitness = current_fitness

# PSO algorithm
def PSO(num_particles=5, max_iter=20):
    # Initialize swarm
    swarm = [Particle(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(num_particles)]

    # Initialize global best
    global_best = min(swarm, key=lambda p: p.best_fitness)
    global_best_position = list(global_best.best_position)
    global_best_fitness = global_best.best_fitness

    print("Initial Global Best:", global_best_position, "Fitness =", global_best_fitness)

    # Run iterations
    for t in range(max_iter):
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()

        # Update global best
        current_best = min(swarm, key=lambda p: p.best_fitness)
        if current_best.best_fitness < global_best_fitness:
            global_best_position = list(current_best.best_position)
            global_best_fitness = current_best.best_fitness

        print(f"Iteration {t+1}: Global Best = {global_best_position}, Fitness = {global_best_fitness:.4f}")

    return global_best_position, global_best_fitness


# Run PSO
best_pos, best_fit = PSO()
print("\nFinal Optimal Sensor Placement:", best_pos)
print("Minimum Distance Squared (Fitness):", best_fit)
