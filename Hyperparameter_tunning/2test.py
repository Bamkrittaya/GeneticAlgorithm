import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

random.seed(42)
np.random.seed(42)

LR_MIN, LR_MAX = 1e-5, 1e-1
BS_OPTIONS = [16, 32, 64, 128]

GENS = 20
POP_SIZE = 20

# ============================================================
# MOCK BLACK-BOX OBJECTIVE FUNCTION
# ============================================================

def mock_validation_loss(lr, bs):
    """
    Simulated validation loss.
    Global optimum approximately at:
        lr ≈ 0.01
        bs ≈ 64
    """
    lr_term = (np.log10(lr) + 2) ** 2
    bs_term = ((bs - 64) / 64) ** 2
    noise = np.random.normal(0, 0.05)
    return lr_term + bs_term + noise


def fitness(individual):
    return -mock_validation_loss(*individual)


# ============================================================
# GA OPERATORS
# ============================================================

def random_individual():
    lr = 10 ** random.uniform(-5, -1)
    bs = random.choice(BS_OPTIONS)
    return (lr, bs)


def roulette_selection(population, fitnesses):
    min_fit = min(fitnesses)
    weights = [f - min_fit + 1e-6 for f in fitnesses]
    return random.choices(population, weights=weights, k=1)[0]


def crossover(parent1, parent2, p=0.9):
    if random.random() < p:
        lr = (parent1[0] + parent2[0]) / 2
        bs = random.choice([parent1[1], parent2[1]])
        return (lr, bs), (lr, bs)
    return parent1, parent2


def mutate(individual, p=0.2):
    lr, bs = individual
    if random.random() < p:
        lr *= 10 ** np.random.normal(0, 0.2)
        lr = np.clip(lr, LR_MIN, LR_MAX)
    if random.random() < p:
        bs = random.choice(BS_OPTIONS)
    return (lr, bs)


# ============================================================
# GENETIC ALGORITHM (SINGLE RUN)
# ============================================================

def run_ga():
    population = [random_individual() for _ in range(POP_SIZE)]

    best_fitness_history = []
    avg_fitness_history = []

    best_so_far = None
    best_so_far_fit = -np.inf

    for gen in range(GENS):
        fitnesses = [fitness(ind) for ind in population]

        gen_best_fit = max(fitnesses)
        gen_avg_fit = np.mean(fitnesses)

        if gen_best_fit > best_so_far_fit:
            best_so_far_fit = gen_best_fit
            best_so_far = population[np.argmax(fitnesses)]

        best_fitness_history.append(best_so_far_fit)
        avg_fitness_history.append(gen_avg_fit)

        # Elitism
        new_population = [best_so_far]

        while len(new_population) < POP_SIZE:
            p1 = roulette_selection(population, fitnesses)
            p2 = roulette_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))

        population = new_population

    return best_so_far, best_fitness_history, avg_fitness_history


# ============================================================
# RUN GA
# ============================================================

best_solution, best_hist, avg_hist = run_ga()

print("Best hyperparameters found:")
print(f"  Learning rate: {best_solution[0]:.5f}")
print(f"  Batch size:    {best_solution[1]}")

# ============================================================
# VISUALIZATION (EXAMPLE IMPLEMENTATION FIGURE)
# ============================================================

plt.figure()
plt.plot([-f for f in best_hist], label="Best validation loss")
plt.plot([-f for f in avg_hist], label="Average validation loss", linestyle="--")
plt.xlabel("Generation")
plt.ylabel("Validation loss")
plt.title("Genetic Algorithm Convergence")
plt.legend()
plt.grid(True)
plt.show()
