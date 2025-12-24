# 8_queens_ga.py
# Genetic Algorithm for the 8-Queens Problem (single-file version)

import random
import copy
import matplotlib.pyplot as plt


# ==================================================
# ASCII board printing
# ==================================================

def print_board(board):
    n = len(board)
    for r in range(n):
        print(" ", end="")
        for _ in range(n):
            print("---", end=" ")
        print("\n|", end="")
        for c in range(n):
            if board[r] == c:
                print(" X |", end="")
            else:
                print("   |", end="")
        print()
    print(" ", end="")
    for _ in range(n):
        print("---", end=" ")
    print("\n")


def save_result_to_file(solution, filename="8_queens_result.txt"):
    with open(filename, "w") as f:
        f.write(f"Solution found: {solution}\n\n")
        n = len(solution)
        for r in range(n):
            f.write(" " + "--- " * n + "\n")
            f.write("|")
            for c in range(n):
                if solution[r] == c:
                    f.write(" X |")
                else:
                    f.write("   |")
            f.write("\n")
        f.write(" " + "--- " * n + "\n")


def plot_board(solution, filename="8_queens_solution.png"):
    n = len(solution)
    board = [[0] * n for _ in range(n)]
    for r, c in enumerate(solution):
        board[r][c] = 1

    plt.figure(figsize=(4, 4))
    plt.imshow(board, cmap="binary")
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.grid(True)
    plt.title("8-Queens Solution")
    plt.savefig(filename, dpi=300)
    plt.close()


# ==================================================
# Genetic Algorithm (simplified, problem-specific)
# ==================================================

class GeneticAlgorithm:
    def __init__(
        self,
        n=8,
        population_size=200,
        generations=100,
        crossover_probability=0.8,
        mutation_probability=0.2,
        elitism=True,
    ):
        self.n = n
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.population = []

    # ---------- Representation ----------

    def create_individual(self):
        ind = list(range(self.n))
        random.shuffle(ind)
        return ind

    # ---------- Fitness ----------

    def fitness(self, individual):
        collisions = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(individual[i] - individual[j]) == abs(i - j):
                    collisions += 1
        return collisions

    # ---------- Genetic operators ----------

    def crossover(self, p1, p2):
        idx = random.randrange(1, self.n)
        c1 = p1[:idx] + [x for x in p2 if x not in p1[:idx]]
        c2 = p2[:idx] + [x for x in p1 if x not in p2[:idx]]
        return c1, c2

    def mutate(self, ind):
        i, j = random.sample(range(self.n), 2)
        ind[i], ind[j] = ind[j], ind[i]

    def select(self):
        return random.choice(self.population)

    # ---------- Core GA ----------

    def initialize(self):
        self.population = [self.create_individual()
                           for _ in range(self.population_size)]

    def evolve(self, log_callback=None):
        self.initialize()
        fitness_history = []

        for gen in range(self.generations):
            scored = [(self.fitness(ind), ind) for ind in self.population]
            scored.sort(key=lambda x: x[0])

            best_fitness, best_ind = scored[0]
            fitness_history.append(best_fitness)

            if log_callback:
                log_callback(gen, best_fitness)

            if best_fitness == 0:
                break

            new_population = []

            if self.elitism:
                new_population.append(copy.deepcopy(best_ind))

            while len(new_population) < self.population_size:
                p1 = self.select()
                p2 = self.select()

                if random.random() < self.crossover_probability:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                if random.random() < self.mutation_probability:
                    self.mutate(c1)
                if random.random() < self.mutation_probability:
                    self.mutate(c2)

                new_population.extend([c1, c2])

            self.population = new_population[:self.population_size]

        return best_fitness, best_ind, fitness_history


# ==================================================
# Experiments
# ==================================================

def run_single(elitism=True, plot=False):
    history = []

    def logger(gen, fitness):
        history.append(fitness)
        print(f"Gen {gen:3d} | Best fitness: {fitness}")

    ga = GeneticAlgorithm(elitism=elitism)
    best_fitness, solution, history = ga.evolve(log_callback=logger)

    if plot:
        plt.figure()
        plt.plot(history, marker="o")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (Collisions)")
        plt.title("Fitness vs Generation")
        plt.grid(True)
        plt.show()

    return best_fitness, solution, history


def run_experiment(elitism=True, runs=10):
    success = 0
    convergence = []

    for _ in range(runs):
        best_fitness, _, history = run_single(elitism=elitism, plot=False)
        if 0 in history:
            success += 1
            convergence.append(history.index(0))

    return success, convergence


# ==================================================
# Main
# ==================================================

def main():
    print("\n=== SINGLE RUN (SHOWING ALL ITERATIONS) ===\n")
    best_fitness, solution, history = run_single(elitism=True, plot=True)

    if best_fitness == 0:
        print("\n✅ Solution found:", solution)
        print_board(solution)
        save_result_to_file(solution)
        plot_board(solution)
        print("\n📄 Saved: 8_queens_result.txt")
        print("🖼️  Saved: 8_queens_solution.png")

    print("\n=== ELITISM COMPARISON ===\n")
    runs = 10
    s_on, conv_on = run_experiment(elitism=True, runs=runs)
    s_off, conv_off = run_experiment(elitism=False, runs=runs)

    print(f"Elitism ON  | Success: {s_on}/{runs} | Avg convergence:",
          sum(conv_on) / len(conv_on) if conv_on else "N/A")

    print(f"Elitism OFF | Success: {s_off}/{runs} | Avg convergence:",
          sum(conv_off) / len(conv_off) if conv_off else "N/A")

    plt.figure()
    plt.boxplot([conv_on, conv_off], labels=["Elitism ON", "Elitism OFF"])
    plt.ylabel("Generations to Solution")
    plt.title("Convergence Speed Comparison")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
