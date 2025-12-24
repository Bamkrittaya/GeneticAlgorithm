import random
from ga_first import GeneticAlgorithm


# --------------------------------------------------
# Chessboard printing
# --------------------------------------------------

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


# --------------------------------------------------
# Main GA logic
# --------------------------------------------------

def main():
    seed_data = list(range(8))

    ga = GeneticAlgorithm(
        seed_data,
        population_size=200,
        generations=100,
        crossover_probability=0.8,
        mutation_probability=0.2,
        elitism=True,
        maximise_fitness=False,
    )

    # permutation-based individual
    def create_individual(data):
        ind = data[:]
        random.shuffle(ind)
        return ind

    ga.create_individual = create_individual

    # order-preserving crossover
    def crossover(p1, p2):
        idx = random.randrange(1, len(p1))
        c1 = p1[:idx] + [x for x in p2 if x not in p1[:idx]]
        c2 = p2[:idx] + [x for x in p1 if x not in p2[:idx]]
        return c1, c2

    ga.crossover_function = crossover

    # swap mutation
    def mutate(individual):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

    ga.mutate_function = mutate

    # random selection
    ga.selection_function = lambda pop: random.choice(pop)

    # diagonal collision fitness
    def fitness(individual, _):
        collisions = 0
        n = len(individual)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(individual[i] - individual[j]) == abs(i - j):
                    collisions += 1
        return collisions

    ga.fitness_function = fitness

    # run GA (SAFE: no multiprocessing)
    ga.run()

    best_fitness, solution = ga.best_individual()

    if best_fitness == 0:
        print("Solution found:", solution)
        print_board(solution)
    else:
        print("No solution found. Best:", solution, "fitness:", best_fitness)


# --------------------------------------------------
# Required for macOS / multiprocessing safety
# --------------------------------------------------

if __name__ == "__main__":
    main()
