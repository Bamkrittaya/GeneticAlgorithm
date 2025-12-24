# ============================================================
# 8 Queens — Genetic Algorithm (Diagram-Mapped Version)
#
# This implementation explicitly follows the standard GA flow:
#
# Initial population P(t)
#   ↓
# Evaluation (fitness computation)
#   ↓
# Termination check
#   ↓
# Selection (parents)
#   ↓
# Crossover → Mutation (offspring)
#   ↓
# New population P(t+1)
#
# Trace mode prints and saves how GA works step-by-step.
# ============================================================

import random
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

N = 8
POP_SIZE = 50          # population size per generation
GENERATIONS = 50       # max generations
MUTATION_RATE = 0.2
ELITISM = True         # keep best individual
TRACE = True           # ON/OFF: show GA internals
OUT_FILE = "ga_trace_8queens.txt"

random.seed(42)


# ============================================================
# UTILITY: Save logs
# ============================================================

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# ============================================================
# REPRESENTATION
# board[i] = column of queen in row i
# This enforces one queen per row & column
# ============================================================

def random_board():
    return random.sample(range(N), N)


# ============================================================
# FITNESS FUNCTION (Evaluation box in diagram)
# Counts diagonal collisions (lower is better)
# ============================================================

def fitness(board):
    collisions = 0
    for i in range(N):
        for j in range(i + 1, N):
            if abs(board[i] - board[j]) == abs(i - j):
                collisions += 1
    return collisions


# ============================================================
# SELECTION (Roulette wheel equivalent)
# Here: truncation selection (top-K)
# ============================================================

def select_parents(population, k=20):
    return random.sample(population[:k], 2)


# ============================================================
# CROSSOVER (Crossover box in diagram)
# Order-preserving crossover for permutations
# ============================================================

def crossover(p1, p2):
    cut = random.randint(1, N - 1)
    child = p1[:cut] + [x for x in p2 if x not in p1[:cut]]
    return child, cut


# ============================================================
# MUTATION (Mutation box in diagram)
# Swap mutation
# ============================================================

def mutate(child):
    i, j = random.sample(range(N), 2)
    child[i], child[j] = child[j], child[i]
    return i, j


# ============================================================
# MAIN GA LOOP
# ============================================================

def solve_8_queens():

    # --------------------------------------------------------
    # 1. INITIAL POPULATION  P(t)
    # --------------------------------------------------------
    population = [random_board() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):

        # ----------------------------------------------------
        # 2. EVALUATION (Fitness computation)
        # ----------------------------------------------------
        population.sort(key=fitness)
        best = population[0]
        best_fit = fitness(best)

        log(f"\nGen {gen:2d} | Best fitness: {best_fit}")

        # ----------------------------------------------------
        # 3. TERMINATION CONDITION
        # ----------------------------------------------------
        if best_fit == 0:
            log("✔ Termination condition met (solution found)")
            return best, gen

        # ----------------------------------------------------
        # 4. CREATE NEW POPULATION  P(t+1)
        # ----------------------------------------------------
        new_population = []

        # ---- Elitism (optional) ----
        if ELITISM:
            new_population.append(best.copy())
            log("  [Elitism] Best individual preserved")

        # ----------------------------------------------------
        # 5. SELECTION → CROSSOVER → MUTATION
        # These steps collectively form P(t+1)
        # ----------------------------------------------------
        while len(new_population) < POP_SIZE:

            # ---- Selection ----
            p1, p2 = select_parents(population)

            if TRACE:
                log(f"  Parent 1: {p1}")
                log(f"  Parent 2: {p2}")

            # ---- Crossover ----
            child, cut = crossover(p1, p2)

            if TRACE:
                log(f"  Crossover index: {cut}")
                log(f"  Child before mutation: {child}")

            # ---- Mutation ----
            if random.random() < MUTATION_RATE:
                i, j = mutate(child)
                if TRACE:
                    log(f"  Mutation swap indices: {i} {j}")
                    log(f"  Child after mutation: {child}")

            new_population.append(child)

        # ----------------------------------------------------
        # 6. REPLACE OLD POPULATION
        # ----------------------------------------------------
        population = new_population[:POP_SIZE]

    # If max generations reached
    return population[0], GENERATIONS


# ============================================================
# PRINT BOARD
# ============================================================

def print_board(board):
    for r in range(N):
        print(" ", "--- " * N)
        print("|", end="")
        for c in range(N):
            print(" X |" if board[r] == c else "   |", end="")
        print()
    print(" ", "--- " * N)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    log("=== 8 Queens Genetic Algorithm Trace ===")
    log(f"Start time: {datetime.now()}")
    log(f"Population size: {POP_SIZE}, Max generations: {GENERATIONS}")
    log(f"Elitism: {ELITISM}, Mutation rate: {MUTATION_RATE}")
    log("-" * 50)

    solution, gen_found = solve_8_queens()

    log("\n=== FINAL RESULT ===")
    log(f"Solution found at generation {gen_found}")
    log(f"Board: {solution}")
    log(f"Fitness: {fitness(solution)}")

    # Save trace
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(log_lines))

    print("\nFinal Board:")
    print_board(solution)

    print(f"\nTrace saved to: {OUT_FILE}")
