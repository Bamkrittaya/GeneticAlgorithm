import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

random.seed(42)
np.random.seed(42)

# Learning rate bounds (scientific notation)
LR_MIN = 1e-5   # 0.00001
LR_MAX = 1e-1   # 0.1

# Discrete batch-size options
BS_OPTIONS = [16, 32, 64, 128]

POP_SIZE = 20
GENS = 20

OUT = "ga_representation_outputs"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# MOCK BLACK-BOX OBJECTIVE FUNCTION
# ============================================================

def mock_validation_loss(lr, bs):
    lr_term = (np.log10(lr) + 2) ** 2
    bs_term = ((bs - 64) / 64) ** 2
    noise = np.random.normal(0, 0.05)
    return lr_term + bs_term + noise


def fitness(ind):
    return -mock_validation_loss(*ind)

# ============================================================
# GA OPERATORS
# ============================================================

def random_individual():
    lr = 10 ** random.uniform(-5, -1)
    bs = random.choice(BS_OPTIONS)
    return (lr, bs)


def roulette_selection(pop, fits):
    min_fit = min(fits)
    weights = [f - min_fit + 1e-6 for f in fits]
    return random.choices(pop, weights=weights, k=1)[0]


def crossover(p1, p2, p=0.9):
    if random.random() < p:
        lr = (p1[0] + p2[0]) / 2
        bs = random.choice([p1[1], p2[1]])
        return (lr, bs), (lr, bs)
    return p1, p2


def mutate(ind, p=0.2):
    lr, bs = ind
    if random.random() < p:
        lr *= 10 ** np.random.normal(0, 0.2)
        lr = np.clip(lr, LR_MIN, LR_MAX)
    if random.random() < p:
        bs = random.choice(BS_OPTIONS)
    return (lr, bs)

# ============================================================
# INITIAL POPULATION (DATASET & REPRESENTATION)
# ============================================================

initial_population = [random_individual() for _ in range(POP_SIZE)]

# ---- Create TABLE ----
table_data = []
for i, (lr, bs) in enumerate(initial_population):
    table_data.append({
        "Individual": i + 1,
        "Learning Rate": lr,
        "Batch Size": bs,
        "Initial Loss": mock_validation_loss(lr, bs)
    })

df_init = pd.DataFrame(table_data)

df_init.to_csv(f"{OUT}/initial_population_table.csv", index=False)

print("\nInitial Population (Sample):")
print(df_init.head(8))

# ---- Initial Population PLOT ----
lrs = [np.log10(ind[0]) for ind in initial_population]
bss = [ind[1] for ind in initial_population]
losses = [row["Initial Loss"] for row in table_data]

plt.figure()
plt.scatter(lrs, bss, c=losses, cmap="viridis", s=80)
plt.xlabel("log10(Learning Rate)")
plt.ylabel("Batch Size")
plt.title("Initial Population Distribution (Dataset & Representation)")
plt.colorbar(label="Validation Loss")
plt.grid(True)
plt.savefig(f"{OUT}/initial_population_plot.png", dpi=220)
plt.show()

# ============================================================
# GENETIC ALGORITHM (DIVERSITY ANALYSIS)
# ============================================================

def lr_diversity(pop):
    lrs = [np.log10(ind[0]) for ind in pop]
    return np.std(lrs)


def run_ga():
    pop = initial_population.copy()
    lr_div_history = []

    best_fit = -np.inf
    best_ind = None

    print("\nRunning Genetic Algorithm...\n")

    for g in range(GENS):
        fits = [fitness(ind) for ind in pop]

        gen_best_fit = max(fits)
        gen_best_ind = pop[np.argmax(fits)]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = gen_best_ind

        lr_div_history.append(lr_diversity(pop))

        print(
            f"Gen {g:02d} | "
            f"Best fitness: {gen_best_fit:.4f} | "
            f"Best (lr, bs): ({gen_best_ind[0]:.5f}, {gen_best_ind[1]})"
        )

        # elitism
        new_pop = [best_ind]

        while len(new_pop) < POP_SIZE:
            p1 = roulette_selection(pop, fits)
            p2 = roulette_selection(pop, fits)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))

        pop = new_pop

    return lr_div_history, best_ind, -best_fit

# ============================================================
# POPULATION DIVERSITY VISUALIZATION
# ============================================================

lr_div_hist, best_ind, best_loss = run_ga()

plt.figure()
plt.plot(lr_div_hist, marker="o")
plt.xlabel("Generation")
plt.ylabel("Learning Rate Diversity (std of log10)")
plt.title("Population Diversity Over Generations")
plt.grid(True)
plt.savefig(f"{OUT}/population_diversity.png", dpi=220)
plt.show()

print("\n================ FINAL RESULT ================")
print(f"Best learning rate found: {best_ind[0]:.6f}")
print(f"Best batch size found:   {best_ind[1]}")
print(f"Best validation loss:    {best_loss:.6f}")
print(f"\n📁 All outputs saved in: {OUT}")
