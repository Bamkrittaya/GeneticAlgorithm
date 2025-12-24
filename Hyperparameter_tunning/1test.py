import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# SETUP
# ============================================================

random.seed(42)
np.random.seed(42)

OUT = "ga_hyperparam_outputs"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# HYPERPARAMETER SPACE
# ============================================================

LR_MIN, LR_MAX = 1e-5, 1e-1
BS_OPTIONS = [16, 32, 64, 128]

# ============================================================
# MOCK TRAINING FUNCTION (BLACK BOX)
# ============================================================

def mock_validation_loss(lr, bs):
    """
    Simulated validation loss with noise.
    Global optimum approx:
        lr ≈ 0.01
        bs ≈ 64
    """
    lr_term = (np.log10(lr) + 2) ** 2
    bs_term = ((bs - 64) / 64) ** 2
    noise = np.random.normal(0, 0.05)
    return lr_term + bs_term + noise


def fitness(ind):
    lr, bs = ind
    return -mock_validation_loss(lr, bs)

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
# GENETIC ALGORITHM
# ============================================================

def run_ga(gens=30, pop_size=20):
    pop = [random_individual() for _ in range(pop_size)]

    best_hist = []
    avg_hist = []
    noise_hist = []

    best_so_far = None
    best_so_far_fit = -np.inf

    for g in range(gens):
        fits = [fitness(ind) for ind in pop]
        losses = [-f for f in fits]

        best_fit = max(fits)
        avg_fit = np.mean(fits)
        noise = np.std(losses)

        if best_fit > best_so_far_fit:
            best_so_far_fit = best_fit
            best_so_far = pop[np.argmax(fits)]

        best_hist.append(best_so_far_fit)
        avg_hist.append(avg_fit)
        noise_hist.append(noise)

        print(f"GA Gen {g:02d} | "
              f"Best loss: {-best_so_far_fit:.3f} | "
              f"Avg loss: {-avg_fit:.3f} | "
              f"Noise: {noise:.3f}")

        new_pop = [best_so_far]

        while len(new_pop) < pop_size:
            p1 = roulette_selection(pop, fits)
            p2 = roulette_selection(pop, fits)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2))

        pop = new_pop

    return best_so_far, best_hist, avg_hist, noise_hist

# ============================================================
# RANDOM SEARCH BASELINE
# ============================================================

def run_random_search(evals=600):
    best_hist = []
    noise_hist = []

    best_fit = -np.inf
    losses = []

    for i in range(evals):
        ind = random_individual()
        f = fitness(ind)
        losses.append(-f)

        if f > best_fit:
            best_fit = f

        best_hist.append(best_fit)
        noise_hist.append(np.std(losses))

    return best_hist, noise_hist

# ============================================================
# RUN EXPERIMENTS
# ============================================================

best_ga, ga_best, ga_avg, ga_noise = run_ga()
rs_best, rs_noise = run_random_search()

print("\nBest GA hyperparameters:")
print(f"  learning rate = {best_ga[0]:.5f}")
print(f"  batch size    = {best_ga[1]}")

# ============================================================
# SAVE DATA
# ============================================================

df = pd.DataFrame({
    "gen": range(len(ga_best)),
    "ga_best_fitness": ga_best,
    "ga_avg_fitness": ga_avg,
    "ga_noise": ga_noise
})
df.to_csv(f"{OUT}/ga_metrics.csv", index=False)

# ============================================================
# PLOTS
# ============================================================

plt.figure()
plt.plot([-f for f in ga_best], label="GA best")
plt.plot([-f for f in ga_avg], label="GA average")
plt.plot([-f for f in rs_best[:len(ga_best)]], label="Random search")
plt.xlabel("Generation")
plt.ylabel("Validation loss")
plt.title("GA vs Random Search")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT}/comparison.png", dpi=220)
plt.close()

plt.figure()
plt.plot(ga_noise, label="GA noise")
plt.plot(rs_noise[:len(ga_noise)], label="Random search noise")
plt.xlabel("Generation")
plt.ylabel("Loss standard deviation")
plt.title("Stability / Noise Analysis")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT}/stability.png", dpi=220)
plt.close()

print(f"\n📁 All outputs saved in: {OUT}")
