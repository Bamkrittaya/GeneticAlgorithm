import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# ============================================================
# SETUP
# ============================================================

random.seed(42)
np.random.seed(42)
plt.rcParams["figure.figsize"] = (6, 4)

# Create output directory
OUT = "ga_outputs"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. DATASET (Initial Population)
# ============================================================

candidates = ["A", "B", "C", "D", "E"]
initial_x = np.array([-8, -1, 3, 9, 14])

data = pd.DataFrame({"Candidate": candidates, "x": initial_x})
print("\n=== Initial Dataset ===")
print(data)
data.to_csv(f"{OUT}/dataset.csv", index=False)

# ============================================================
# 2. INITIAL POPULATION PLOT (SAVE)
# ============================================================

plt.scatter(initial_x, np.zeros_like(initial_x), marker="s")
for label, x in zip(candidates, initial_x):
    plt.text(x, 0.1, label, ha="center")

plt.yticks([])
plt.xlabel("x (candidate value)")
plt.title("Initial Population (Positions)")
plt.grid(True, axis="x", linestyle="--", alpha=0.3)
plt.savefig(f"{OUT}/initial_population.png", dpi=220)
plt.close()

# ============================================================
# 3. REPRESENTATION: BINARY ENCODING
# ============================================================

BITS = 5
OFFSET = 16  # maps [-16, 15] → [0, 31]


def encode(x: int) -> str:
    return format(x + OFFSET, f"0{BITS}b")


def decode(ch: str) -> int:
    return int(ch, 2) - OFFSET


rep_table = pd.DataFrame({
    "Candidate": candidates,
    "x": initial_x,
    "Chromosome": [encode(int(x)) for x in initial_x]
})

print("\n=== Representation Table ===")
print(rep_table)
rep_table.to_csv(f"{OUT}/representation.csv", index=False)

# ============================================================
# 4. CROSSOVER & MUTATION ILLUSTRATION
# ============================================================

parent1 = encode(initial_x[0])
parent2 = encode(initial_x[1])

cx_point = 3
child = parent1[:cx_point] + parent2[cx_point:]

mut_point = 1
child_mut = list(child)
child_mut[mut_point] = "1" if child_mut[mut_point] == "0" else "0"
child_mut = "".join(child_mut)

print("\n=== Crossover and Mutation Demo ===")
print("Parent 1:", parent1)
print("Parent 2:", parent2)
print("Crossover at index:", cx_point)
print("Child before mutation:", child)
print("Mutation at index:", mut_point)
print("Child after mutation:", child_mut)

with open(f"{OUT}/crossover_mutation_demo.txt", "w") as f:
    f.write("Parent1: " + parent1 + "\n")
    f.write("Parent2: " + parent2 + "\n")
    f.write("Crossover point: " + str(cx_point) + "\n")
    f.write("Child before mutation: " + child + "\n")
    f.write("Mutation index: " + str(mut_point) + "\n")
    f.write("Child after mutation: " + child_mut + "\n")

# ============================================================
# 5. FITNESS FUNCTIONS
# ============================================================

def objective(x: int) -> float:
    return (x - 4)**2


def fitness(x: int) -> float:
    return -objective(x)


# ============================================================
# 6. FITNESS LANDSCAPE PLOT
# ============================================================

xs = np.arange(-16, 16)
ys = (xs - 4)**2

plt.plot(xs, ys)
plt.scatter(initial_x, (initial_x - 4)**2, color="red")
for label, x in zip(candidates, initial_x):
    plt.text(x, (x - 4)**2 + 1, label)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Fitness Landscape")
plt.grid(True)
plt.savefig(f"{OUT}/fitness_landscape.png", dpi=220)
plt.close()

# ============================================================
# 7. GA UTILITIES
# ============================================================

def roulette_selection(pop, fits):
    total = sum(fits)
    if total == 0:
        probs = [1/len(pop)] * len(pop)
    else:
        probs = [f/total for f in fits]
    return random.choices(pop, weights=probs, k=1)[0]


def crossover(p1, p2, p=0.9):
    if random.random() < p:
        pt = random.randint(1, BITS - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1, p2


def mutate(ch, p=0.1):
    bits = list(ch)
    for i in range(len(bits)):
        if random.random() < p:
            bits[i] = "1" if bits[i] == "0" else "0"
    return "".join(bits)

# ============================================================
# 8. RUN THE GENETIC ALGORITHM
# ============================================================

def run_ga(gens=25, pop_size=12, pc=0.9, pm=0.1):
    pop = [encode(int(x)) for x in np.random.randint(-16, 16, pop_size)]
    best_history = []
    fit_history = []

    report = []

    for g in range(gens):
        decoded = [decode(ch) for ch in pop]
        fits = [fitness(x) for x in decoded]

        best_idx = np.argmax(fits)
        best_x = decoded[best_idx]
        best_fit = fits[best_idx]

        best_history.append(best_x)
        fit_history.append(best_fit)

        line = f"Gen {g:02d} — Best x={best_x}, fitness={best_fit:.3f}"
        print(line)
        report.append(line)

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = roulette_selection(pop, fits)
            p2 = roulette_selection(pop, fits)

            c1, c2 = crossover(p1, p2, pc)
            c1 = mutate(c1, pm)
            c2 = mutate(c2, pm)

            new_pop += [c1, c2]

        pop = new_pop[:pop_size]

    # Save GA generation log
    with open(f"{OUT}/ga_generations.txt", "w") as f:
        f.write("\n".join(report))

    # Final best
    decoded = [decode(ch) for ch in pop]
    fits = [fitness(x) for x in decoded]
    best_idx = np.argmax(fits)
    return decoded[best_idx], fits[best_idx], best_history, fit_history


best_x, best_fit, best_hist, fit_hist = run_ga()

# ============================================================
# 9. CONVERGENCE CURVE
# ============================================================

plt.plot(fit_hist, marker="o")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence Curve")
plt.grid(True)
plt.savefig(f"{OUT}/convergence_curve.png", dpi=220)
plt.close()

# ============================================================
# 10. FINAL REPORT (SAVE)
# ============================================================

with open(f"{OUT}/final_report.txt", "w") as f:
    f.write("=== FINAL GA RESULT ===\n")
    f.write(f"Best x found: {best_x}\n")
    f.write(f"Objective f(x): {objective(best_x)}\n")
    f.write(f"Fitness: {best_fit}\n\n")
    f.write("=== Interpretation ===\n")
    if abs(best_x - 4) <= 1:
        f.write("The GA found a solution very close to the global minimum (x=4).\n")
    else:
        f.write("The GA converged to a reasonably good solution.\n")

print("\nAll outputs saved inside:", OUT)
