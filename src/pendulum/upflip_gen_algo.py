from double_pendulum import DoublePendulum
import numpy as np

h = 0.005
T = 10

N = int(T / h)

t = np.linspace(0, T, N)

gene_size = N // 100
n_genes = N // gene_size

n_population = 48
half_population_even = n_population // 2
if not (half_population_even % 2 == 0):
    half_population_even -= 1

n_mutations = N // 20
mutation_indices = np.arange(N)

mutation_sigma = 0.01

# Randomize accelerations
xpp = np.random.random((n_population, N)) * 100 - 50

theta_1 = np.zeros(n_population)
theta_2 = np.zeros(n_population)
theta_1_p = np.zeros(n_population)
theta_2_p = np.zeros(n_population)

m1, L1, m2, L2 = 1.0, 1.0, 1.0, 1.0

double_pendula = DoublePendulum(m1, L1, m2, L2, theta_1_0=theta_1, theta_2_0=theta_2, theta_1_p_0=theta_1_p, theta_2_p_0=theta_2_p)

N_generations = 100

max_int = -np.inf
best = np.zeros(N)

for gen in range(N_generations):
    double_pendula.reset()

    y2_integral = np.zeros(n_population)

    for i in range(N):
        double_pendula.x0_pp = xpp[:, i]

        double_pendula.runge_kutta_4(h)

        y1 = -L1 * np.cos(double_pendula.theta_1)
        y2 = y1 - L2 * np.cos(double_pendula.theta_2)

        y2_integral += y2 * h

    reproducers = np.argsort(y2_integral)[::-1][:half_population_even]

    this_max = max(y2_integral)
    if this_max > max_int:
        best = xpp[y2_integral.argmax()]
        max_int = this_max

    reproducers = xpp[reproducers]

    print(f"Generation {gen}: max int(y2) = {np.max(y2_integral)}")
    print(f"Best yet: {max_int}")

    xpp_new = []

    for i in range(0, half_population_even, 2):
        mutated_parent_1 = reproducers[i]
        mutated_parent_1[np.random.choice(mutation_indices, n_mutations, replace=False)] *= np.random.randn(n_mutations) * mutation_sigma + 1.0
        mutated_parent_2 = reproducers[i + 1]
        mutated_parent_2[np.random.choice(mutation_indices, n_mutations, replace=False)] *= np.random.randn(n_mutations) * mutation_sigma + 1.0

        xpp_new.append(mutated_parent_1)
        xpp_new.append(mutated_parent_2)

        child_1_chunks = np.random.choice([0, 1], n_genes)
        child_2_chunks = (~(child_1_chunks.astype(bool))).astype(int) + i
        child_1_chunks += i

        child_1 = np.zeros_like(mutated_parent_1)
        child_2 = np.zeros_like(mutated_parent_2)

        for gene in range(n_genes):
            lower = gene * gene_size
            upper = (gene + 1) * gene_size
            child_1[lower:upper] = reproducers[child_1_chunks[gene]][lower:upper]
            child_2[lower:upper] = reproducers[child_2_chunks[gene]][lower:upper]

        child_1[np.random.choice(mutation_indices, n_mutations, replace=False)] *= np.random.randn(n_mutations) * mutation_sigma + 1.0
        child_2[np.random.choice(mutation_indices, n_mutations, replace=False)] *= np.random.randn(n_mutations) * mutation_sigma + 1.0

        xpp_new.extend([child_1, child_2])

    xpp = np.array(xpp_new)
        
np.save("best.npy", best)
