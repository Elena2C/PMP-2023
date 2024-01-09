import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return pi, error


# Numărul de încercări
num_trials = 100

# Numerele de puncte
N_values = [100, 1000, 10000]

# Listele pentru stocarea rezultatelor
pi_estimates = []
errors = []

# Rulați codul de mai multe ori pentru fiecare valoare a lui N
for N in N_values:
    pi_vals = []
    error_vals = []
    for _ in range(num_trials):
        pi, error = estimate_pi(N)
        pi_vals.append(pi)
        error_vals.append(error)

    # Calculați media și deviația standard a erorii
    mean_error = np.mean(error_vals)
    std_error = np.std(error_vals)

    # Stocați rezultatele în listele principale
    pi_estimates.append(pi_vals)
    errors.append((mean_error, std_error))

# Vizualizați rezultatele folosind plt.errorbar()
plt.figure(figsize=(10, 6))

for i, N in enumerate(N_values):
    plt.errorbar(N, errors[i][0], yerr=errors[i][1], fmt='o', label=f'N = {N}')

plt.xscale('log')  # Pentru a vizualiza mai bine pe scală logaritmică
plt.xlabel('Number of Points (N)')
plt.ylabel('Error (%)')
plt.title('Estimation of π with Error Bars')
plt.legend()
plt.show()
