import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

lambda1 = 4
lambda2 = 6

ratio = 1.5

p_first_mechanic = 0.4

n_samples = 10000
samples = []

for _ in range(n_samples):
    random_num = np.random.rand()

    if random_num < p_first_mechanic:
        time = stats.expon(scale=1 / lambda1).rvs()
    else:
        time = stats.expon(scale=1 / lambda2).rvs()

    samples.append(time)

data = {"X": np.array(samples)}

az.plot_posterior(data)
plt.show()

