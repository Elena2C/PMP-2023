import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def posterior_grid(grid_points=50, heads=6, tails=9, prior=None):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    if prior is None:
        prior = np.repeat(1/grid_points, grid_points)  # default: uniform prior

    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


# Datele observate
data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h

# Distribuție a priori uniformă
prior_uniform = np.repeat(1/points, points)
grid_uniform, posterior_uniform = posterior_grid(points, h, t, prior_uniform)

# Distribuție a priori bazată pe valori mai mici sau egale cu 0.5
prior_less_than_equal_0_5 = (grid_uniform <= 0.5).astype(int)
grid_less_than_equal_0_5, posterior_less_than_equal_0_5 = posterior_grid(points, h, t, prior_less_than_equal_0_5)

# Distribuție a priori bazată pe distanța față de 0.5
prior_distance_from_0_5 = np.abs(grid_uniform - 0.5)
grid_distance_from_0_5, posterior_distance_from_0_5 = posterior_grid(points, h, t, prior_distance_from_0_5)

# Plotați rezultatele
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(grid_uniform, posterior_uniform, 'o-')
plt.title('Uniform Prior')
plt.xlabel('θ')
plt.ylabel('Posterior Probability')

plt.subplot(132)
plt.plot(grid_less_than_equal_0_5, posterior_less_than_equal_0_5, 'o-')
plt.title('Prior <= 0.5')
plt.xlabel('θ')
plt.ylabel('Posterior Probability')

plt.subplot(133)
plt.plot(grid_distance_from_0_5, posterior_distance_from_0_5, 'o-')
plt.title('Prior based on Distance from 0.5')
plt.xlabel('θ')
plt.ylabel('Posterior Probability')

plt.tight_layout()
plt.show()
