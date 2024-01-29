import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def plot_posterior(grid, posterior, h, t):
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')
    plt.show()


def find_max_posterior_theta(grid, posterior):
    theta_max_posterior = grid[np.argmax(posterior)]
    return theta_max_posterior


data = np.repeat([0, 1], (10, 3))
h = data.sum()
t = len(data) - h
points = 100
grid, posterior = posterior_grid(points, h, t)

plot_posterior(grid, posterior, h, t)

theta_max_posterior = find_max_posterior_theta(grid, posterior)
print(f"θ care maximizează probabilitatea a posteriori: {theta_max_posterior}")
