import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

# Define observed data
observed_data = np.array(Y_values)

# Create a PyMC model
model = pm.Model()

with model:
    # Prior for n, Poisson distribution with mean 10
    n = pm.Poisson("n", mu=10)

    # Likelihood, binomial distribution for Y given θ
    for Y in Y_values:
        for theta in theta_values:
            Y_obs = pm.Binomial(f"Y_obs_{Y}_{theta}", n=n, p=theta, observed=Y)

    # Bayesian inference
    step = pm.Metropolis()  # You can choose a different step method if needed
    trace = pm.sample(2000, tune=1000, step=step, cores=1)

# Visualize results
az.plot_posterior(trace)
plt.show()
