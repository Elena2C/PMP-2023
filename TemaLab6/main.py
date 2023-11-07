import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

observed_data = np.array(Y_values)

model = pm.Model()

with model:
    n = pm.Poisson("n", mu=10)

    for Y in Y_values:
        for theta in theta_values:
            Y_obs = pm.Binomial(f"Y_obs_{Y}_{theta}", n=n, p=theta, observed=Y)

    step = pm.Metropolis()
    trace = pm.sample(2000, tune=1000, step=step, cores=1)

az.plot_posterior(trace)
plt.show()
