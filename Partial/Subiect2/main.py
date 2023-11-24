import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(42)
timp_mediu_asteptare = np.random.normal(loc=10, scale=2, size=200)

model = pm.Model()

with model:
    miu = pm.Normal('miu', mu=10, sigma=5)
    sigma = pm.HalfNormal('sigma', sd=5)

    likelihood = pm.Normal('likelihood', mu=miu, sigma=sigma, observed=timp_mediu_asteptare)

with model:
    trace = pm.sample(2000, tune=1000, cores=2)

az.plot_posterior(trace, var_names=['sigma'], credible_interval=0.95)
plt.title('Distribuția a posteriori pentru sigma')
plt.show()
