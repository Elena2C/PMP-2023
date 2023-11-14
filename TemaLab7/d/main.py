import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('auto-mpg.csv')
data = data[['horsepower', 'mpg']].dropna()

plt.scatter(data['horsepower'], data['mpg'], alpha=0.5)
plt.xlabel('Horsepower (HP)')
plt.ylabel('Miles per Gallon (MPG)')
plt.title('Relatia dintre horsepower si mpg')
plt.show()

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    linear_combination = alpha + beta * data['horsepower']
    mu = pm.Deterministic('mu', linear_combination)

    mpg_obs = pm.Normal('mpg_obs', mu=mu, sd=sigma, observed=data['mpg'])

    trace = pm.sample(2000, tune=1000)

az.plot_posterior(trace, hdi_prob=0.95, var_names=['alpha', 'beta', 'sigma'])
plt.show()
