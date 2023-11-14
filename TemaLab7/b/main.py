import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('auto-mpg.csv')
df = data[['horsepower', 'mpg']]

df = df[pd.to_numeric(df['horsepower'], errors='coerce').notna()]

df['horsepower'] = pd.to_numeric(df['horsepower'])

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=1)

    linear_combination = alpha + beta * df['horsepower']
    mu = pm.Deterministic('mu', linear_combination)

    mpg_obs = pm.Normal('mpg_obs', mu=mu, sd=sigma, observed=df['mpg'])

    trace = pm.sample(2000, tune=1000)

az.plot_posterior(trace)
plt.show()
