import pymc as pm
import pandas as pd
import numpy as np

# Citirea datelor
data = pd.read_csv('Prices.csv')

# Definirea modelului
with pm.Model() as model:
    # A priori pentru parametri
    alpha = pm.Normal('alpha', mu=0, tau=1.0 / 10**2)
    beta1 = pm.Normal('beta1', mu=0, tau=1.0 / 10**2)
    beta2 = pm.Normal('beta2', mu=0, tau=1.0 / 10**2)
    sigma = pm.Uniform('sigma', lower=0, upper=10)

    # Modelul de regresie
    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])

    # A priori pentru datele observate
    price = pm.Normal('price', mu=mu, tau=1.0 / sigma**2, observed=True, value=data['Price'])

    # MCMC sampling
    trace = pm.sample(2000, burn=1000, thin=1)

# Estimarea HDI pentru beta1 și beta2
hdi_beta1 = np.percentile(trace['beta1'], [2.5, 97.5])
hdi_beta2 = np.percentile(trace['beta2'], [2.5, 97.5])

print("Estimarea HDI pentru beta1:", hdi_beta1)
print("Estimarea HDI pentru beta2:", hdi_beta2)

# Evaluarea utilității predictorilor
beta1_in_HDI = (0 >= hdi_beta1[0]) and (0 <= hdi_beta1[1])
beta2_in_HDI = (0 >= hdi_beta2[0]) and (0 <= hdi_beta2[1])

print("Frecvența procesorului este un predictor util?", beta1_in_HDI)
print("Mărimea hard diskului este un predictor util?", beta2_in_HDI)

# Simulare extrageri din distribuția așteptată a prețului
new_data = {'Speed': [33], 'HardDrive': [np.log(540)]}
with model:
    post_pred = pm.sample(5000, vars=[price], new_values=new_data)

# Construirea intervalului HDI pentru preț
hdi_price = np.percentile(post_pred['price'], [5, 95])

print("Intervalul HDI pentru prețul așteptat:", hdi_price)

# Simulare extrageri din distribuția predictivă posterioară
with model:
    post_pred_full = pm.sample(5000)

# Construirea intervalului HDI pentru predicție
hdi_pred = np.percentile(post_pred_full['price'], [5, 95])

print("Intervalul HDI pentru predicție:", hdi_pred)
