import pandas as pd
import pymc as pm

data = pd.read_csv("BostonHousing.csv")

model = pm.Model()

with model:
    rm = pm.Data("rm", data["rm"])
    crim = pm.Data("crim", data["crim"])
    indus = pm.Data("indus", data["indus"])

    medv = pm.Data("medv", data["medv"])

    beta_rm = pm.Normal("beta_rm", mu=0, tau=1)
    beta_crim = pm.Normal("beta_crim", mu=0, tau=1)
    beta_indus = pm.Normal("beta_indus", mu=0, tau=1)

    alpha = pm.Normal("alpha", mu=0, tau=1)

    mu = alpha + beta_rm*rm + beta_crim*crim + beta_indus*indus

    obs = pm.Normal("obs", mu=mu, tau=1, observed=medv)

n_samples = 1000

with model:
    step = pm.NUTS()
    trace = pm.sample(n_samples, tune=1000, step=step)

with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=n_samples)

y_pred_mean = post_pred["obs"].mean(axis=0)

lower_bound, upper_bound = pm.stats.hdi(post_pred["obs"], hdi_prob=0.5)

print("Interval de predictie de 50% HDI pentru valoarea locuintelor:")
print("Lower Bound:", lower_bound.mean())
print("Upper Bound:", upper_bound.mean())

pm.summary(trace, hdi_prob=0.95)

print("Coeficientii modelului:")
print("Beta_rm:", trace["beta_rm"].mean())
print("Beta_crim", trace["beta_crim"].mean())
print("Beta_indus", trace["beta_indus"].mean())
