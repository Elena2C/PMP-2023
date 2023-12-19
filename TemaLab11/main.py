import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

np.random.seed(42)
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
plt.hist(mix, bins=30, density=True, alpha=0.5)

data = mix

with pm.Model() as model:
    # Model with 2 components
    w_2 = pm.Dirichlet('w_2', a=np.ones(clusters))
    mu_2 = pm.Normal.dist(mu=np.zeros(clusters), tau=1, size=clusters)
    components_2 = [pm.Normal.dist(mu=mu_2, tau=1) for _ in range(clusters)]
    obs_model_2 = pm.Mixture('obs_model_2', w=w_2, comp_dists=components_2, observed=True, value=data)

    trace_2 = pm.sample(2000, tune=1000)

    # Model with 3 components
    w_3 = pm.Dirichlet('w_3', a=np.ones(clusters))
    mu_3 = pm.Normal.dist(mu=np.zeros(clusters), tau=1, size=clusters)
    components_3 = [pm.Normal.dist(mu=mu_3, tau=1) for _ in range(clusters)]
    obs_model_3 = pm.Mixture('obs_model_3', w=w_3, comp_dists=components_3, observed=True, value=data)

    trace_3 = pm.sample(2000, tune=1000)

    # Model with 4 components
    w_4 = pm.Dirichlet('w_4', a=np.ones(clusters))
    mu_4 = pm.Normal.dist(mu=np.zeros(clusters), tau=1, size=clusters)
    components_4 = [pm.Normal.dist(mu=mu_4, tau=1) for _ in range(clusters)]
    obs_model_4 = pm.Mixture('obs_model_4', w=w_4, comp_dists=components_4, observed=True, value=data)

    trace_4 = pm.sample(2000, tune=1000)

pm.traceplot(trace_2)
pm.traceplot(trace_3)
pm.traceplot(trace_4)
plt.show()
