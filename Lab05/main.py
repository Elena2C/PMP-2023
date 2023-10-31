import pymc as pm
import numpy as np

traffic_data = np.loadtxt('traffic.csv', delimiter=',')

intervals = [(4, 7), (7, 8), (8, 16), (16, 19), (19, 24)]

with pm.Model() as model:
    lambdas = []
    for start, end in intervals:
        lambda_ = pm.Uniform(f'lambda_{start}_{end}', 0, 10)
        lambdas.append(lambda_)

    poisson_values = []
    for i in range(len(intervals)):
        start, end = intervals[i]
        poisson = pm.Poisson(f'poisson_{start}_{end}', mu=lambdas[i], observed=traffic_data[(traffic_data[:, 0] >= start) & (traffic_data[:, 0] < end)][:, 1])
        poisson_values.append(poisson)

with model:
    trace = pm.sample(2000, cores=2, target_accept=0.95)

pm.summary(trace)
