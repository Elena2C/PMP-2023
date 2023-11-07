import numpy as np
import pymc as pm

traffic_data = np.loadtxt('trafic.csv', delimiter=',', skiprows=1)

trafic_observed = traffic_data[:, 1]

with pm.Model() as model:
    lambda_1 = pm.Exponential('lambda_1', lam=1.0)
    lambda_2 = pm.Exponential('lambda_2', lam=1.0)
    lambda_3 = pm.Exponential('lambda_3', lam=1.0)
    lambda_4 = pm.Exponential('lambda_4', lam=1.0)
    lambda_5 = pm.Exponential('lambda_5', lam=1.0)

    trafic_model = pm.Poisson('trafic_model', mu=pm.math.switch(
        traffic_data[:, 0] < 180, lambda_1, pm.math.switch(
            traffic_data[:, 0] < 240, lambda_2, pm.math.switch(
                traffic_data[:, 0] < 960, lambda_3, pm.math.switch(
                    traffic_data[:, 0] < 1201, lambda_4, lambda_5
                )
            )
        )
    ), observed=trafic_observed)

    trace = pm.sample(2000, tune=1000, target_accept=0.9)
