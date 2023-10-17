import pymc3 as pm


model = pm.Model()

with model:
    E = pm.Bernoulli('E', p=0.0005)
    F = pm.Bernoulli('F', p=0.01)
    A = pm.Bernoulli('A', p=0.01)

    F_cond = pm.Deterministic('F_cond', pm.math.switch(E, 0.03, F))

    A_cond = pm.Deterministic('A_cond', pm.math.switch(E, 0.02, 0.01))

with model:
    trace = pm.sample(10000, chains=1)

alarm_triggered = trace['A'] == 1
earthquake_given_alarm = trace['E'][alarm_triggered].mean()
print("Probability of earthquake given alarm: {:.4f}".format(earthquake_given_alarm))

no_alarm = trace['A'] == 0
fire_no_alarm = trace['F_cond'][no_alarm].mean()
print("Probability of fire without alarm: {:.4f}".format(fire_no_alarm))
