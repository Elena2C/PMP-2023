from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
import numpy as np

model = BayesianNetwork([('Coin_p0', 'Outcome_p0'),
                       ('Coin_p1', 'Outcome_p1'),
                       ('Outcome_p0', 'Win_p0'),
                       ('Outcome_p1', 'Win_p1')])

data = np.random.randint(2, size=(20000, 4))
data[:, 2] = np.random.choice([0, 1], size=20000)
data[:, 3] = np.random.choice([0, 1], size=20000)

model.fit(data, estimator=ParameterEstimator)

inference = VariableElimination(model)
result = inference.query(variables=['Outcome_p0', 'Outcome_p1', 'Win_p0', 'Win_p1'],
                         evidence={'Outcome_p1': 0, 'Outcome_p0': 0, 'Win_p0': 0, 'Win_p1': 0})
print(result)
