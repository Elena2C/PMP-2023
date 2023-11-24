from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = BayesianModel([('Player', 'Outcome_p0'),
                       ('Outcome_p0', 'Win_p0'),
                       ('Outcome_p0', 'Outcome_p1'),
                       ('Outcome_p1', 'Win_p1')])


data = np.zeros((20000, 4), dtype=int)
data[:, 0] = np.random.choice([0, 1], size=20000)
data[:, 1] = np.random.choice([0, 1], size=20000)
data[:, 2] = np.random.choice([0, 1], size=20000)
data[:, 3] = np.random.choice([0, 1], size=20000)

cpd_player = TabularCPD(variable='Player', variable_card=2, values=[[1 / 2], [1 / 2]])
cpd_outcome_p0 = TabularCPD(variable='Outcome_p0', variable_card=2, values=[[1 / 3, 2 / 3], [1 / 3, 2 / 3]],
                            evidence=['Player'], evidence_card=[2])
cpd_outcome_p1 = TabularCPD(variable='Outcome_p1', variable_card=2, values=[[1 / 2, 1 / 2], [1 / 2, 1 / 2]],
                            evidence=['Outcome_p0'], evidence_card=[2])
cpd_win_p0 = TabularCPD(variable='Win_p0', variable_card=2, values=[[1, 0], [0, 1]],
                        evidence=['Outcome_p0', 'Outcome_p1'], evidence_card=[2, 2])
cpd_win_p1 = TabularCPD(variable='Win_p1', variable_card=2, values=[[0, 1], [1, 0]],
                        evidence=['Outcome_p0', 'Outcome_p1'], evidence_card=[2, 2])

model.add_cpds(cpd_player, cpd_outcome_p0, cpd_outcome_p1, cpd_win_p0, cpd_win_p1)

inference = VariableElimination(model)
result_p0_wins = inference.query(variables=['Win_p0'], evidence=None)
result_p1_wins = inference.query(variables=['Win_p1'], evidence=None)

print("Player 0's chances of winning:", result_p0_wins.values[1])
print("Player 1's chances of winning:", result_p1_wins.values[1])

result_coin_face = inference.query(variables=['Outcome_p0'], evidence={'Outcome_p1': 0})
print(result_coin_face)
