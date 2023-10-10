import numpy as np
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

alpha = [4, 4, 5, 5]
lambda_ = [3, 2, 2, 3]

prob_server = [0.25, 0.25, 0.30, 0.20]

n_simulations = 10000

def total_service_time():
    server_choice = np.random.choice([0, 1, 2, 3], p=prob_server)
    return np.random.gamma(alpha[server_choice], scale=1/lambda_[server_choice]) + np.random.exponential(scale=4)

service_times = [total_service_time() for _ in range(n_simulations)]

prob = np.mean(np.array(service_times) > 3)

print(f"Probabilitatea ca timpul necesar servirii unui client (X) să fie mai mare de 3 milisecunde: {prob:.4f}")

data = {"X": np.array(service_times)}

az.plot_posterior(data)
plt.show()