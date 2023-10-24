import numpy as np
from scipy.stats import poisson, norm, expon
from scipy.optimize import minimize_scalar

# Parametrii
lambda_ = 20
mu_normal = 2
sigma_normal = 0.5
desired_probability = 0.95
max_service_time = 15


def total_service_time(alpha):
    poisson_pmf = poisson.pmf(np.arange(0, max_service_time // mu_normal + 1), lambda_)

    normal_pdf = norm.pdf(np.arange(0, max_service_time + 1), mu_normal, sigma_normal)

    expon_pdf = expon.pdf(np.arange(0, max_service_time + 1), scale=alpha)

    convolution = np.convolve(poisson_pmf, normal_pdf, mode='full')
    convolution = np.convolve(convolution, expon_pdf, mode='full')

    probability = 1 - np.sum(convolution[:max_service_time])

    return probability


result = minimize_scalar(lambda alpha: -total_service_time(alpha), bounds=(0, 15), method='bounded')
optimal_alpha = result.x
optimal_probability = total_service_time(optimal_alpha)
rho = lambda_ / (1 / optimal_alpha)
L = (rho**1 / 1) * rho / ((1 - rho)**2)
average_waiting_time = L / lambda_

print(f"Alpha maxim pentru a servi toți clienții într-o oră cu o probabilitate de {desired_probability * 100}% este: {optimal_alpha:.2f} minute")
print(f"Probabilitatea corespunzătoare este: {optimal_probability * 100:.2f}%")
print(f"Timpul mediu de așteptare pentru a fi servit al unui client este: {average_waiting_time:.2f} minute")
