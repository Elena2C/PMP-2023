from scipy.stats import beta, binom
import numpy as np


def metropolis_beta_binomial(draws=10000, alpha_prior=2, beta_prior=2):
    trace = np.zeros(draws)
    old_theta = 0.5
    old_prob = binom.pmf(6, 13, old_theta) * beta.pdf(old_theta, alpha_prior, beta_prior)

    for i in range(draws):
        new_theta = old_theta + np.random.normal(0, 0.1)
        new_prob = binom.pmf(6, 13, new_theta) * beta.pdf(new_theta, alpha_prior, beta_prior)

        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_theta
            old_theta = new_theta
            old_prob = new_prob
        else:
            trace[i] = old_theta

    return trace
