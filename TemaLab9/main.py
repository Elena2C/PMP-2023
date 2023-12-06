import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az


def read_data():
    file_path = 'Admission.csv'
    df = pd.read_csv(file_path)
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
    df = df.dropna()
    return df['GRE'].values, df['GPA'].values, df['Admission'].values.astype(int)


def plot_data(gre, gpa, admission):
    plt.scatter(gre, gpa, c=admission, cmap='viridis', edgecolors='k', marker='o', s=50)
    plt.xlabel('GRE')
    plt.ylabel('GPA')
    plt.title('Admission Data')
    plt.show()


def main():
    gre, gpa, admission = read_data()
    plot_data(gre, gpa, admission)

    with pm.Model() as logistic_model:
        beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
        beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
        beta_2 = pm.Normal('beta_2', mu=0, sigma=10)
        pm.Bernoulli('observed_data', p=pm.math.sigmoid(beta_0 + beta_1 * gre + beta_2 * gpa), observed=admission)
        trace = pm.sample(2000, tune=1000, cores=1)

    # Plot trace
    az.plot_trace(trace, var_names=['beta_0', 'beta_1', 'beta_2'])
    plt.show()

    # Decision boundary and 94% HDI
    beta_0_samples = trace['beta_0']
    beta_1_samples = trace['beta_1']
    beta_2_samples = trace['beta_2']
    bd_hdi = pm.hdi(-beta_0_samples / beta_1_samples)

    plt.plot(gre, (-beta_0_samples.mean() - beta_1_samples.mean() * gre) / beta_2_samples.mean(), color='C2', lw=3)
    plt.fill_betweenx([gpa.min(), gpa.max()], bd_hdi[0] / beta_2_samples.mean(), bd_hdi[1] / beta_2_samples.mean(),
                      color='k', alpha=0.5)
    plt.scatter(gre, gpa, c=admission, cmap='viridis', edgecolors='k', marker='o', s=50)
    plt.xlabel('GRE')
    plt.ylabel('GPA')
    plt.title('Decision Boundary and 94% HDI')
    plt.show()

    # Probability of admission for a student with GRE 550 and GPA 3.5
    new_student_data = {'GRE': 550, 'GPA': 3.5}
    p_admission = 1.0 / (1.0 + np.exp(
        -(beta_0_samples + beta_1_samples * new_student_data['GRE'] + beta_2_samples * new_student_data['GPA'])))
    hdi_90 = pm.hdi(p_admission)

    print(f"Intervalul de 90% HDI pentru probabilitatea admiterii pentru studentul dat: {hdi_90}")


if __name__ == "__main__":
    np.random.seed(1)
    main()
