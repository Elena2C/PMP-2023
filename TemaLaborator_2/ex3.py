import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

num_aruncari = 10
num_experiente = 100

prob_stema = 0.3

rezultate_ss = np.zeros(num_experiente)
rezultate_sb = np.zeros(num_experiente)
rezultate_bs = np.zeros(num_experiente)
rezultate_bb = np.zeros(num_experiente)

for i in range(num_experiente):
    aruncari_moneda1 = np.random.choice(['s', 'b'], size=num_aruncari, replace=True)
    aruncari_moneda2 = np.random.choice(['s', 'b'], size=num_aruncari, replace=True, p=[1 - prob_stema, prob_stema])

    rezultate_ss[i] = np.sum(np.logical_and(aruncari_moneda1 == 's', aruncari_moneda2 == 's'))
    rezultate_sb[i] = np.sum(np.logical_and(aruncari_moneda1 == 's', aruncari_moneda2 == 'b'))
    rezultate_bs[i] = np.sum(np.logical_and(aruncari_moneda1 == 'b', aruncari_moneda2 == 's'))
    rezultate_bb[i] = np.sum(np.logical_and(aruncari_moneda1 == 'b', aruncari_moneda2 == 'b'))

# distribuția pentru rezultatele 'ss'
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.hist(rezultate_ss, bins=range(num_aruncari + 2), density=True, alpha=0.6, color='g')
plt.title("Distribuția 'ss'")
plt.xlabel('Numărul de apariții')
plt.ylabel('Densitate')

# distribuția pentru rezultatele 'sb'
plt.subplot(222)
plt.hist(rezultate_sb, bins=range(num_aruncari + 2), density=True, alpha=0.6, color='b')
plt.title("Distribuția 'sb'")
plt.xlabel('Numărul de apariții')
plt.ylabel('Densitate')

# distribuția pentru rezultatele 'bs'
plt.subplot(223)
plt.hist(rezultate_bs, bins=range(num_aruncari + 2), density=True, alpha=0.6, color='r')
plt.title("Distribuția 'bs'")
plt.xlabel('Numărul de apariții')
plt.ylabel('Densitate')

# distribuția pentru rezultatele 'bb'
plt.subplot(224)
plt.hist(rezultate_bb, bins=range(num_aruncari + 2), density=True, alpha=0.6, color='y')
plt.title("Distribuția 'bb'")
plt.xlabel('Numărul de apariții')
plt.ylabel('Densitate')

plt.tight_layout()
plt.show()
