import numpy as np
import matplotlib.pyplot as plt

# Parametre
N = 1000          # počet rôznych slov
alpha = 1.0       # Zipfov exponent
T = 10000         # počet vygenerovaných slov

# Zipfova distribúcia
rank = np.arange(1, N + 1)
p = 1 / rank**alpha
p = p / p.sum()

# Generovanie frekvencií slov
freq = np.random.multinomial(T, p)

# Zoradenie frekvencií
freq = np.sort(freq)[::-1]

# Graf
plt.loglog(rank, freq, 'o', markersize=3, label="Simulácia")
plt.loglog(rank, T * p, 'r-', label="Teória")

plt.xlabel("Poradie slova (rank)")
plt.ylabel("Frekvencia")
plt.legend()
plt.grid(True)
plt.show()