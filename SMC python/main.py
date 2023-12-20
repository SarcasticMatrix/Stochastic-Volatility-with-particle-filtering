
from Model import Model
from SVModel import SVModel

# Crée une instance de SVModel avec des paramètres spécifiques
myModel = SVModel(T=1000, N=10000, alpha=1, seed=100)
x, y = myModel.generate_trajectory()

# Affiche le graphique de la trajectoire
#myModel.plot_trajectory()

# Calcule l'estimation
myModel.compute_estimation()

# Affiche l'estimation
#myModel.plot_estimation()


import matplotlib.pyplot as plt
import numpy as np

cdfT,x = myModel.cdf_t(100,np.arange(0,50,0.1))
plt.figure()
plt.plot(cdfT)
plt.show()

pdfT = myModel.pdf_t(100, x)
plt.figure()
plt.plot(x,pdfT)
plt.show()

probabilities = np.array([0.25,0.75])  # Remplacez par vos probabilités

from tqdm import tqdm
list_lower_bound, list_upper_bound = [], []

for t in tqdm(range(myModel.T)):
    print(myModel.quantile_t(t,0.25))

    lower_bound = myModel.quantile_t(t,0.25)
    upper_bound = myModel.quantile_t(t,0.75)
    list_lower_bound.append(lower_bound)
    list_upper_bound.append(upper_bound)

plt.figure(figsize=(10, 6))
plt.plot(myModel.time, myModel.estimation, label="Estimation")
plt.fill_between(myModel.time, list_lower_bound, list_upper_bound, color='gray', alpha=0.2, label=f"{0.95 * 100}% Confidence Interval")
            
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Estimation with Confidence Interval")
plt.legend()
plt.show()