import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import hyp2f1
import pandas as pd

np.random.seed(42)

area = 675.409
densidades_cells = [2.6576489208760914, 0.6040784176698859, 0.3079615462630791, 0.2842721965505346, 0.029611687140680683, 0.0059223374281361365, 0.0059223374281361365, 0.004441753071102103]
potencia = [40, 20, 200, 0.0355, 0.251, 0.501, 0.221, 31.6]
quantidade = [1795, 408, 208, 192, 20, 4, 4, 3]
alpha = [3, 3, 4, 2.5, 2.5, 2.5, 2.5, 3]

dados = pd.DataFrame({'Potência (W)': potencia, 'Quantidade': quantidade,
                      'Densidade de células da camada': densidades_cells,
                      'Expoente de Perda': alpha})
dados.set_index('Potência (W)', inplace=True)

lambda_u = 100000/area

x_min, x_max = 0, 16
y_min, y_max = 0, 42
region_area = (x_max - x_min) * (y_max - y_min)

expected_num_cell_points = [int(region_area * density) for density in densidades_cells]
print("Número esperado de células por tipo de célula (considerando a área aproximada):", expected_num_cell_points)
num_points_per_cell = [np.random.poisson(lam) for lam in expected_num_cell_points]
print("Número de células por tipo de célula calculado pelo processo de Poisson:", num_points_per_cell)

points = [np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(n_points, 2))
          for n_points in num_points_per_cell]

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']


T_values = np.sort(np.unique(np.random.uniform(0, 20, 100))).tolist()

def Z(T, alpha_j):
    a = 1
    b = 1 - (2 / alpha_j)
    c = 2 - (2 / alpha_j)
    z = -T
    result = (2 * T) / (alpha_j - 2) * hyp2f1(a, b, c, z)
    return result


def Cj(lambda_j, Pj, Pk, alpha_j, Zj):
    Cj = lambda_j * ((Pj / Pk) ** (2 / alpha_j) + Zj)
    return Cj


def integrando(r, alphas, lambdas, potencias, k, T):
    soma_Cj = 0
    for j in range(len(potencias)):
        Cj_value = Cj(lambdas[j], potencias[j], potencias[k], alphas[j], Z(T, alphas[j]))
        soma_Cj += Cj_value * r ** (2 / (alphas[j] / alphas[k]))
    return r * np.exp(-np.pi * soma_Cj)

Pkc = [[] for _ in range(len(densidades_cells))]

#1
for T in T_values:
    pkc_values = []
    for k in range(len(densidades_cells)):
        lambda_k = densidades_cells[k]
        integral_result, _ = quad(integrando, 0, np.inf, args=(alpha, densidades_cells, potencia, k, T))
        pkc_value = 2 * np.pi * lambda_k * integral_result
        pkc_values.append(pkc_value)

    for i in range(len(pkc_values)):
        Pkc[i].append(pkc_values[i])

fig, axes = plt.subplots(2, 4, figsize=(14, 16))
axes = axes.flatten()

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']
for i in range(len(Pkc)):
    axes[i].plot(T_values, Pkc[i], marker='o', linestyle='-', color=colors[i], label=f'Standard: Layer {i + 1}', markersize=5)
    axes[i].legend()
    axes[i].set_xlabel('T')
    axes[i].set_ylabel('Probability')

plt.tight_layout()

#2

Pkc = [[] for _ in range(len(densidades_cells))]

potencia = [p*2 for p in alpha]

for T in T_values:
    pkc_values = []
    for k in range(len(densidades_cells)):
        lambda_k = densidades_cells[k]
        integral_result, _ = quad(integrando, 0, np.inf, args=(alpha, densidades_cells, potencia, k, T))
        pkc_value = 2 * np.pi * lambda_k * integral_result
        pkc_values.append(pkc_value)

    for i in range(len(pkc_values)):
        Pkc[i].append(pkc_values[i])

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']
for i in range(len(Pkc)):
    axes[i].plot(T_values, Pkc[i], marker='^', linestyle='-', color=colors[i], label=f'Times 2 For All: Layer {i + 1}', markersize=5)
    axes[i].legend()
    axes[i].set_xlabel('T')
    axes[i].set_ylabel('Probability')

plt.tight_layout()


#3

Pkc = [[] for _ in range(len(densidades_cells))]

potencia = [40*4, 20, 200, 0.0355, 0.251, 0.501, 0.221, 31.6]

for T in T_values:
    pkc_values = []
    for k in range(len(densidades_cells)):
        lambda_k = densidades_cells[k]
        integral_result, _ = quad(integrando, 0, np.inf, args=(alpha, densidades_cells, potencia, k, T))
        pkc_value = 2 * np.pi * lambda_k * integral_result
        pkc_values.append(pkc_value)

    for i in range(len(pkc_values)):
        Pkc[i].append(pkc_values[i])

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']
for i in range(len(Pkc)):
    axes[i].plot(T_values, Pkc[i], marker='s', linestyle='-', color=colors[i], label=f'Times 4 For Layer 1: Layer {i + 1}', markersize=5)
    axes[i].legend()
    axes[i].set_xlabel('T')
    axes[i].set_ylabel('Probability')

plt.tight_layout()

plt.show()


'''
#2

Pkc = [[] for _ in range(len(densidades_cells))]

alpha = [a*2 for a in alpha]

for T in T_values:
    pkc_values = []
    for k in range(len(densidades_cells)):
        lambda_k = densidades_cells[k]
        integral_result, _ = quad(integrando, 0, np.inf, args=(alpha, densidades_cells, potencia, k, T))
        pkc_value = 2 * np.pi * lambda_k * integral_result
        pkc_values.append(pkc_value)

    for i in range(len(pkc_values)):
        Pkc[i].append(pkc_values[i])

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']
for i in range(len(Pkc)):
    axes[i].plot(T_values, Pkc[i], marker='^', linestyle='-', color=colors[i], label=f'Times 2 For All: Layer {i + 1}')
    axes[i].legend()
    axes[i].set_xlabel('T')
    axes[i].set_ylabel('Probability')

plt.tight_layout()

#3

Pkc = [[] for _ in range(len(densidades_cells))]

alpha = [6, 3, 4, 2.5, 2.5, 2.5, 2.5, 3]

for T in T_values:
    pkc_values = []
    for k in range(len(densidades_cells)):
        lambda_k = densidades_cells[k]
        integral_result, _ = quad(integrando, 0, np.inf, args=(alpha, densidades_cells, potencia, k, T))
        pkc_value = 2 * np.pi * lambda_k * integral_result
        pkc_values.append(pkc_value)

    for i in range(len(pkc_values)):
        Pkc[i].append(pkc_values[i])

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']
for i in range(len(Pkc)):
    axes[i].plot(T_values, Pkc[i], marker='s', linestyle='-', color=colors[i], label=f'Times 2 for Layer 1: Layer {i + 1}')
    axes[i].legend()
    axes[i].set_xlabel('T')
    axes[i].set_ylabel('Probability')

plt.tight_layout()
plt.show()
'''

############


def Z_t(t, alpha_j):
    a = 1
    b = 1 - (2 / alpha_j)
    c = 2 - (2 / alpha_j)
    z = -(np.exp(t) - 1)
    result = (2 * (np.exp(t) - 1)) / (alpha_j - 2) * hyp2f1(a, b, c, z)
    return result


def integranda(t, alpha):
    Z_value = Z_t(t, alpha)
    return 1 / (1 + Z_value)

intervals_t = [(0, 10), (10, 100), (100, np.inf)]


Alphas = np.sort(np.unique(np.random.uniform(2.5, 5.5, 100))).tolist()
Results = []
for Alpha in Alphas:
    result = 0
    for it in intervals_t:
        partial_result, _ = quad(integranda, it[0], it[1], args=(Alpha))
        result += partial_result
    Results.append(round(result/np.log(2), 2))
plt.plot(Alphas, Results, marker='o', markersize=3)
plt.xlabel('Path Loss Exponent')
plt.ylabel('Average Ergodic Rate')
plt.show()

T_values = np.sort(np.unique(np.random.uniform(0, 20, 100))).tolist()


def Z(T, alpha_j):
    a = 1
    b = 1 - (2 / alpha_j)
    c = 2 - (2 / alpha_j)
    z = -T
    result = (2 * T) / (alpha_j - 2) * hyp2f1(a, b, c, z)
    return result


def Cj(lambda_j, Pj, Pk, alpha_j, Zj):
    Cj = lambda_j * ((Pj / Pk) ** (2 / alpha_j) + Zj)
    return Cj


def integrando(r, alphas, lambdas, potencias, k, T):
    soma_Cj = 0
    for j in range(len(potencias)):
        Cj_value = Cj(lambdas[j], potencias[j], potencias[k], alphas[j], Z(T, alphas[j]))
        soma_Cj += Cj_value * r ** (2 / (alphas[j] / alphas[k]))
    return r * np.exp(-np.pi * soma_Cj)


def calcular_pkc(densidades_cells, potencia, alpha, T_values):
    Pkc = [[] for _ in range(len(densidades_cells))]
    for T in T_values:
        pkc_values = []
        for k in range(len(densidades_cells)):
            lambda_k = densidades_cells[k]
            integral_result, _ = quad(integrando, 0, np.inf, args=(alpha, densidades_cells, potencia, k, T))
            pkc_value = 2 * np.pi * lambda_k * integral_result
            pkc_values.append(pkc_value)
        for i in range(len(pkc_values)):
            Pkc[i].append(pkc_values[i])
    return Pkc

DELTA = [1/4, 1/2, 1, 2, 4]
def plot_results(variacao, labels, parametro):
    fig, axes = plt.subplots(2, 4, figsize=(14, 16))
    axes = axes.flatten()
    colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']

    for var, label in zip(variacao, labels):
        Pkc = calcular_pkc(var, potencia, alpha, T_values)
        for i in range(len(Pkc)):
            axes[i].plot(T_values, Pkc[i], marker='o', linestyle='-', label=f'{label} Layer {i + 1}')
            axes[i].legend()
            axes[i].set_xlabel('T')
            axes[i].set_ylabel('Probability')

    plt.suptitle(f'Análise de Sensibilidade - {parametro}')
    plt.tight_layout()
    plt.show()
densidades_variadas = [[d * delta for d in densidades_cells] for delta in DELTA]
labels_densidades = ["1/4x Densidade", "1/2x Densidade", "1x Densidade", "2x Densidade", "4x Densidade"]

# Variação das potências
potencias_variadas = [[p * delta for p in potencia] for delta in DELTA]
labels_potencias = ["Original", "1.5x Potência", "0.5x Potência", "2x Potência", "4x Potência"]

# Variação dos expoentes de perda
alphas_variadas = [[a * delta for a in alpha] for delta in DELTA]
labels_alphas = ["Original", "1.5x Alpha", "0.5x Alpha", "2x Alpha", "4x Alpha"]

# Realizar análises de sensibilidade
#plot_results(densidades_variadas, labels_densidades, "Densidade de Células")
plot_results(potencias_variadas, labels_potencias, "Potência")
#plot_results(alphas_variadas, labels_alphas, "Expoente de Perda")


