import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D

def integrand(x, alpha):
    return 1 / (1 + x ** (alpha / 2))


def probability_of_coverage(alpha, T):
    result, _ = quad(integrand, T ** (-2 / alpha), np.inf, args=(alpha,))
    rho = T ** (2 / alpha) * result
    return rho


def integral(t, alpha, p_active):
    integrand = lambda x: 1 / (1 + x ** (alpha / 2))
    lower_limit = (np.exp(t) - 1) ** (-2 / alpha)
    inner_result, _ = quad(integrand, lower_limit, np.inf)
    return 1 / (1 + p_active * (np.exp(t) - 1) ** (2 / alpha) * inner_result)


def tau(lambd, alpha, p_active):
    integral_func = lambda t: integral(t, alpha, p_active)
    tau, _ = quad(integral_func, 0, np.inf)
    return tau

density_u = np.sort(np.unique(np.random.uniform(0.0025, 0.0125, 10))).tolist()
density_b = np.sort(np.unique(np.random.uniform(0.01, 0.05, 10))).tolist()

alphas = [3, 4, 5]


fig = plt.figure(figsize=(18, 6))

for idx, alpha in enumerate(alphas):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    density_b_mesh, density_u_mesh = np.meshgrid(density_b, density_u)
    tau_values = np.zeros_like(density_b_mesh)

    for i in range(len(density_u)):
        for j in range(len(density_b)):
            lambd = density_b[j]
            p_active = 1 - (1 + (density_u[i]) / (density_b[j] * 3.5)) ** (-3.5)
            tau_values[i, j] = tau(lambd, alpha, p_active) / np.log(2)

    ax.plot_surface(density_b_mesh, density_u_mesh, tau_values, cmap='viridis')
    ax.set_xlabel('Density of Base Stations (density_b)')
    ax.set_ylabel('Density of Users (density_u)')
    ax.set_zlabel('Tau (bps/Hz)')
    ax.set_title(f'Alpha = {alpha}')

plt.tight_layout()
plt.show()


density_u = 0.03
density_b = 0.01
alphas = [3, 4, 5]  # Lista de expoentes de perda de caminho
T_values = np.sort(np.unique(np.random.uniform(0, 20, 100))).tolist()  # Lista de valores de SINR
colors = ['red', 'green', 'blue']
for i, alpha in enumerate(alphas):
    PC = []
    for T in T_values:


        def integrand(x):
            return 1/(1 + x**(alpha/2))
        result, error = quad(integrand, T**(-2/alpha), np.inf)

        p_active = 1 - (1 + (density_u)/(density_b*3.5))**(-3.5)

        rho = T**(2/alpha)*result
        pc = (density_b)/(density_u)*p_active/(1 + p_active*rho)
        PC.append(pc)
    plt.plot(T_values, PC, marker='o', markersize=3, color=colors[i], label=f'alpha = {alpha}')
    print(PC, len(PC))
plt.xlabel('SINR')
plt.ylabel('Probability of Coverage')
plt.title('Probability of Coverage x SINR')
plt.grid(True)
plt.legend()
plt.show()



density_u_values = np.sort(np.unique(np.random.uniform(0.0025, 0.0125, 10)))
density_b_values = np.sort(np.unique(np.random.uniform(0.01, 0.05, 10)))
alphas = [3, 4, 5]  # Lista de expoentes de perda de caminho


def calculate_pc(density_u, density_b, alpha, T):
    def integrand(x):
        return 1 / (1 + x ** (alpha / 2))

    result, error = quad(integrand, T ** (-2 / alpha), np.inf)

    p_active = 1 - (1 + (density_u) / (density_b * 3.5)) ** (-3.5)
    rho = T ** (2 / alpha) * result
    pc = (density_b) / (density_u) * p_active / (1 + p_active * rho)

    return pc


fig = plt.figure(figsize=(18, 6))
T = 1

for i, alpha in enumerate(alphas):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    X, Y = np.meshgrid(density_u_values, density_b_values)
    Z = np.array([[calculate_pc(d_u, d_b, alpha, T) for d_u in density_u_values] for d_b in density_b_values])

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlabel('User Density ($\lambda_u$)')
    ax.set_ylabel('Base Station Density ($\lambda_b$)')
    ax.set_zlabel('Probability of Coverage')
    ax.set_title(f'Alpha = {alpha}')
    ax.grid(True)

plt.tight_layout()
plt.show()