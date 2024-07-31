import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, mapping, shape
import random
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.integrate import quad, dblquad
from scipy.special import hyp2f1
import warnings
import pandas as pd
import folium


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
densidade de bs: 0.01
densidade de usuários: 0.03
expoente de perda de caminho: 4 
'''

np.random.seed(4)

# Region
x_min, x_max = 0, 100
y_min, y_max = 0, 100
region_area = (x_max - x_min) * (y_max - y_min)

# BS density
density_b = 0.01
expected_num_points = int(region_area * density_b)

# Generating Point from a PPP
num_points = np.random.poisson(expected_num_points)
points = np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(num_points, 2))

# Plot users
density_u = 0.03
expected_num_pointss = int(region_area * density_u)
num_pointss = np.random.poisson(expected_num_pointss)
pointss = np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(num_pointss, 2))

# Creating Voronoi Cells
vor = Voronoi(points)
voronoi_plot_2d(vor)

for i in range(len(pointss)):
    xs, ys = pointss[i]
    plt.plot(xs, ys, marker='o', markersize=5, label='Users', color='black')

for region_index in vor.regions:
    if region_index and -1 not in region_index:
        region_vertices = vor.vertices[region_index]
        line_vert = LineString(region_vertices)
        centroid = line_vert.centroid
        x, y = centroid.xy
        plt.plot(*line_vert.xy, marker='o', markersize=5, label='Points', color='blue')
        plt.plot(x, y, marker='o', markersize=5, label='Centroid', color='red')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Cells Diagram With Homogeneous PPP')

# Selecting random user
i = random.randint(1, num_pointss)
user = pointss[i]
plt.plot(user[0], user[1], marker='^', label = 'Selected MU', markersize=7, color='green')


# Parameters and Minimum Distance
alpha = 4

R = []
for j in range(len(points)):
    bs = points[j]
    dist = np.linalg.norm(np.sqrt((user[0] - bs[0])**2 + (user[1] - bs[1])**2))
    R.append(dist.round(2))
r = min(R)
R.remove(r)

server_bs = []
for j in range(len(points)):
    bs = points[j]
    if np.linalg.norm(np.sqrt((user - bs)**2)).round(2) == r:
        server_bs.append(bs)
        plt.plot(bs[0], bs[1], marker='^', color='blue')


# Links bs and user:
line = LineString([user, server_bs[0]])
xl, yl = line.xy
plt.plot(xl, yl, color = 'yellow')
plt.show()

# Coverage Probability:
alpha = 4
T_values = np.sort(np.unique(np.random.uniform(0, 20, 100))).tolist()
PC = []
for T in T_values:
    def integrand(x):
        return 1/(1 + x**(alpha/2))
    result, error = quad(integrand, T**(-2/alpha), np.inf)

    p_active = 1 - (1 + (density_u)/(density_b*3.5))**(-3.5)
    rho = T**(2/alpha)*result
    pc = (density_b)/(density_u)*p_active/(1 + p_active*rho)
    PC.append(pc)
plt.plot(T_values, PC, marker='o', color='red', markersize=3, label=f'alpha = {alpha}')
print(PC, len(PC))
plt.xlabel('SINR')
plt.ylabel('Probability of Coverage')
plt.title('Probability of Coverage x SINR')
plt.grid(True)
plt.legend()
plt.show()

'''
Saturated Network:
PC = []
T = np.sort(np.unique(np.random.uniform(0, 20, 100))).tolist()
for t in T:
    rho = t ** (2 / alpha) * result
    pc = 1 / (1 + rho)
    PC.append(pc)

plt.plot(T, PC, marker='o', markersize=3)
plt.xlabel('SINR')
plt.ylabel('Probability of Coverage')
plt.title('Probability of Coverage x SINR')
plt.grid(True)
plt.show()
'''

#Taxa de dados
def integral(t, alpha):
    integrand = lambda x: 1 / (1 + x**(alpha/2))
    lower_limit = (np.exp(t) - 1)**(-2/alpha)
    inner_result, _ = quad(integrand, lower_limit, np.inf)
    return 1 / (1 + p_active*(np.exp(t) - 1)**(2/alpha) * inner_result)

def tau(lambd, alpha):
    integral_func = lambda t: integral(t, alpha)
    tau, _ = quad(integral_func, 0, np.inf)
    return tau

print('='*100)

print(f'A taxa ergódica média de dados (SE) é de aproximadamente {round(tau(density_b, alpha)/np.log(2), 2)} bps/Hz')

print('='*100)


'''Loading data base'''

df = pd.read_csv('florianopolis_rbs.csv', encoding='latin1', low_memory=False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

colunas = ['SiglaUf', 'CodMunicipio', 'Tecnologia', 'FreqTxMHz',
           'FreqRxMHz', 'AlturaAntena', 'PotenciaTransmissorWatts',
           'Latitude', 'Longitude', 'NomeEntidade']
df = df.loc[:, colunas]

print('Área de Florianópolis: 675.409 Km^2')
area = 675.409

'''
df_coordinates = df.loc[:, ['Latitude', 'Longitude']]
df_coordinates = df_coordinates.drop_duplicates()
map_center = [df_coordinates['Latitude'].mean(), df_coordinates['Longitude'].mean()]
map = folium.Map(location=map_center, zoom_start=2)


for idx, row in df_coordinates.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']],
                  popup=f'Latitude: {row["Latitude"]}, Longitude: {row["Longitude"]}'
                 ).add_to(map)

map.show_in_browser()'''


'''Selecting data from a single operator'''

timdf = df.loc[df['NomeEntidade'] == 'TIM S A']

'''colunas = ['Tecnologia', 'PotenciaTransmissorWatts', 'FreqTxMHz']
timdf = timdf.loc[:, colunas]
#print(timdf.head(20))
#print(timdf['PotenciaTransmissorWatts'].value_counts())

grouped = timdf.groupby(['Tecnologia', 'FreqTxMHz'])['PotenciaTransmissorWatts'].describe()
print(grouped)'''


densidades_cells = []
for quantidade in timdf['PotenciaTransmissorWatts'].value_counts():
    lambda_cell = quantidade/area
    densidades_cells.append(lambda_cell)

potencia = [40, 20, 200, 0.0355, 0.251, 0.501, 0.221, 31.6]
quantidade = [1795, 408, 208, 192, 20, 4, 4, 3]
alpha = [3, 3, 4, 2.5, 2.5, 2.5, 2.5, 3]

dados = pd.DataFrame({'Potência (W)': potencia, 'Quantidade': quantidade,
                      'Densidade de células da camada': densidades_cells,
                      'Expoente de Perda': alpha})
dados.set_index('Potência (W)', inplace=True)

'''print(dados)

df_coordinates = timdf.loc[:, ['Latitude', 'Longitude']]
df_coordinates = df_coordinates.drop_duplicates()
map_center = [df_coordinates['Latitude'].mean(), df_coordinates['Longitude'].mean()]
timmap = folium.Map(location=map_center, zoom_start=2)

for idx, row in df_coordinates.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']],
                  popup=f'Latitude: {row["Latitude"]}, Longitude: {row["Longitude"]}'
                 ).add_to(timmap)

timmap.show_in_browser()'''

# Users density
lambda_u = 100000/area

# Region
x_min, x_max = 0, 16
y_min, y_max = 0, 42
region_area = (x_max - x_min) * (y_max - y_min)#Approx the area by 16*42 Km^2


expected_num_cell_points = [int(region_area * density) for density in densidades_cells]
print("Número esperado de células por tipo de célula (considerando a área aproximada):", expected_num_cell_points)
num_points_per_cell = [np.random.poisson(lam) for lam in expected_num_cell_points]
print("Número de células por tipo de célula calculado pelo processo de Poisson:", num_points_per_cell)

points = [np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(n_points, 2))
          for n_points in num_points_per_cell]

colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']

# Creating the Voronoi Diagram
all_points = np.vstack(points)
all_colors = np.concatenate([[colors[i]] * len(points[i]) for i in range(len(points))])
vor = Voronoi(all_points)
fig, ax = plt.subplots(figsize=(10, 10))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=2)
for i, point_set in enumerate(points):
    plt.scatter(point_set[:, 0], point_set[:, 1], color=colors[i])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Voronoi Diagram, Classification of the Layer Based on Color.')

# Plot Users
expected_num_pointss = int(region_area * lambda_u)
num_pointss = np.random.poisson(expected_num_pointss)
pointss = np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(num_pointss, 2))
'''for i in range(len(pointss)):
    xs, ys = pointss[i]
    plt.plot(xs, ys, marker='o', markersize=5, color='black')'''


# Random User
i = random.randint(0, num_pointss-1)
user = pointss[i]
plt.plot(user[0], user[1], marker='o', label = 'Selected MU', markersize=10, color='white')

distancia_atendimento = []
Server_BS = []
distancia_interferente = []
for i in range(len(points)):
    R = []
    for j in range(len(points[i])):
        bs = points[i][j]
        dist = np.linalg.norm(np.sqrt((user[0] - bs[0])**2 + (user[1] - bs[1])**2))
        R.append(dist.round(2))
    r = min(R)
    distancia_atendimento.append(r)
    R.remove(r)
    distancia_interferente.append(R)

    server_bs = []
    for j in range(len(points[i])):
        bs = points[i][j]
        if np.linalg.norm(np.sqrt((user - bs)**2)).round(2) == r:
            server_bs.append(bs)
            plt.plot(bs[0], bs[1], marker='^', color=colors[i], markersize=10, label=f'Server BS layer{i+1}')
            plt.legend()

    Server_BS.append(server_bs[0])

for i in range(len(Server_BS)):
    line = LineString([user, Server_BS[i]])
    xl, yl = line.xy
    plt.plot(xl, yl, color = 'white')

plt.show()


A = []
N = []
for k in range(len(potencia)):
    def integrand(r):
        sum_exponential = 0
        for j in range(len(potencia)):
            term = densidades_cells[j] * ((potencia[j] / potencia[k]) ** (2 / alpha[j])) * r**(2 * alpha[k] / alpha[j])
            sum_exponential += term
        return r * np.exp(-np.pi * sum_exponential)

    result, error = quad(integrand, 0, np.inf)
    Ak = 2 * np.pi * densidades_cells[k] * result
    n = 2 * np.pi * lambda_u * result
    A.append(Ak)
    N.append(n)

for k in range(len(A)):
    print('The probability of a random user being associated with level {} is: {}%'.format(k+1, A[k]*100))
    print('The average number of users associated with a BS in layer {} at any given time is: {}'.format(k+1, N[k]))

# Probability of coverage
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
    axes[i].plot(T_values, Pkc[i], marker='o', linestyle='-', color=colors[i], label=f'Layer {i + 1}')
    axes[i].legend()
    axes[i].set_xlabel('T')
    axes[i].set_ylabel('Probability')

plt.tight_layout()
plt.show()


# SE:
def Z_t(t, alpha_j):
    a = 1
    b = 1 - (2 / alpha_j)
    c = 2 - (2 / alpha_j)
    z = -(np.exp(t) - 1)
    result = (2 * (np.exp(t) - 1)) / (alpha_j - 2) * hyp2f1(a, b, c, z)
    return result

def Cj(t, lambda_j, Pj, Pk, alpha_j):
    Zj = Z_t(t, alpha_j)
    Cj_value = lambda_j * ((Pj / Pk) ** (2 / alpha_j)) * (1 + Zj)  # B = 1
    return Cj_value

def integrando(x, t, alphas, lambdas, potencias, k):
    soma_Cj = 0
    for j in range(len(potencias)):
        Cj_value = Cj(t, lambdas[j], potencias[j], potencias[k], alphas[j])
        soma_Cj += (x ** (2 / (alphas[j] / alphas[k]))) * Cj_value
    return x * np.exp(-np.pi * soma_Cj)

def integral_dupla(alphas, lambdas, potencias, k):
    def integrand(x, t):
        return integrando(x, t, alphas, lambdas, potencias, k)

    result = 0
    intervals_x = [(0, 10), (10, 100), (100, np.inf)]
    intervals_t = [(0, 10), (10, 100), (100, np.inf)]

    for ix in intervals_x:
        for it in intervals_t:
            partial_result, _ = dblquad(integrand, ix[0], ix[1], lambda x: it[0], lambda x: it[1])
            result += partial_result

    return result

tau = 0
for k in range(len(densidades_cells)):
    lambda_k = densidades_cells[k]
    integral_result = integral_dupla(alpha, densidades_cells, potencia, k)
    tau += 2 * np.pi * lambda_k * integral_result

print('='*100)
print('Average network spectral efficiency, in bps/Hz, for the case with different path loss exponents:', round(tau/np.log(2), 2))

# Equal path loss:

def integranda(t, alpha):
    Z_value = Z_t(t, alpha)
    return 1 / (1 + Z_value)

Alpha = 4

intervals_t = [(0, 10), (10, 100), (100, np.inf)]

for it in intervals_t:
    partial_result, _ = quad(integranda, it[0], it[1], args=(Alpha))
    result += partial_result


print('Average network spectral efficiency, in bps/Hz, for the case with all path loss exponents equal to 4:', round(result/np.log(2), 2))
print('='*100)