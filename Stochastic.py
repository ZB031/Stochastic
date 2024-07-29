import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point, mapping, shape
import random
from scipy.integrate import quad
import warnings
from functools import partial
import math
from rtree import index
import pyproj
import pandas as pd
import geopandas as gpd
import folium
from scipy.special import hyp2f1
from scipy.integrate import dblquad

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
densidade de bs: 0.01
densidade de usuários: 0.03
expoente de perda de caminho: 4 
'''

np.random.seed(4)

# Definindo os parâmetros da região retangular
x_min, x_max = 0, 100
y_min, y_max = 0, 100
region_area = (x_max - x_min) * (y_max - y_min)

# Definindo a densidade de pontos (pontos por unidade de área)
density_b = 0.01
expected_num_points = int(region_area * density_b)

# Gerando os pontos de acordo com o processo de Poisson homogêneo
num_points = np.random.poisson(expected_num_points)
points = np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(num_points, 2))

# Plotando os usuários de dados, com a mesma seed, de forma que regiões mais densas de bs tenham mais usuários
density_u = 0.03
expected_num_pointss = int(region_area * density_u)
num_pointss = np.random.poisson(expected_num_pointss)
pointss = np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(num_pointss, 2))

# Criando as células de Voronoi
vor = Voronoi(points)
voronoi_plot_2d(vor)

for i in range(len(pointss)):
    xs, ys = pointss[i]
    plt.plot(xs, ys, marker='o', markersize=5, label='Usuários', color='black')

for region_index in vor.regions:
    if region_index and -1 not in region_index:
        region_vertices = vor.vertices[region_index]
        line_vert = LineString(region_vertices)
        centroid = line_vert.centroid
        x, y = centroid.xy
        plt.plot(*line_vert.xy, marker='o', markersize=5, label='Pontos', color='blue')
        plt.plot(x, y, marker='o', markersize=5, label='Centróide', color='red')


# Definindo os limites do gráfico
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Voronoi Cells Diagram With Homogeneous PPP')

# Selecionando um usuário aleatório:
i = random.randint(1, num_pointss)
user = pointss[i]
plt.plot(user[0], user[1], marker='^', label = 'Selected MU', markersize=7, color='green')


# Parâmetros:
alpha = 4

R = []
for j in range(len(points)):
    bs = points[j]
    dist = np.linalg.norm(np.sqrt((user[0] - bs[0])**2 + (user[1] - bs[1])**2))
    R.append(dist.round(2))
r = min(R)#distância de um usuário até a estação base mais próxima
R.remove(r)#distância das estações interferentes até o usuário considerado

server_bs = []
for j in range(len(points)):
    bs = points[j]
    if np.linalg.norm(np.sqrt((user - bs)**2)).round(2) == r:
        server_bs.append(bs)
        plt.plot(bs[0], bs[1], marker='^', color='blue')

T = 1
lambd = density_b
sigma = 0

# Liga bs e user:
line = LineString([user, server_bs[0]])
xl, yl = line.xy
plt.plot(xl, yl, color = 'yellow')
plt.show()

#Probabilidade de cobertura
def integrand(x):
    return 1/(1 + x**(alpha/2))
result, error = quad(integrand, T**(-2/alpha), np.inf)

p_active = 1 - (1 + (density_u)/(density_b*3.5))**(-3.5)

rho = T**(2/alpha)*result
pc = (density_b)/(density_u)*p_active/(1 + p_active*rho)

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

#Resultados:

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

print('='*100)

print(f'A taxa ergódica média de dados (SE) é de aproximadamente {round(tau(lambd, alpha)/np.log(2), 2)} bps/Hz')

print('='*100)


'''Análise Mais Detalhada: Começaremos com as informações relativas às células, fornecidas pela ANATEL.'''

# Estudando Florianópolis
df = pd.read_csv('florianopolis_rbs.csv', encoding='latin1', low_memory=False)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

colunas = ['SiglaUf', 'CodMunicipio', 'Tecnologia', 'FreqTxMHz',
           'FreqRxMHz', 'AlturaAntena', 'PotenciaTransmissorWatts',
           'Latitude', 'Longitude', 'NomeEntidade']
df = df.loc[:, colunas]

print('Área de Florianópolis: 675.409 Km^2')
area = 675.409

#Mostrando as células de transmissão no mapa:
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


'''Agora vou pegar dados de uma única operadora, a saber TIM.'''

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

#Caso considerado: sem viés (o usuário será associado à estação base com sinal mais forte e mais próxima)
# e sem ruído térmico.

lambda_u = 100000/area#Densidade de usuários, dado de teste

# Definindo os parâmetros da região retangular
x_min, x_max = 0, 16
y_min, y_max = 0, 42
region_area = (x_max - x_min) * (y_max - y_min)#Aproximei a area de Florianópolis por 16*42 Km^2


# Número esperado de pontos por célula
expected_num_cell_points = [int(region_area * density) for density in densidades_cells]
print("Número esperado de células por tipo de célula (considerando a área aproximada):", expected_num_cell_points)
num_points_per_cell = [np.random.poisson(lam) for lam in expected_num_cell_points]
print("Número de células por tipo de célula calculado pelo processo de Poisson:", num_points_per_cell)

# Plotando os pontos para cada célula
points = [np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(n_points, 2))
          for n_points in num_points_per_cell]

# Cores para cada célula
colors = ['b', 'green', 'red', 'orange', 'yellow', 'purple', 'pink', 'gray']

# Criando um diagrama de Voronoi para as células
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

#Plotando usuários
expected_num_pointss = int(region_area * lambda_u)
num_pointss = np.random.poisson(expected_num_pointss)
pointss = np.random.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(num_pointss, 2))
'''for i in range(len(pointss)):
    xs, ys = pointss[i]
    plt.plot(xs, ys, marker='o', markersize=5, color='black')'''


# Selecionando um usuário aleatório:
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
    r = min(R)#distância de um usuário até a estação base mais próxima
    distancia_atendimento.append(r)
    R.remove(r)#distância das estações interferentes até o usuário considerado
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
    print('A probabilidade de um usuário aleatório estar associado com o nível {} é: {}%'.format(k+1, A[k]*100))
    print('O número médio de usuários associados a uma BS na camada {}, em um dado instante, é: {}'.format(k+1, N[k]))

# Probabilidade de cobertura: Seja T o SINR
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


# Calculando a eficiência espectral:
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
print('Eficiência espectral média da rede, em bps/Hz, para o caso com diferentes expoentes de perda de caminho:', round(tau/np.log(2), 2))

#Caso com todos os expoentes de perda de caminho iguais:

def integranda(t, alpha):
    Z_value = Z_t(t, alpha)
    return 1 / (1 + Z_value)

Alpha = 4

intervals_t = [(0, 10), (10, 100), (100, np.inf)]

for it in intervals_t:
    partial_result, _ = quad(integranda, it[0], it[1], args=(Alpha))
    result += partial_result


print('Eficiência espectral média da rede, em bps/Hz, para o caso com todos os expoentes de perda de caminho iguais a 4:', round(result/np.log(2), 2))
print('='*100)