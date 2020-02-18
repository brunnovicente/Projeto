import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score
import keras.backend as K
from scipy.stats import t
from scipy.stats import norm
from Clustering import DEC

sca = MinMaxScaler()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

dados = pd.read_csv('d:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y

L, U, y, yu = train_test_split(X,Y, train_size=0.05, test_size=0.95, stratify=Y)

""" INICIALIZAÇÃO DO MODELO """
DSL = DEC(6, np.size(U, axis=1), 6, 0.1)
PL, PU = DSL.inicialicazao(U, L, y)

#GRÁFICO DO AGRUPAMENTO
marcador = ['.', 's', 'p', 'x', 'D', '^']
Ut = tsne.fit_transform(DSL.encoder.predict(U))
preditas = DSL.DSL.predict_classes(U)
cores = ['#000000','#0000FF','#7FFFD4','#008000','#CD853F','#8B008B','#FF0000','#FFA500','#FFFF00','#FF1493']

for i, x in enumerate(Ut):
    plt.scatter(x[0], x[1], s = 5, marker=marcador[preditas[i]], color='black')
plt.legend()
plt.show()

""" Rotulação do Modelo """
L = pd.DataFrame(L)
L['classe']=y

Lt, Ut = DSL.divisao_grupos(PU, PL)









