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

dados = pd.read_csv('c:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values
dados = pd.DataFrame(X)
dados['classe'] = Y

DSL = DEC(6, np.size(X, axis=1), 6)
DSL.inicialicazao(X)

p = DSL.DSL.predict(X)
q = DSL.p_mat(p)
DSL.DSL.fit(X, q, epochs=100)

#GRÁFICO DO AGRUPAMENTO
Xt = tsne.fit_transform(DSL.encoder.predict(X))
preditas = DSL.DSL.predict_classes(X)
cores = ['#000000','#0000FF','#7FFFD4','#008000','#CD853F','#8B008B','#FF0000','#FFA500','#FFFF00','#FF1493']

for i, x in enumerate(Xt):
    plt.scatter(x[0], x[1], c = cores[preditas[i]], s = 5)
plt.legend()
plt.show()