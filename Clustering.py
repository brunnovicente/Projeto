import sys
import numpy as np
import pandas as pd
import keras.backend as K
from keras.initializers import RandomNormal
from keras.engine.topology import Layer, InputSpec
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
from keras.callbacks import LearningRateScheduler
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from numpy import linalg
from scipy.stats import entropy

class ClusteringLayer(Layer):
    '''
    Clustering layer which converts latent space Z of input layer
    into a probability vector for each cluster defined by its centre in
    Z-space. Use Kullback-Leibler divergence as loss, with a probability
    target distribution.
    # Arguments
        output_dim: int > 0. Should be same as number of clusters.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        alpha: parameter in Student's t-distribution. Default is 1.0.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, input_dim=None, weights=None,  alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha
        # kmeans cluster centre locations
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        vetor = K.expand_dims(x, 1) - self.W
        quadrado = K.square(vetor)
        somatorio = K.sum(quadrado, axis=2)
        raiz = raiz = K.sqrt(somatorio)**2
        
        q = 1.0/(1.0 + raiz /self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        #v = []
        #for w in self.initial_weights:
            #xi = K.get_value(x)
            #xi = K.cast(x, dtype='float32')
        #    norma = linalg.norm(x - w)
        #    den = 1 / np.sqrt((1 + norma**2))
        #    v.append(den)
        #den = np.sum(v)
        #q = v / den        
        
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class DEC:
    
    def __init__(self, z=10, entrada=10, k=10, t=0.1):
        self.k = k
        self.entrada = entrada
        self.z = z
        self.t = t
        
        input_img = Input((self.entrada,))
        #encoded = Dense(50, activation='relu')(input_img)
        #drop = Dropout(0.2)(encoded)
        encoded = Dense(10, activation='relu')(input_img)
        #drop = Dropout(0.2)(encoded)
        #encoded = Dense(100, activation='relu')(drop)
        
        Z = Dense(self.z, activation='relu')(encoded)
        
        decoded = Dense(10, activation='relu')(Z)
        #drop = Dropout(0.2)(decoded)
        #decoded = Dense(50, activation='relu')(drop)
        #drop = Dropout(0.2)(decoded)
        #decoded = Dense(250, activation='relu')(drop)
        decoded = Dense(self.entrada, activation='sigmoid')(decoded)
                        
        self.encoder = Model(input_img, Z)
        self.autoencoder = Model(input_img, decoded)
        #self.autoencoder.summary()
        self.autoencoder.compile(loss='mse', optimizer=SGD(lr=0.1, decay=0, momentum=0.9))

    def p_mat(self, q):
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T
    
    def inicialicazao(self, U, L, y):
        
        indiceL = np.arange(np.size(L, axis=0))
        indiceU = np.arange(np.size(L, axis=0), np.size(L, axis=0) + np.size(U, axis=0))
        self.rotulos = np.zeros(np.size(U, axis=0))-1
        
        self.autoencoder.fit(U, U, epochs=100)
        self.kmeans = KMeans(n_clusters=self.k, n_init=20)
        self.kmeans.fit(self.encoder.predict(U))
        self.y_pred = self.kmeans.predict(self.encoder.predict(U))
        self.cluster_centres = self.kmeans.cluster_centers_
        
        self.DSL = Sequential([self.encoder, ClusteringLayer(self.k, weights=self.cluster_centres, name='clustering')])
        self.DSL.compile(loss='kullback_leibler_divergence', optimizer='adadelta')
        self.DSL.fit(U, self.p_mat(self.DSL.predict(U)))
        
        PL = pd.DataFrame(self.DSL.predict(L), index=indiceL)
        PL['classe'] = y
        PL['grupo'] = self.DSL.predict_classes(L)
        PU = pd.DataFrame(self.DSL.predict(U), index=indiceU)
        PU['grupo'] = self.DSL.predict_classes(U)
        self.fi = np.size(L, axis=0)
        
        return PL, PU
        
    def divisao_grupos(self, U, L):
        y = L['classe'].values
        gl = L['grupo'].values
        indiceL = L.index.values
        L = L.drop(['grupo'], axis=1)
        
        """ DIVISÃO DOS GRUPOS """
        indice = U.index.values

        for i in np.arange(self.k):
            Ut = U[U['grupo'] == i]
            Ut = Ut.drop(['grupo'], axis=1).values
            
            for a, x in enumerate(Ut):
                r = self.rotular_amostras(x, L.drop(['classe'], axis=1).values, y, self.k, self.t)
                self.rotulos[indice[a]-self.fi]  = r

        """ Remoção dos elementos rotulados """
        Ut = U.drop(['grupo'], axis=1)
        Ut['classe'] = self.rotulos
        novos = Ut[Ut['classe'] != -1]
        L = pd.concat([L, novos])
        Ut = Ut[Ut['classe']==-1]
        Ut = Ut.drop(['classe'], axis=1)
        return L, Ut
    
    def rotular_amostras(self, x, L, y, k, t):

        """ Calculando distância da Amostra para cada elemento de L """        
        dis = []
        for xr in L:
            #dis.append(distance.euclidean(x, xr))
            divergencia = entropy(x, xr)            #Calculando Divergência Kullback Leibler
            dis.append(divergencia)
        
        """ Descobrindo os k vizinhos rotulados menos divergentes """
        rot = pd.DataFrame(L)
        rot['y'] = y
        
        
        rot['dis'] = dis
        rot = rot.sort_values(by='dis')
        vizinhos = rot.iloc[0:k,:]
        vizinhos = vizinhos[vizinhos['dis']<=t]        
        
        """ Caso não existem vizinhos rotulados suficientes """
        if np.size(vizinhos, axis=1) < k:
            return -1
        
        """ Calculando as Classes """
        classes = np.unique(y)
        P = []
        for c in classes:
            q = (vizinhos['y'] == c).sum()
            p = q / k
            P.append(p)
        classe = self.calcular_classe(P)
        
        return classe
    
    def calcular_classe(self, probabilidades):
        c = -1
        for i, p in enumerate(probabilidades):
            pr = np.round(p)
            if pr == 1.:
                c = i
                break
        return c
    
    def ajuste_fino(self, PL, PU):
        
        pass 
        
        
        
        