# libs
import numpy as np
from modules import *

# pyod
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.cd import CD
from pyod.models.auto_encoder_torch import InnerAutoencoder
from pyod.models.copod import COPOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.kpca import KPCA
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.loci import LOCI
from pyod.models.lunar import LUNAR
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.rgraph import RGraph
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.suod import SUOD
from pyod.models.vae import VAE
from pyod.models.xgbod import XGBOD

# run
X = hyperbol2d(n = 1000).X
Y = rectangle2d(ll = [-2, -2], ur = [2, 2], n = 10000).X

# choose anomaly detection
# not good
#ad = ALAD(contamination=0.0001) # not good, complex algorithm, need to investigate further
#ad = AnoGAN(contamination = 0.0001, verbose = 1) # very slow, ...
#ad = AutoEncoder(contamination = 0.0001, verbose = 1, hidden_neurons = [2, 10, 2], epochs=1000) # not very good
#ad = CBLOF(contamination = 0.0001, n_clusters = 8) # doesn't work, needs more config
#ad = CD(contamination = 0.1) # seems not to work in unsupervised mode
#ad = COF(contamination = 0.1, n_neighbors = 5) # lousy
#ad = COPOD(contamination=0.4) # not great
#ad = ECOD(contamination = 0.4999) # mediocre
#ad = InnerAutoencoder(n_features = 2) # has different IF
#ad = HBOS(contamination = 0.0001, n_bins = 200)
#ad = IForest(contamination = 0.0001)
#ad =  KDE(contamination = 0.0001)
#ad = LMDD(contamination = 0.0001, dis_measure = 'var')
#ad = LODA(contamination = 0.0001)
#ad = LOCI(contamination = 0.0001)
#ad = LUNAR(model_type = 'SCORE', n_neighbours = 50)
#ad = MAD()
#ad = MCD(contamination = 0.0001)
#ad = MO_GAAL(contamination = 0.0001)
#ad = PCA(contamination = 0.0001)
#ad = RGraph(contamination = 0.0001)
#ad = Sampling(contamination = 0.0001)
#ad = SOD(contamination = 0.0001)
#ad = SO_GAAL(contamination = 0.0001)
#ad = SOS(contamination = 0.0001)
#ad = VAE(contamination = 0.0001, encoder_neurons = [2, 50])
#ad = XGBOD(contamination = 0.0001)

# good
#ad = ABOD(contamination = 0.0001) 
#ad = DeepSVDD(contamination = 0.0001, epochs=200, hidden_neurons = [20, 50, 20], l2_regularizer = 0.0) 
#ad = FeatureBagging(contamination = 0.0001, n_estimators = 10)
#ad = GMM(contamination = 0.0001, n_components = 10)
#ad = INNE(contamination = 0.0001)
#ad = KNN(contamination = 0.0001)
#ad = KPCA(contamination = 0.0001, gamma = 100)
#ad = LOF(contamination = 0.0001)
#ad = LSCP([ABOD(), FeatureBagging(), KNN(), LOF()], contamination = 0.0001)
#ad = OCSVM(contamination = 0.0001, gamma = 100)
#ad = SUOD([ABOD(), FeatureBagging(), KNN(), LOF()], contamination = 0.0001)


# use anomaly detector
ad.fit(X)
idx = ad.predict(Y) == 1
Y = Y[idx]

# plot
plot2d(L = [[X, 'scatter', 'blue', 5], [Y, 'scatter', 'red', 5]])






