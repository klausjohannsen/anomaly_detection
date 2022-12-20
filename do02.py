# libs
import numpy as np
from time import time
from modules import *

# config
PLOT = True

# pyod
from pyod.models.abod import ABOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.inne import INNE
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.ocsvm import OCSVM
from pyod.models.suod import SUOD

# run
X = hyperbol2d(n = 1000).X
Y = rectangle2d(ll = [-2, -2], ur = [2, 2], n = 10000).X

# choose anomaly detection

# linear
#ad = ABOD(contamination = 0.0001) 
#ad = GMM(contamination = 0.0001, n_components = 10)
#ad = KNN(contamination = 0.0001)

# nlogn
#ad = FeatureBagging(contamination = 0.0001, n_estimators = 10)
ad = LOF(contamination = 0.0001)
#ad = INNE(contamination = 0.0001)
#ad = LSCP([ABOD(), FeatureBagging(), KNN(), LOF()], contamination = 0.0001)
#ad = SUOD([ABOD(), FeatureBagging(), KNN(), LOF()], contamination = 0.0001)

# slower
#ad = KPCA(contamination = 0.0001, gamma = 100)
#ad = DeepSVDD(contamination = 0.0001, epochs=200, hidden_neurons = [20, 50, 20], l2_regularizer = 0.0) 
#ad = OCSVM(contamination = 0.0001, gamma = 100)

# use anomaly detector
for k, e in enumerate([0, 1, 2, 3]):
    # data
    n = 1000 * (10 ** e)
    X = hyperbol2d(n = n).X
    Y = rectangle2d(ll = [-2, -2], ur = [2, 2], n = n).X

    # offset
    if k == 0:
        ad.fit(X)

    # ad
    t0 = time()
    ad.fit(X)
    idx = ad.predict(Y) == 1
    print(f'n = {n}, time = {time() - t0} sec')

    # plot
    Y = Y[idx]
    if PLOT:
        plot2d(L = [[X, 'scatter', 'blue', 5], [Y, 'scatter', 'red', 5]])






