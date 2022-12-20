# libs
import numpy as np
from time import time
from modules import *

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
data = data3B(n = 30, encode = True)
print(data)

# choose anomaly detection
# linear
#ad = ABOD(contamination = 0.0001) 
ad = GMM(contamination = 0.00001, n_components = 20)
#ad = KNN(contamination = 0.0001)

# nlogn
#ad = FeatureBagging(contamination = 0.0001, n_estimators = 10)
#ad = LOF(contamination = 0.0001)
#ad = INNE(contamination = 0.0001)
#ad = LSCP([ABOD(), FeatureBagging(), KNN(), LOF()], contamination = 0.0001)
#ad = SUOD([ABOD(), FeatureBagging(), KNN(), LOF()], contamination = 0.0001)

# slower
#ad = KPCA(contamination = 0.0001, gamma = 100)
#ad = DeepSVDD(contamination = 0.0001, epochs=200, hidden_neurons = [20, 50, 20], l2_regularizer = 0.0) 
#ad = OCSVM(contamination = 0.0001, gamma = 100)

# use anomaly detector
ad.fit(data.Xj)

X_idx = []
for k, X in enumerate(data.X):
    print(k)
    z = ad.predict(X)
    X_idx += [np.sum(z) > 0]
X_idx = np.array(X_idx)

Y_idx = []
for k, Y in enumerate(data.Y):
    z = ad.predict(Y)
    Y_idx += [np.sum(z) > 0]
Y_idx = np.array(Y_idx)

tp = np.sum(Y_idx)
tn = np.sum(~X_idx)
fp = np.sum(X_idx)
fn = np.sum(~Y_idx)

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f'precision = {100 * precision:.1f}%')
print(f'recall    = {100 * recall:.1f}%')








