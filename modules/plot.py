# libs
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# fcts
def plot2d(size = (10, 10), L = None):
    assert(L is not None) 

    figure(figsize = size)
    for l in L:
        X = l[0]
        mode = l[1]
        color = l[2]
        if mode == 'scatter':
            s = l[3]
            plt.scatter(X[:, 0], X[:, 1], s = s, color = color)

    plt.show()





