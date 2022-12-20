# libs
import numpy as np
import numpy.linalg as la

# fcts
def hyperbol2d_fct(x):
    if x[0] * x[1] > 0.2:
        return(False)
    if x[0] * x[1] < -0.2:
        return(False)
    return(True)

# 2d data
class rectangle2d:
    def __init__(self, ll = None, ur = None, n = 1000):
        assert(ll is not None)
        assert(ur is not None)
        self.ll = np.array(ll).reshape(1, 2)
        self.ur = np.array(ur).reshape(1, 2)
        self.n = n
        self.X = self.ll + (self.ur - self.ll) * np.random.rand(n, 2)

    def __str__(self):
        s = ''
        s += '# rectangle2d\n'
        s += f'n = {self.n}\n'
        s += f'll = {self.ll}\n'
        s += f'ur = {self.ur}\n'
        return(s)

class hyperbol2d:
    def __init__(self, l = 1, n = 1000):
        self.ll = np.array([-l, -l])
        self.ur = np.array([l, l])
        self.n = n
        self.X = np.zeros((n, 2))
        n = 0
        while True:
            x = self.ll + (self.ur - self.ll) * np.random.rand(2)
            if hyperbol2d_fct(x):
                self.X[n] = x
                n += 1
            if n == self.n:
                break

    def __str__(self):
        s = ''
        s += '# hyperbol2d\n'
        s += f'n = {self.n}\n'
        s += f'll = {self.ll}\n'
        s += f'ur = {self.ur}\n'
        return(s)

# 3B anomaly detection
class data3B:
    def __init__(self, n = 30, encode = False):
        self.n = n
        self.encode = encode

        # load normal time series
        self.ts_normal_list = []
        for k in range(1133):
            self.ts_normal_list += [np.load('data/3B/normal-%0.4d.npy' % k)]

        # load non-normal time series
        self.ts_abnormal_list = []
        for k in range(111):
            self.ts_abnormal_list += [np.load('data/3B/non-normal-%0.4d.npy' % k)]

        # split
        self.X = self.split_tsl(self.ts_normal_list)
        self.Y = self.split_tsl(self.ts_abnormal_list)

        # encode
        if encode:
            p0 = np.ones(self.n)
            p0 = p0 / la.norm(p0)
            p1 = np.arange(self.n)
            p1 = p1 - np.dot(p1, p0) * p0
            p1 = p1 / la.norm(p1)
            p2 = p1 * p1
            p2 = p2 - np.dot(p2, p0) * p0 - np.dot(p2, p1) * p1
            p2 = p2 / la.norm(p2)
            self.PB = np.vstack((p0, p1, p2))
            self.X = [self.encoder(x) for x in self.X]
            self.Y = [self.encoder(x) for x in self.Y]

        # join
        self.Xj = np.vstack(self.X)
        self.Yj = np.vstack(self.Y)

    def encoder(self, x):
        assert(x.shape[1] == self.n + 1)
        m = x.shape[0]
        y = np.zeros((m, 4))
        for k in range(m):
            y[k, 0] = x[k, 0]
            y[k, 1:] = self.PB @ x[k, 1:]
        return(y)

    def split_tsl(self, tsl):
        chunks = []
        for ts in tsl:
            if ts.shape[0] >= self.n:
                c = np.zeros((ts.shape[0] - self.n + 1, self.n + 1))
                for k in range(ts.shape[0] - self.n + 1):
                    c[k, 0] = ts[k, 0]
                    c[k, 1:] = ts[k:(k + self.n), 1]
                chunks += [c]
        return(chunks)

    def __str__(self):
        s = ''
        s += '# 3B data\n'
        s += f'ts_normal_list: {len(self.ts_normal_list)} ({self.Xj.shape})\n'
        s += f'ts_abnormal_list: {len(self.ts_abnormal_list)} ({self.Yj.shape})\n'
        return(s)


