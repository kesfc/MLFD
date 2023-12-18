import numpy as np
import matplotlib.pyplot as plt
import math


def Euclidean(x1, x2):
    return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)
def k_nn(x, y, k, trans=False):
    data = x
    if(trans):
        new_x = []
        for i in range(len(x)):
            x1 = (x[i][0]**2 + x[i][1]**2)**0.5
            x2 = math.atan(x[i][1]/(x[i][0]+.0000000001))
            new_x.append([x1, x2])
        data = new_x
    data = np.array(data)
    y = np.array(y)
    X = np.linspace(-3, 3)
    Y = np.linspace(-3, 3)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    Xt = X
    Yt = Y
    if(trans):
        X1 = []
        X2 = []
        for i in range(X.shape[0]):
            t1 = []
            t2 = []
            for j in range(X.shape[1]):
                t1.append((X[i][j]**2 + Y[i][j]**2)**0.5)
                t2.append(math.atan((Y[i][j]+0.0000001)/X[i][j]))
            X1.append(t1)
            X2.append(t2)
        X = np.array(X1)
        Y = np.array(X2)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dist = []
            for l in range(len(data)):
                dist.append([Euclidean([X[i][j], Y[i][j]], data[l]), l])
            dist.sort()
            count = 0
            for l in range(k):
                count += y[dist[l][1]]
            if(count > 0):
                Z[i][j] = 1
            else:
                Z[i][j] = -1
    plt.contourf(Xt, Yt, Z)
    n1 = []
    n2 = []
    for i in range(len(y)):
        if(y[i] == 1):
            n1.append(x[i])
        else:
            n2.append(x[i])
    n1 = np.array(n1)
    n2 = np.array(n2)
    plt.scatter(n1[:, 0], n1[:, 1], c="r", label='1')
    plt.scatter(n2[:, 0], n2[:, 1], c="g", label='-1')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    x = [[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]]
    y = [-1, -1, -1, -1, 1, 1, 1]
    k_nn(x, y, 1)
    k_nn(x, y, 3)
    k_nn(x, y, 1, True)
    k_nn(x, y, 3, True)