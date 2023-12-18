import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Data:
    def __init__(self, rad, sep, thk, count = 1000):
        self.rad = rad
        self.sep = sep
        self.thk = thk
        self.count = count
    def generate(self):
        self.data = []
        self.blue = []
        self.red = []
        for i in range(self.count):
            angle = random.uniform(0, 360)
            rad_lenth = random.uniform(self.rad, self.rad + self.thk)
            if(0<=angle<180):
                x = rad_lenth * np.cos(angle)
                y = (rad_lenth**2 - x**2)**0.5
                self.blue.append([x, y])
            else:
                x = rad_lenth * np.cos(angle-180)
                y = -(rad_lenth**2 - x**2)**0.5 - self.sep
                temp = self.thk/2
                x += self.rad + temp
                self.red.append([x, y])
            self.data.append([x, y])

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
    X = np.linspace(-20, 30)
    Y = np.linspace(-20, 20)
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

if __name__ == '__main__':
    D = Data(10, 5, 5)
    D.generate()
    x = D.blue + D.red
    y = [1]*len(D.blue) + [-1]*len(D.red)
    k_nn(x, y, 1)
    k_nn(x, y, 3)