import matplotlib.pyplot as plt
import numpy as np
import random

class Data:
    def __init__(self, rad, sep, thk, count = 2000):
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

def pla(data1, data2):
    Break1 = False
    Break2 = False
    W = [0, 0, 0]
    count = 0
    while(Break1 == False or Break2 == False):
        count +=1
        Break1 = True
        Break2 = True
        for i in data1:
            if((W[0]+W[1]*i[0]+W[2]*i[1] <= 0)):
                W[0] +=1
                W[1] +=i[0]
                W[2] += i[1]
                Break1 = False
                break
        if(Break1 == False):
            continue
        else:
            for i in data2:
                if((W[0]+W[1]*i[0]+W[2]*i[1] >= 0)):
                    W[0] -=1
                    W[1] -=i[0]
                    W[2] -= i[1]
                    Break2 = False
                    break
    return W, count
def linear_regression(red, blue):
    X = []
    Y = []
    for i in red:
        temp = []
        temp.append(1)
        temp.append(i[0])
        temp.append(i[1])
        X.append(temp)
        Y.append(1)
    for i in blue:
        temp = []
        temp.append(1)
        temp.append(i[0])
        temp.append(i[1])
        X.append(temp)
        Y.append(-1)
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    In_XTX = np.linalg.inv(XTX)
    T = np.dot(In_XTX, XT)
    W = np.dot(T, Y)
    return W

            

if __name__ == '__main__':
    #3.1
    data = Data(10, 5, 5)
    data.generate()
    blue = np.array(data.blue)
    red = np.array(data.red)
    plt.scatter(blue[:,0], blue[:,1], c='blue', s=1)
    plt.scatter(red[:,0], red[:,1], c='red', s=1)
    plt.title('Generated Data')
    W = pla(blue, red)[0]
    x = np.linspace(-data.rad - data.thk, 2*data.rad + 2*data.thk)
    y = W[1]/W[2]*x - W[0]/W[2]
    plt.plot(x, y, color='black', label='PLA')
    W2 = linear_regression(red, blue)
    x2 = np.linspace(-data.rad - data.thk, 2*data.rad + 2*data.thk)
    y2 = W2[1]*x2 + W2[0]
    plt.plot(x2, y2, color='green', label='Linear Regression')
    plt.legend()
    plt.show()
    #3.2
    sep = []
    temp = 0.2
    while temp <= 5:
        sep.append(temp)
        temp += 0.2
    iteration = []
    for i in sep:
        data = Data(10, i, 5)
        data.generate()
        blue = np.array(data.blue)
        red = np.array(data.red)
        W = pla(blue, red)[1]
        iteration.append(W)
    plt.plot(sep, iteration)
    plt.xlabel('sep')
    plt.ylabel('iteration')
    plt.title('sep vs iteration')
    plt.show()