import numpy as np
import matplotlib.pyplot as plt
import random

def read_data(file):
    file = open(file, "r")
    lines = file.readlines()
    file.close()
    data = []
    result = []
    for line in lines:
        temp = line.split()
        if int(float(temp[0])) == 1:
            temp2 = []
            for i in range(1, len(temp)):
                temp2.append(float(temp[i]))
            data.append(temp2)
            result.append(1)
        else:
            temp2 = []
            for i in range(1, len(temp)):
                temp2.append(float(temp[i]))
            data.append(temp2)
            result.append(-1)
    return data, result

def intensity(data):
    count = 0
    for i in data:
        count += i
    return count

def symmetric(data):
    image = np.array(data)
    image = image.reshape(16, 16)
    count = 0
    for i in range(16):
        for j in range(16):
            count += abs(image[i][j] - image[i][15-j])
    return count

def convert_data(data):
    new_data = []
    for i in data:
        S = symmetric(i)
        I = intensity(i)
        new_data.append([S, I])
    return new_data

def normalize(data):
    new_data = []
    data1 = np.array(data)[:,0]
    data2 = np.array(data)[:,1]
    Max1 = max(data1)
    Min1 = min(data1)
    Max2 = max(data2)
    Min2 = min(data2)
    for i in range(0, len(data1)):
        data1[i] = (data1[i] - Min1) / (Max1 - Min1) * 2 - 1
        data2[i] = (data2[i] - Min2) / (Max2 - Min2) * 2 - 1
        data1[i] = round(data1[i], 2)
        data2[i] = round(data2[i], 2)
        new_data.append([data1[i], data2[i]])
    
    return new_data

def random_data(data, result):
    test_data = []
    test_result = []
    for i in range(300):
        index = random.randint(0, len(data)-1)
        test_data.append(data[index])
        test_result.append(result[index])
        data.pop(index)
        result.pop(index)
    return test_data, test_result, data, result

class Neural_Netowrk:
    def __init__(self, x, y, m = 10, learning_rate = 0.1, beta = 0.8, alpha = 1.2, weight = 0.3):
        self.X = x
        self.Y = y
        self.m = m
        self.learning_rate = learning_rate
        self.ein = 1
        self.layer1 = np.full((3, m), weight)
        self.layer2 = np.full((m+1, 1), weight)
        self.beta = beta
        self.alpha = alpha
    def forward(self, x, weight1 = None, weight2 = None):
        if weight1 is None:
            weight1 = self.layer1
        if weight2 is None:
            weight2 = self.layer2
        x0 = np.concatenate((np.ones(1), np.array(x)))
        s1 = np.matmul(weight1.T, x0)
        s1 = np.tanh(s1)
        x1 = np.concatenate((np.ones(1), s1))
        s2 = np.matmul(weight2.T, x1)
        x2 = s2
        return x0, x1, x2
    def backward(self, x, y):
        d2 = 2*(x[2] - y)
        d1 = (1 - x[2]**2)* (self.layer2 @ d2)[1:]
        return d1, d2
    def gradiant_numeric(self, x, d1, d2):
        x0 = np.array([x[0]]).T
        d1 = np.array([d1])
        out1 = np.dot(x0, d1)
        x1 = np.array([x[1]]).T
        d2 = np.array([d2])
        out2 = np.dot(x1, d2)
        return out1, out2
    
    def gradiant_decent(self):
        g1 = np.zeros((3, self.m))
        g2 = np.zeros((self.m+1, 1))
        ein = 0
        for i in range(len(self.X)):
            x = self.forward(self.X[i])
            d = self.backward(x, self.Y[i])
            g = self.gradiant_numeric(x, d[0], d[1])
            g1 += g[0]*self.learning_rate / len(self.X)
            g2 += g[1]*self.learning_rate / len(self.X)
            ein +=0.25/len(self.X) * (x[2] - self.Y[i])**2
        temp1 = self.layer1 - g1
        temp2 = self.layer2 - g2
        new_ein = 0
        for i in range(len(self.X)):
            x = self.forward(self.X[i], temp1, temp2)
            new_ein += 0.25/len(self.X) * (x[2] - self.Y[i])**2 
        if(new_ein < ein):
            self.layer1 = temp1
            self.layer2 = temp2
            self.learning_rate *= self.alpha
            return new_ein
        else:
            self.learning_rate *= self.beta
            return ein
    
    def SGD(self):
        elist = []
        weightlist = []
        eauglist = []
        ein = 1
        for i in range(200000):
            index = random.randint(0, len(self.X)-1)
            x = self.forward(self.X[index])
            d = self.backward(x, self.Y[index])
            g = self.gradiant_numeric(x, d[0], d[1])
            g1 = g[0]
            g2 = g[1]
            self.layer1 -= g1*self.learning_rate
            self.layer2 -= g2*self.learning_rate
            new_ein = 0
            for j in range(len(self.X)):
                x = self.forward(self.X[j])
                new_ein += 0.25/len(self.X) * (x[2] - self.Y[j])**2
            new_ein = list(new_ein)[0]
            if(new_ein < ein):
                elist.append(ein)
                ein = new_ein
            else:
                elist.append(ein)
            eauglist.append(self.eaug(ein))
            weightlist.append([self.layer1, self.layer2])
            if(i%10000 == 0):
                print(i, ein)
        return elist, weightlist, eauglist
    
    def SGD2(self, ein):
        index = random.randint(0, len(self.X)-1)
        x = self.forward(self.X[index])
        d = self.backward(x, self.Y[index])
        g = self.gradiant_numeric(x, d[0], d[1])
        g1 = g[0]
        g2 = g[1]
        temp1 = self.layer1 - g1*self.learning_rate
        temp2 = self.layer2 - g2*self.learning_rate
        new_ein = 0
        for j in range(len(self.X)):
            x = self.forward(self.X[j], temp1, temp2)
            new_ein += 0.25/len(self.X) * (x[2] - self.Y[j])**2
        new_ein = list(new_ein)[0]
        if(new_ein < ein):
            self.layer1 = temp1
            self.layer2 = temp2
            self.learning_rate *= self.alpha
            return new_ein
        else:
            self.learning_rate *= self.beta
            return ein

    def eaug(self, ein, lamda = 0.01):
        eaug = ein.copy()
        eaug += lamda/len(self.X) * np.sum(self.layer1**2)
        eaug += lamda/len(self.X) * np.sum(self.layer2**2)
        return eaug

def decision_bond(x, y, weight):
    X = np.linspace(-1, 1)
    Y = np.linspace(-1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if(Neural_Netowrk([i,j], 1).forward([X[i][j], Y[i][j]],weight[0],weight[1])[2] > 0):
                Z[i][j] = 1
            else:
                Z[i][j] = -1
    plt.contourf(X, Y, Z)
    ones = []
    minus = []
    for i in range(len(y)):
        if(y[i] == 1):
            ones.append(x[i])
        else:
            minus.append(x[i])
    ones = np.array(ones)
    minus = np.array(minus)
    plt.scatter(ones[:,0], ones[:,1], c = "red")
    plt.scatter(minus[:,0], minus[:,1], c = "blue")
    plt.show()
    
if __name__ == "__main__":
    file = "hw9.txt"
    data, result = read_data(file)
    data = convert_data(data)
    data = normalize(data)
    Train_x, Train_y, Validation_x, Validation_y = random_data(data, result)
    TI = Neural_Netowrk(Train_x, Train_y, 10, 0.01, 0.5)
    # elist, weightlist, eauglist = TI.SGD()
    # plt.plot(range(1,len(elist)+1), elist)
    # plt.show()
    # plt.plot(range(1,len(eauglist)+1), eauglist)
    # plt.show()
    # Emin = min(eauglist)
    # print("Emin: ", Emin, "index", eauglist.index(Emin))
    # decision_bond(Train_x, Train_y, weightlist[-1])
    # decision_bond(Train_x, Train_y, weightlist[eauglist.index(Emin)])
    elist = []
    eauglist = []
    weightlist = []
    for i in range(2000000):
        elist.append(TI.gradiant_decent())
        eauglist.append(TI.eaug(elist[-1]))
        weightlist.append([TI.layer1, TI.layer2])
        if(i%1000 == 0):
            print(i)
    elist.pop(0)
    plt.plot(range(1,len(elist)+1), elist)
    plt.show()
    plt.plot(range(1,len(eauglist)+1), eauglist)
    plt.show()
    decision_bond(Train_x, Train_y, [TI.layer1, TI.layer2])
    Emin = min(eauglist)
    print("Emin: ", Emin, "index", eauglist.index(Emin))
    decision_bond(Train_x, Train_y, weightlist[eauglist.index(Emin)])
    Test = Neural_Netowrk(Validation_x, Validation_y, 10, 0.01, 0.5)
    Test.layer1 = weightlist[eauglist.index(Emin)][0]
    Test.layer2 = weightlist[eauglist.index(Emin)][1]
    etest = 0
    for i in range(len(Validation_x)):
        x = Test.forward(Validation_x[i])
        etest += 0.25/len(Validation_x) * (x[2] - Validation_y[i])**2
    print("Etest: ", etest)
    # V_x = Train_x[0:50]
    # V_y = Train_y[0:50]
    # T_x = Train_x[50:]
    # T_y = Train_y[50:]
    # Early_stop = Neural_Netowrk(T_x, T_y, 10, 0.01)
    # validation = Neural_Netowrk(V_x, V_y, 10, 0.01)
    # e_v0 = [1]
    # e_v1 = [0.9]
    # einlist = [1]
    # evalist = []
    # weightlist = []
    # count = 0
    # while(count < 1000):
    #     print(e_v0[0], e_v1[0])
    #     e_v0 = e_v1
    #     ein = Early_stop.SGD2(einlist[-1])
    #     validation.layer1 = Early_stop.layer1
    #     validation.layer2 = Early_stop.layer2
    #     einlist.append(ein)
    #     evalist.append(e_v0[0])
    #     weightlist.append([validation.layer1, validation.layer2])
    #     e_v1 = 0
    #     for i in range(len(V_x)):
    #         x = validation.forward(V_x[i])
    #         e_v1 += 0.25/len(V_x) * (x[2] - V_y[i])**2
    #     if(e_v1 >= e_v0[0]):
    #         count += 1
    
    # plt.plot(range(1,len(einlist)+1), einlist, label = "Ein")
    # plt.plot(range(1,len(evalist)+1), evalist, label = "Eval")
    # plt.show()
    # Emin = min(evalist)
    # print("Emin: ", Emin, "index", evalist.index(Emin))
    # decision_bond(Train_x, Train_y, weightlist[evalist.index(Emin)])
    # Etest = 0
    
    




