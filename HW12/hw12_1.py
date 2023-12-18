import numpy as np
import matplotlib.pyplot as plt
class Neural_Netowrk:
    def __init__(self, input, output, hidden, weight = 0.15, m = 2):
        self.x = input[0]
        self.y = input[1]
        self.output = output
        self.hidden = hidden
        self.weight = weight
        self.m = m
    def forward(self):
        x0 = [1.0] + self.x
        s1 = []
        s1.append(x0[0] * self.weight + x0[1] * self.weight + x0[2] * self.weight)
        s1.append(x0[0] * self.weight + x0[1] * self.weight + x0[2] * self.weight)
        s1 = np.array(s1)
        if self.hidden == "Tanh":
            s1 = np.tanh(s1)
        x1 = np.concatenate((np.ones(1), s1))
        s2 = []
        s2.append(x1[0] * self.weight + x1[1] * self.weight + x1[2]*self.weight)
        s2 = np.array(s2)
        if(self.output == "Tanh"):
            x2 = np.tanh(s2)
        x0 = np.array(x0)
        return x0, x1, x2

    def backward(self, x):
        if(self.output == "Tanh"):
            d2 = 2*(x[2] - self.y) * (1 - x[2]**2)
        else:
            d2 = 2*(x[2] - self.y)
        if(self.hidden == "Tanh"):
            d1 = d2 * (1 - x[1][1:]**2) * np.full((2), self.weight)
        else:
            d1 = d2 * np.full((2), self.weight)
        return d1, d2

    def gradiant_decent(self, x, d1, d2):
        x0 = np.array([x[0]]).T
        d1 = np.array([d1])
        out1 = np.dot(x0, d1)
        x1 = np.array([x[1]]).T
        d2 = np.array([d2])
        out2 = np.dot(x1, d2)
        return out1, out2
            
if __name__ == "__main__":
    TT = Neural_Netowrk([[2.0,1.0],-1], "Tanh", "Tanh", 0.15, 2)
    TI = Neural_Netowrk([[2.0,1.0],-1], "Tanh", "Identity", 0.15, 2)
    x = TT.forward()
    d = TT.backward(x)
    print("d:", d)
    print(TT.gradiant_decent(x, d[0], d[1]))
    x = TI.forward()
    d = TI.backward(x)
    print("d:", d)
    print(TI.gradiant_decent(x, d[0], d[1]))
    TT = Neural_Netowrk([[2.0,1.0],-1], "Tanh", "Tanh", 0.15-0.0001, 2)
    TI = Neural_Netowrk([[2.0,1.0],-1], "Tanh", "Identity", 0.15+0.0001, 2)
    x = TT.forward()
    d = TT.backward(x)
    print(TT.gradiant_decent(x, d[0], d[1]))
    x = TI.forward()
    d = TI.backward(x)
    print(TI.gradiant_decent(x, d[0], d[1]))
