import numpy as np
import matplotlib.pyplot as plt
import random

def read_data(file_name):
    file = open(file_name, "r")
    lines = file.readlines()
    file.close()
    one = []
    five = []
    for line in lines:
        temp = line.split()
        if int(float(temp[0])) == 1:
            temp2 = []
            for i in range(1, len(temp)):
                temp2.append(float(temp[i]))
            one.append(temp2)
        elif int(float(temp[0])) == 5:
            temp2 = []
            for i in range(1, len(temp)):
                temp2.append(float(temp[i]))
            five.append(temp2)
    return one, five
def convert(one, five):
    ones = []
    fives = []
    for i in one:
        temp = []
        temp.append(intensity(i))
        temp.append(symmetric(i))
        ones.append(temp)
    for i in five:
        temp = []
        temp.append(intensity(i))
        temp.append(symmetric(i))
        fives.append(temp)
    return ones, fives
def intensity(data):
    count = 0
    for i in data:
        count += i
    return count / len(data)

def symmetric(data):
    image = np.array(data)
    image = image.reshape(16, 16)
    count = 0
    for i in range(16):
        for j in range(8):
            if image[i][j] == image[i][15-j]:
                count += 1
    return count/len(data) * 2
def error_rate(ones, fives, w):
    count = 0
    for i in ones:
        temp = w[0]
        for j in range(len(i)):
            temp += w[j+1]*i[j]
        if(temp < 0):
            count += 1
    for i in fives:
        temp = w[0]
        for j in range(len(i)):
            temp += w[j+1]*i[j]
        if(temp >= 0):
            count += 1
    return count/(len(ones)+len(fives))
def linear_regression(ones, fives):
    w = np.zeros(len(ones[0]) + 1)
    N = len(ones) + len(fives)
    x = []
    y = []
    for i in ones:
        temp = []
        temp.append(1)
        for j in range(len(i)):
            temp.append(i[j])
        x.append(temp)
        y.append(1)
    for i in fives:
        temp = []
        temp.append(1)
        for j in range(len(i)):
            temp.append(i[j])
        x.append(temp)
        y.append(-1)
    X = np.array(x)
    Y = np.array(y)
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    In_XTX = np.linalg.inv(XTX)
    T = np.dot(In_XTX, XT)
    W = np.dot(T, Y)
    return W

def pock_algo(ones, fives, w, maxi):
    best_w = [w[0],w[1], w[2]]
    error = error_rate(ones, fives, w)/len(ones+ fives)
    temp_w = w
    for i in range(maxi):
        for i in one:
            if(temp_w[0] + temp_w[1]*i[0] + temp_w[2]*i[1] <= 0):
                w[0] +=1
                w[1] +=i[0]
                w[2] += i[1]
                break
        for i in five:
            if(temp_w[0] + temp_w[1]*i[0] + temp_w[2]*i[1] >= 0):
                w[0] -=1
                w[1] -=i[0]
                w[2] -= i[1]
                break
        temp_error = error_rate(ones, fives, w)
        if(temp_error < error):
            error = temp_error
            best_w = [w[0], w[1], w[2]]
    return best_w

def Gradient_Descent(ones, fives, maxi):
    eta = 0.25
    w = np.zeros(len(ones[0]) + 1)
    N = len(ones) + len(fives)
    x = []
    y = []
    for i in ones:
        temp = []
        temp.append(1)
        for j in range(len(i)):
            temp.append(i[j])
        x.append(temp)
        y.append(1)
    for i in fives:
        temp = []
        temp.append(1)
        for j in range(len(i)):
            temp.append(i[j])
        x.append(temp)
        y.append(-1)
    x = np.array(x)
    y = np.array(y)
    for K in range(maxi):
        gt = -(1/N) * sum([x[n] * y[n]/(1 + np.exp(y[n]*((w.T).dot(x[n])))) for n in range(N)])
        w = w - eta*gt
    return w

def SGD(ones, fives, maxi):
    eta = 0.01
    w = np.zeros(len(ones[0]) + 1)
    N = len(ones) + len(fives)
    x = []
    y = []
    for i in ones:
        temp = []
        temp.append(1)
        for j in range(len(i)):
            temp.append(i[j])
        x.append(temp)
        y.append(1)
    for i in fives:
        temp = []
        temp.append(1)
        for j in range(len(i)):
            temp.append(i[j])
        x.append(temp)
        y.append(-1)
    x = np.array(x)
    y = np.array(y)
    for K in range(maxi):
        n = random.randint(0, N-1)
        gt = -(x[n] * y[n]/(1 + np.exp(y[n]*((w.T).dot(x[n])))))
        w = w - eta *gt
    return w        

if __name__ == "__main__":
    file_name = "hw6.txt"
    one, five = read_data(file_name)
    ones, fives = convert(one, five)
    #linear regression
    #train
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    W_linear = linear_regression(ones, fives)
    W_linear = pock_algo(ones, fives, W_linear, 1000)
    E_Linear = error_rate(ones, fives, W_linear)
    x = np.linspace(-1, 0.25)
    y = (-W_linear[0] - W_linear[1]*x)/W_linear[2]
    print("Linear Regression Error Rate: ", E_Linear)
    plt.title('Linear Regression')
    plt.plot(x, y, label='Linear Regression')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #test
    oneT, fiveT = read_data("hw7.txt")
    onesT, fivesT = convert(oneT, fiveT)
    plt.scatter(np.array(onesT)[:,0], np.array(onesT)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fivesT)[:,0], np.array(fivesT)[:,1], c='red', s=10, label='5', marker='x')
    E_LinearT = error_rate(onesT, fivesT, W_linear)
    x = np.linspace(-1, 0.25)
    y = (-W_linear[0] - W_linear[1]*x)/W_linear[2]
    print("Linear Regression Error Rate: ", E_LinearT)
    plt.title('Linear Regression test')
    plt.plot(x, y, label='Linear Regression')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #gradient decent
    #train
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    W_gradient = Gradient_Descent(ones, fives, 2500)
    E_Linear = error_rate(ones, fives, W_gradient)
    x = np.linspace(-1, 1)
    y = (-W_gradient[0] - W_gradient[1]*x)/W_gradient[2]
    print("Gradient Decent Error Rate: ", E_Linear)
    plt.title('Gradient Decent')
    plt.plot(x, y, label='Gradient Decent')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #test
    plt.scatter(np.array(onesT)[:,0], np.array(onesT)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fivesT)[:,0], np.array(fivesT)[:,1], c='red', s=10, label='5', marker='x')
    E_LinearT = error_rate(onesT, fivesT, W_gradient)
    x = np.linspace(-1, 1)
    y = (-W_gradient[0] - W_gradient[1]*x)/W_gradient[2]
    print("Gradient Decent Error Rate: ", E_LinearT)
    plt.title('Gradient Decent test')
    plt.plot(x, y, label='Gradient Decent')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #SGD
    #train
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    W_SGD = SGD(ones, fives, 1000000)
    E_DSGD = error_rate(ones, fives, W_SGD)
    x = np.linspace(-1, 1)
    y = (-W_SGD[0] - W_SGD[1]*x)/W_SGD[2]
    print("SGD Error Rate: ", E_DSGD)
    plt.title('SGD')
    plt.plot(x, y, label='SGD')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #test
    plt.scatter(np.array(onesT)[:,0], np.array(onesT)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fivesT)[:,0], np.array(fivesT)[:,1], c='red', s=10, label='5', marker='x')
    E_DSGDT = error_rate(onesT, fivesT, W_SGD)
    x = np.linspace(-1, 1)
    y = (-W_SGD[0] - W_SGD[1]*x)/W_SGD[2]
    print("SGD Error Rate: ", E_DSGDT)
    plt.title('SGD test')
    plt.plot(x, y, label='SGD')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()

    #3rd order polynomial 
    new_ones = []
    new_fives = []
    for i in ones:
        temp = []
        temp.append(i[0])
        temp.append(i[1])
        temp.append(i[0]**2)
        temp.append(i[1]**2)
        temp.append(i[0]*i[1])
        temp.append(i[0]**3)
        temp.append(i[1]**3)
        temp.append(i[0]**2*i[1])
        temp.append(i[0]*i[1]**2)
        new_ones.append(temp)
    for i in fives:
        temp = []
        temp.append(i[0])
        temp.append(i[1])
        temp.append(i[0]**2)
        temp.append(i[1]**2)
        temp.append(i[0]*i[1])
        temp.append(i[0]**3)
        temp.append(i[1]**3)
        temp.append(i[0]**2*i[1])
        temp.append(i[0]*i[1]**2)
        new_fives.append(temp)
    new_oneT = []
    new_fiveT = []
    for i in onesT:
        temp = []
        temp.append(i[0])
        temp.append(i[1])
        temp.append(i[0]**2)
        temp.append(i[1]**2)
        temp.append(i[0]*i[1])
        temp.append(i[0]**3)
        temp.append(i[1]**3)
        temp.append(i[0]**2*i[1])
        temp.append(i[0]*i[1]**2)
        new_oneT.append(temp)
    for i in fivesT:
        temp = []
        temp.append(i[0])
        temp.append(i[1])
        temp.append(i[0]**2)
        temp.append(i[1]**2)
        temp.append(i[0]*i[1])
        temp.append(i[0]**3)
        temp.append(i[1]**3)
        temp.append(i[0]**2*i[1])
        temp.append(i[0]*i[1]**2)
        new_fiveT.append(temp)
    #linear regression 
    # train
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    w = linear_regression(new_ones, new_fives)
    print(w)
    E_poly = error_rate(new_ones, new_fives, w)
    print("linear regression: ", E_poly)
    x = np.linspace(-1, 1)
    y = np.linspace(0, 1)
    x1, x2 = np.meshgrid(x, y)
    function = w[0] + w[1]*x1 + w[2]*x2 + w[3]*x1**2 + w[4]*x2**2 + w[5]*x1*x2 + w[6]*x1**3 + w[7]*x2**3 + w[8]*x1**2*x2 + w[9]*x1*x2**2
    plt.contour(x1, x2, function, [0])
    plt.title('3rd order polynomial')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #test
    plt.scatter(np.array(onesT)[:,0], np.array(onesT)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fivesT)[:,0], np.array(fivesT)[:,1], c='red', s=10, label='5', marker='x')
    E_polyT = error_rate(new_oneT, new_fiveT, w)
    print("3rd order polynomial Error Rate: ", E_polyT)
    x = np.linspace(-1, 1)
    y = np.linspace(0, 1)
    x1, x2 = np.meshgrid(x, y)
    plt.contour(x1, x2, function, [0])
    plt.title('3rd order polynomial test')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()

    #Gradient Decent
    #train
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    W = Gradient_Descent(new_ones, new_fives, 2500)
    E = error_rate(new_ones, new_fives, W)
    print("Gradient Decent Error Rate: ", E)
    x = np.linspace(-1, 1)
    y = np.linspace(0, 1)
    x1, x2 = np.meshgrid(x, y)
    function = W[0] + W[1]*x1 + W[2]*x2 + W[3]*x1**2 + W[4]*x2**2 + W[5]*x1*x2 + W[6]*x1**3 + W[7]*x2**3 + W[8]*x1**2*x2 + W[9]*x1*x2**2
    plt.contour(x1, x2, function, [0])
    plt.title('Gradient Decent')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #test
    plt.scatter(np.array(onesT)[:,0], np.array(onesT)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fivesT)[:,0], np.array(fivesT)[:,1], c='red', s=10, label='5', marker='x')
    E = error_rate(new_oneT, new_fiveT, W)
    print("Gradient Decent Error Rate: ", E)
    x = np.linspace(-1, 1)
    y = np.linspace(0, 1)
    x1, x2 = np.meshgrid(x, y)
    plt.contour(x1, x2, function, [0])
    plt.title('Gradient Decent test')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    #SGD
    #train
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    W = SGD(new_ones, new_fives, 1000000)
    E = error_rate(new_ones, new_fives, W)
    x = np.linspace(-1, 1)
    y = np.linspace(0, 1)
    x1, x2 = np.meshgrid(x, y)
    function = W[0] + W[1]*x1 + W[2]*x2 + W[3]*x1**2 + W[4]*x2**2 + W[5]*x1*x2 + W[6]*x1**3 + W[7]*x2**3 + W[8]*x1**2*x2 + W[9]*x1*x2**2
    plt.contour(x1, x2, function, [0])
    plt.title('SGD')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()
    print("SGD Error Rate: ", E)
    #test
    plt.scatter(np.array(onesT)[:,0], np.array(onesT)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fivesT)[:,0], np.array(fivesT)[:,1], c='red', s=10, label='5', marker='x')
    E = error_rate(new_oneT, new_fiveT, W)
    print("SGD Error Rate: ", E)
    x = np.linspace(-1, 1)
    y = np.linspace(0, 1)
    x1, x2 = np.meshgrid(x, y)
    plt.contour(x1, x2, function, [0])
    plt.title('SGD test')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()

