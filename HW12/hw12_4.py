import numpy as np
import matplotlib.pyplot as plt
import random
import cvxopt

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
            result.append(1.0)
        else:
            temp2 = []
            for i in range(1, len(temp)):
                temp2.append(float(temp[i]))
            data.append(temp2)
            result.append(-1.0)
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


def kernel_function(x, x_prime):
    return (1 + np.dot(x.T, x_prime)) ** 8



def find_s(X, y, C):
    K = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = kernel_function(X[i], X[j])
    P =  np.outer(y, y) * K
    q = np.array([-1]*len(X), dtype = float)
    G = np.block([[np.eye(len(X)) * 1], [np.eye(len(X)) * -1]])
    h = np.block([np.full(len(X), C), np.zeros(len(X))])
    alpha = cvxopt.solvers.qp(
        cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), 
        cvxopt.matrix(y, (1, len(X))), cvxopt.matrix(0.0))["x"]
    s = []
    for i in range(len(alpha)):
        if alpha[i] > 0:
            if alpha[i] < C:
                s.append([X[i], y[i], alpha[i]])
            else:
                s.append([X[i], y[i], C])
    bias = y[0] - np.sum([s[i][2] * s[i][1] * kernel_function(s[i][0], s[0][0]) for i in range(len(s))])
    return s, bias

def get_Error(s, b, x, y):
    count = 0
    for i in range(len(x)):
        sign = 0
        for j in s:
            sign += j[2] * j[1] * kernel_function(j[0], x[i])
        prediction += b
        if np.sign(sign) != y[i]:
            count += 1
    return count / len(x)


if __name__ == "__main__":
    file = "hw9.txt"
    data, result = read_data(file)
    data = convert_data(data)
    data = normalize(data)
    Train_x, Train_y, Validation_x, Validation_y = random_data(data, result)
    # Train the SVM
    s,b = find_s(np.array(Train_x), np.array(Train_y), 0.01)
    X = np.linspace(-1, 1)
    Y = np.linspace(-1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = Z = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            sign = 0
            for k in s:
                sign += k[2] * k[1] * kernel_function(k[0], [X[i][j],Y[i][j]])
            sign += b
            Z[i][j] = np.sign(sign)

    plt.contourf(X, Y, Z)

    # Plot data points
    plt.scatter(np.array(Train_x)[:,0], np.array(Train_x)[:,1], c = np.array(Train_y), cmap = plt.cm.Paired)
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.title('SVM Decision Boundary')
    plt.show()

    Clist = [0.01]
    Error = []
    slist = []
    blist = []
    for i in Clist:
        E = 0
        for j in range(len(Train_x)):
            train_x = Train_x[:j] + Train_x[j+1:]
            train_y = Train_y[:j] + Train_y[j+1:]
            s, b = find_s(np.array(train_x), np.array(train_y), i)
            slist.append(s)
            blist.append(b)
            sign = 0
            for k in range(len(s)):
                sign += s[k][2] * s[k][1] * kernel_function(s[k][0], Train_x[j])
            sign += b
            if np.sign(sign) != Train_y[j]:
                E += 1
        Error.append(E / len(Train_x))
    min_error = min(Error)
    min_index = Error.index(min_error)
    print("The best C is:", Clist[min_index])
    print("The best error is:", min_error)
    Eval = get_Error(slist[min_index], blist[min_index], Validation_x, Validation_y)
    print("The Eval is:", Eval)
    Z = Z = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            sign = 0
            for k in slist[min_index]:
                sign += k[2] * k[1] * kernel_function(k[0], [X[i][j],Y[i][j]])
            sign += blist[min_index]
            Z[i][j] = np.sign(sign)
    plt.contourf(X, Y, Z)
    plt.scatter(np.array(Validation_x)[:,0], np.array(Validation_x)[:,1], c = np.array(Validation_y), cmap = plt.cm.Paired, s = 1)
    plt.show()

   