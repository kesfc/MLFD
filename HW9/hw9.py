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


def LP(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*LP(n-1,x)-(n-1)*LP(n-2,x))/n

def trans_8_order(data):
    new_data = []
    for k in data:
        temp = []
        x1 = k[0]
        x2 = k[1]
        for i in range(0,9):
            for j in range(0,i+1):
                temp.append(LP(j, x1) * LP(i-j, x2))
        new_data.append(temp)
    return new_data

def linear_regression(Z, y, lamda=0):
    Z = np.array(Z)
    y = np.array(y)
    I = np.identity(len(Z[0]))
    W = np.linalg.inv(Z.T.dot(Z) + lamda*I).dot(Z.T).dot(y)
    return W

def set_function(w, x1, x2):
    function = 0
    w_count = 0
    for i in range(0,9):
        for j in range(0,i+1):
            function += w[w_count] * LP(j, x1) * LP(i-j, x2)
            w_count += 1
    return function 

def error_counting(data, result, w):
    count = 0
    for i in range(0, len(data)):
        x1 = data[i][0]
        x2 = data[i][1]
        sign = set_function(w, x1, x2)
        if sign * result[i] < 0:
            count += 1
    return count

def get_H(Z, lamda):
    Z = np.array(Z)
    I = np.identity(len(Z[0]))
    H = Z.dot(np.linalg.inv(Z.T.dot(Z) + lamda*I)).dot(Z.T)
    return H

def cross_validation(N, y, H):
    E_cv = 0
    y_cap = H.dot(y)
    for i in range(0, N):
        E_cv += ((y[i] - y_cap[i])/(1 - H[i][i])) ** 2
    E_cv = E_cv / N
    return E_cv



if __name__ == "__main__":
    file = "hw9.txt"
    data, result = read_data(file)
    data = convert_data(data)
    data = normalize(data)
    Train_x, Train_y, Validation_x, Validation_y = random_data(data, result)
    #Q2
    plt.scatter(np.array(Train_x)[:,0], np.array(Train_x)[:,1], c = Train_y)
    x = np.linspace(-1,1)
    y = np.linspace(-1,1)
    x1, x2 = np.meshgrid(x,y)
    Order8 = trans_8_order(Train_x)
    w = linear_regression(Order8, Train_y, 0)
    print("Ein = ", error_counting(Train_x, Train_y, w)/len(Train_x))
    function = set_function(w, x1, x2)
    plt.contour(x1, x2, function, [0], colors = 'red')
    plt.show()
    #Q3
    plt.scatter(np.array(Train_x)[:,0], np.array(Train_x)[:,1], c = Train_y)
    x = np.linspace(-1,1)
    y = np.linspace(-1,1)
    x1, x2 = np.meshgrid(x,y)
    w = linear_regression(Order8, Train_y, 2)
    print("Ein = ", error_counting(Train_x, Train_y, w)/len(Train_x))
    function = set_function(w, x1, x2)
    plt.contour(x1, x2, function, [0], colors = 'red')
    plt.show()
    #Q4
    lamda_list = []
    i = 0
    while i <=2:
        lamda_list.append(i)
        i += 0.01
    E_cv_list = []
    E_test_list = []
    for i in lamda_list:
        H = get_H(Order8, i)
        E_cv = cross_validation(len(Train_x), Train_y, H)
        w = linear_regression(Order8, Train_y, i)
        E_test = error_counting(Validation_x, Validation_y, w)/len(Validation_x)
        E_cv_list.append(E_cv)
        E_test_list.append(E_test)
    plt.plot(lamda_list, E_cv_list, label = "E_cv")
    plt.plot(lamda_list, E_test_list, label = "E_test")
    plt.legend()
    plt.show()

    #Q5
    min_lamda = lamda_list[E_cv_list.index(min(E_cv_list))]
    print("min_lamda = ", min_lamda)
    w = linear_regression(Order8, Train_y, min_lamda)
    plt.scatter(np.array(Validation_x)[:,0], np.array(Validation_x)[:,1], c = Validation_y)
    x = np.linspace(-1,1)
    y = np.linspace(-1,1)
    x1, x2 = np.meshgrid(x,y)
    function = set_function(w, x1, x2)
    plt.contour(x1, x2, function, [0], colors = 'red')
    plt.show()
    print("Etest = ", error_counting(Train_x, Train_y, w)/len(Train_x))

    #Q6
    E_test = error_counting(Validation_x, Validation_y, w)/len(Validation_x)
    print("Etest = ", E_test)