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

def Dist(x1, x2):
    return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5

def K_Nearst_Neighbor(data, result, p, k):
    data = np.array(data)
    result = np.array(result)
    Dist_list = []
    for i in range(len(data)):
        Dist_list.append([Dist(data[i], p), i])
    Dist_list.sort()
    count = 0
    for i in range(k):
        count += result[Dist_list[i][1]]
    if(count > 0):
        return 1
    else:
        return -1


def get_Error(Train_x, Train_y, Validation_x, Validation_y, k):
    error = 0
    for i in range(len(Validation_x)):
        if(K_Nearst_Neighbor(Train_x, Train_y, Validation_x[i], k) != Validation_y[i]):
            error += 1
    return error / len(Validation_x)

def cross_validation(data, result, k):
    count = 0
    for i in range(len(data)):
        train_x = data[:i] + data[i+1:]
        train_y = result[:i] + result[i+1:]
        validation_x = data[i]
        validation_y = result[i]
        if(K_Nearst_Neighbor(train_x, train_y, validation_x, k) != validation_y):
            count += 1
    return count / len(data)

def decision_bond(data, result, k):
    data = np.array(data)
    result = np.array(result)
    X = np.linspace(-1, 1)
    Y = np.linspace(-1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if(K_Nearst_Neighbor(data, result, [X[i][j], Y[i][j]], k) == 1):
                Z[i][j] = 1
            else:
                Z[i][j] = -1
    plt.contourf(X, Y, Z)
    ones = []
    minus = []
    for i in range(len(result)):
        if(result[i] == 1):
            ones.append(data[i])
        else:
            minus.append(data[i])
    ones = np.array(ones)
    minus = np.array(minus)
    plt.scatter(ones[:,0], ones[:,1], c='r')
    plt.scatter(minus[:,0], minus[:,1], c='g')
    plt.show()

if __name__ == "__main__":
    file = "hw9.txt"
    data, result = read_data(file)
    data = convert_data(data)
    data = normalize(data)
    Train_x, Train_y, Validation_x, Validation_y = random_data(data, result)
    k_list = []
    for k in range(1,100):
        k_list.append(cross_validation(Train_x, Train_y, k))
    plt.plot(range(1,100), k_list)
    plt.show()
    print("best k: ",k_list.index(min(k_list))+1, "error: ", min(k_list))
    k_best = k_list.index(min(k_list))+1
    decision_bond(Train_x, Train_y, k_best)
    print("Ein = ", get_Error(Train_x, Train_y, Train_x, Train_y, k_best))
    print("Etest = ",get_Error(Train_x, Train_y, Validation_x, Validation_y, k_best))