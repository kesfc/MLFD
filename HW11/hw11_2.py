import numpy as np
import matplotlib.pyplot as plt
import random
import math
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

def Euclidean(x1, x2):
    return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5

def which_cluster(data, centers):
    min_dist = 1000000
    min_index = 0
    for j in range(len(centers)):
        dist = Euclidean(data, centers[j])
        if(dist < min_dist):
            min_dist = dist
            min_index = j
    return min_index

def find_Kcenters(data, k):
    centers = []
    centers.append(random.choice(data))
    while(len(centers) < k):
        max_dist = 0
        max_index = 0
        for i in range(len(data)):
            dist = 0
            for j in range(len(centers)):
                dist += Euclidean(data[i], centers[j])
            if(dist > max_dist):
                max_dist = dist
                max_index = i
        centers.append(data[max_index])  
    for k in range(5):
        clusters = []
        for i in range(len(centers)):
            clusters.append([])
        for i in range(len(data)):
            index = which_cluster(data[i], centers)
            clusters[index].append(data[i])
        for i in range(len(clusters)):
            if(len(clusters[i]) != 0):
                x=[]
                y=[]
                for j in range(len(clusters[i])):
                    x.append(clusters[i][j][0])
                    y.append(clusters[i][j][1])
                centers[i] = [sum(x)/len(x), sum(x)/len(x)]
    return centers


def calculate_z(point, center, k):
    r = 2/(math.sqrt(k))
    z = Euclidean(point, center)/r
    z = math.exp(-0.5*z**2)
    return z

def find_weight(x, y, centers):
    k = len(centers)
    Z = []
    for i in range(len(x)):
        temp = [1]
        for j in range(k):
            z = calculate_z(x[i], centers[j], k)
            temp.append(z)
        Z.append(temp)
    Z = np.array(Z)
    y = np.array(y)
    w = np.dot(np.linalg.pinv(Z), y)
    return w

def get_error(w, x, y, centers):
    k = len(centers)
    count = 0
    for i in range(len(x)):
        temp = [1]
        for j in range(k):
            z = calculate_z(x[i], centers[j], k)
            temp.append(z)
        if(np.dot(w, temp)*y[i] < 0):
            count += 1
    return count/len(x)

def cross_validation(x, y, k):
    Z = []
    for i in range(len(x)):
        temp = [1]
        for j in range(k):
            z = calculate_z(x[i], x[j], k)
            temp.append(z)
        Z.append(temp)
    count = 0
    for i in range(len(x)):
        train_z = Z[:i] + Z[i+1:]
        train_y = y[:i] + y[i+1:]
        train_z = np.array(train_z)
        train_y = np.array(train_y)
        w = np.dot(np.linalg.pinv(train_z), train_y)
        if(np.dot(w, Z[i])*y[i] < 0):
            count += 1
    return count/len(x)

def decision_bond(w,centers, x, y):
    k = len(centers)
    r = 2/(math.sqrt(k))
    X = np.linspace(-1, 1)
    Y = np.linspace(-1, 1)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            temp = [1]
            for l in range(k):
                z = calculate_z([X[i][j], Y[i][j]], centers[l], k)
                temp.append(z)
            if(np.dot(w, temp) > 0):
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
    for k in range(1, 101):
        E_cv = cross_validation(Train_x, Train_y, k)
        print("k = ", k, "E_cv = ", E_cv)
        k_list.append(E_cv)
    plt.plot(range(1, 101), k_list)
    plt.show()
    k_min = min(k_list)
    index = k_list.index(k_min)
    print("best k: ", index+1, "error: ", k_min)
    centers = find_Kcenters(Train_x, index+1)
    w = find_weight(Train_x, Train_y, centers)
    Ein = get_error(w, Train_x, Train_y, centers)
    print("Ein = ", Ein)
    Etest = get_error(w, Validation_x, Validation_y, centers)
    print("Etest = ", Etest)
    decision_bond(w, centers, Train_x, Train_y)