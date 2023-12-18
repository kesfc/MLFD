import numpy as np
import matplotlib.pyplot as plt
import random
import time

def Dist(x1, x2):
    return ((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**0.5

def Nearest_Neighbor(data, p):
    min_dist = 1000000
    min_index = 0
    for j in range(len(data)):
        dist = Dist(data[j], p)
        if(dist < min_dist):
            min_dist = dist
            min_index = j
    return data[min_index]
def find_centers(data, k):
    data = np.array(data)
    data = data.tolist()
    centers = []
    centers.append(data.pop(random.randint(0, len(data)-1)))
    while(len(centers) < k):
        max_dist = 0
        max_index = 0
        for i in range(len(data)):
            dist = 0
            for j in range(len(centers)):
                dist += Dist(data[i], centers[j])
            if(dist > max_dist):
                max_dist = dist
                max_index = i
        point = data.pop(max_index)
        centers.append(point)
    return centers

def which_cluster(data, centers):
    min_dist = 1000000
    min_index = 0
    for j in range(len(centers)):
        dist = Dist(data, centers[j])
        if(dist < min_dist):
            min_dist = dist
            min_index = j
    return min_index
def cluster(data, centers):
    for i in range(10):
        clusters = []
        for i in range(len(centers)):
            clusters.append([])
        for i in range(len(data)):
            index = which_cluster(data[i], centers)
            clusters[index].append(data[i])
        # update centers
        for i in range(len(clusters)):
            centers[i] = np.mean(np.array(clusters[i]), axis = 0).tolist()
    return clusters, centers
    
if __name__ == '__main__':
    data = []
    for i in range(10000):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        data.append([x1, x2])
    centers = find_centers(data, 10)
    clusters, centers = cluster(data, centers)
    for i in range(len(clusters)):  
        plt.scatter(np.array(clusters[i])[:,0], np.array(clusters[i])[:,1])
    plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], marker = 'x', color = 'black')
    plt.show()
    QP = []
    for i in range(0, 10000):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        QP.append([x1, x2])
    T = time.time()
    for i in range(len(QP)):
        Nearest_Neighbor(data, QP[i])
    print(time.time() - T)
    T = time.time()
    for i in range(len(QP)):
        index = which_cluster(QP[i], centers)
        Nearest_Neighbor(clusters[index], QP[i])
    print(time.time() - T)
    cluster1 = []
    centers = []
    data = []
    for i in range(10):
        t1 = random.uniform(0, 1)
        t2 = random.uniform(0, 1)
        center = [t1, t2]
        cluster1.append([])
        for i in range(1000):
            point = np.random.normal(center, 0.1)
            data.append(point)
            cluster1[-1].append(point)
        centers.append(center)
    cluster1, centers = cluster(data, centers)
    for i in range(len(cluster1)):
        plt.scatter(np.array(cluster1[i])[:,0], np.array(cluster1[i])[:,1])
    plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], marker = 'x', color = 'black')
    plt.show()
    QP = []
    for i in range(10000):
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        QP.append([x1, x2])
    time1 = time.time()
    for i in range(len(QP)):
        Nearest_Neighbor(data, QP[i])
    print(time.time() - time1)
    time1 = time.time()
    for i in range(len(QP)):
        index = which_cluster(QP[i], centers)
        Nearest_Neighbor(cluster1[index], QP[i])
    print(time.time() - time1)