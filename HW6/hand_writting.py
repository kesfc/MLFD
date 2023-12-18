import numpy as np
import matplotlib.pyplot as plt

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
        for j in range(8):
            if image[i][j] == image[i][15-j]:
                count += 1
    return count*2

if __name__ == "__main__":
    file = open("hw6.txt", "r")
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
    #part1
    image1 = one[0]
    image1 = np.array(image1)
    image1 = image1.reshape(16, 16)
    plt.imshow(image1, cmap='gray')
    plt.show()
    image5 = five[0]
    image5 = np.array(image5)
    image5 = image5.reshape(16, 16)
    plt.imshow(image5, cmap='gray')
    plt.show()
    plt.close()
    #part3:
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
    plt.scatter(np.array(ones)[:,0], np.array(ones)[:,1], c='blue', s=10, label='1', marker='o')
    plt.scatter(np.array(fives)[:,0], np.array(fives)[:,1], c='red', s=10, label='5', marker='x')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.legend()
    plt.show()