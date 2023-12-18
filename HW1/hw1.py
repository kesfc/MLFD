import matplotlib.pyplot as plt
import numpy as np
import random
#just change this number to change number of data need
DataSet = 1000

# Define target function f and corresponding weight
a = 3
b = 4
x1 = np.linspace(0, 1000)  # Adjust the range as needed
y1 = a * x1 + b
W_t = [-8, -6, 2]

# Define current g and corresponding weight
x2 = np.linspace(0, 1000)
y2 = 0 * x2 + 0
W_g = [0, 0, 0]

Tx = []
Ty = []
Fx = []
Fy = []
# Generate Datas
for i in range(DataSet):
    temp1 = (random.randint(1,1000))
    temp2 = (random.randint(1,1000))
    if((W_t[0] + W_t[1]*temp1 + W_t[2]*temp2) > 0):
        Tx.append(temp1)
        Ty.append(temp2)
    else:
        Fx.append(temp1)
        Fy.append(temp2)
plt.scatter(Tx, Ty, label = 'True', color = "green", marker = "x")
plt.scatter(Fx, Fy, label = 'False', color = "yellow", marker = "o")
plt.plot(x1, y1, label='target function: y = {}x + {}'.format(a, b))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
# PLA algorithim
Check1 = False
Check2 = False
count = 0
while Check1 == False or Check2 == False:
    count +=1
    Check1 = True
    for i in range(len(Tx)):
        if((W_g[0]+W_g[1]*Tx[i]+W_g[2]*Ty[i] <= 0)):
            W_g[0] +=1
            W_g[1] +=Tx[i]
            W_g[2] += Ty[i]
            Check1 = False
            break
    if(Check1 == False):
        continue
    else:
        for i in range(len(Fx)):
            Check2 = True
            if((W_g[0]+W_g[1]*Fx[i]+W_g[2]*Fy[i] > 0)):
                W_g[0] -=1
                W_g[1] -=Fx[i]
                W_g[2] -= Fy[i]
                Check2 = False
                break
# Update g based on it's W
plt.scatter(Tx, Ty, label = 'True', color = "green", marker = "x")
plt.scatter(Fx, Fy, label = 'False', color = "yellow", marker = "o")
plt.plot(x1, y1, label='target function: y = {}x + {}'.format(a, b))
plt.title("number of iteration {}".format(count))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
x2 = np.linspace(0, 1000)  # Adjust the range as needed
if(W_g[2] == 0):
    W_g[2] == 0.000001
y2 = -W_g[1]/W_g[2] * x1 + W_g[0]/W_g[2]
plt.plot(x2, y2, label='g function')
plt.show()