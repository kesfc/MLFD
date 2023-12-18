import numpy as np
import matplotlib.pyplot as plt

def Pmax(N, x, u):
    low = int(3 - 6*x)
    up = int(3 + 6*x)
    s = 0
    for i in range(low, up + 1):
        chose = 1
        for j in range(1, N + 1):
            chose = chose * j
        for j in range(1, i + 1):
            chose = chose/j
        for j in range(1, N - i + 1):
            chose = chose/j
        s += chose * (u ** i) * ((1 - u) ** (N - i))
    return 1 - s ** 2

x = np.arange(0, 1, 0.01)
y1 = []
y2 = []
for i in x:
    y1.append(Pmax(6, i, 0.5))
    y2.append(2 * np.exp(-2 * 6 * (i ** 2)))

plt.plot(x, y1, label="P")
plt.plot(x, y2, label="Hoeffding bond")
plt.xlabel("eplison")
plt.ylabel("P")
plt.legend()
plt.show()