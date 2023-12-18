import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    plt.scatter(x = 1, y = 0, color = "blue")
    plt.scatter(x = -1, y=0, color = "red")
    x1 = np.linspace(-1, 1)
    plt.plot(x1*0, x1, color = "blue")
    plt.plot(x1, x1**3, color = "red")
    plt.show()