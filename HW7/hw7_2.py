import matplotlib.pyplot as plt
import math 
def gradient_descent(x, y, eta):
    f_out = []
    x_out = []
    y_out = []
    for i in range(50):
        f = x**2 + 2 * y**2 + 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)
        f_out.append(f)
        x_temp = x - eta * (2 * x + 4 * math.pi * math.cos(2 * math.pi * x) * math.sin(2 * math.pi * y))
        y = y - eta * (4 * y + 4 * math.pi * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y))
        x = x_temp
        x_out.append(x)
        y_out.append(y)
    return f_out, x_out, y_out
if __name__ == '__main__':
    #part a
    f_out, x, y = gradient_descent(0.1, 0.1, 0.01)
    itr = []
    for i in range(1, 51):
        itr.append(i)
    plt.plot(itr, f_out)
    plt.xlabel('iteration')
    plt.ylabel('f')
    plt.show()
    f_out, x, y = gradient_descent(0.1, 0.1, 0.1)
    plt.plot(itr, f_out)
    plt.xlabel('iteration')
    plt.ylabel('f')
    plt.show()
    #part b
    x_y = [(0.1, 0.1), (1, 1), (-0.5, -0.5), (-1, -1)]
    etas = [0.1, 0.01]
    for i in x_y:
        for j in etas:
            f_out, x_out, y_out = gradient_descent(i[0], i[1], j)
            f_min = min(f_out)
            x = x_out[f_out.index(f_min)]
            y = y_out[f_out.index(f_min)]
            print("Initial point: ", i, "eta: ", j, "x: ", x, "y: ", y, "min value: ", min(f_out))