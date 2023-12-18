import numpy as np
import random
import matplotlib.pyplot as plt
def test(n):
    L = []
    for i in range(n):
        count = 0
        for j in range(10):
            if(random.randint(0,1) == 1):
                count += 1
        L.append(count/10)
    c1 = L[0]
    cr = L[random.randint(0,n-1)]
    cm = min(L)
    return (c1, cr, cm)

def test_m():
    v1 = []
    vr = []
    vm = []
    n = 10000
    for i in range(n):
        (c1, cr, cm) = test(1000)
        v1.append(c1)
        vr.append(cr)
        vm.append(cm)
    e = np.arange(0,1,0.01)
    hoeffding_bound = [0.25*np.exp(-20*((x-0.55)**2)) for x in e]
    weights = np.ones_like(v1) / float(len(v1))
    plt.hist(v1, weights=weights, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.show()
    plt.hist(v1, weights=weights, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.plot(e,hoeffding_bound, color='red')
    plt.show()
    
    weights = np.ones_like(vr)/float(len(vr))
    plt.hist(vr, weights=weights, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.show()
    plt.hist(vr, weights=weights, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.plot(e,hoeffding_bound, color='red')
    plt.show()
    
    hoeffding_bound = [0.6 * np.exp(-20 * ((x - 0.55) ** 2)) for x in e]
    weights = np.ones_like(vm)/float(len(vm))
    plt.hist(vm, weights=weights, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.show()
    plt.hist(vm, weights=weights, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    plt.plot(e,hoeffding_bound, color='red')
    plt.show()
if __name__ == '__main__':
    test_m()