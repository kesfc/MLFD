import numpy
def find():
    greek = 0.05
    dvc = 10
    N = 1
    while(True):

        if(0.05 - (8 / N * numpy.log(4 * ((2 * N) ** dvc + 1) / greek)) ** 0.5 >0):
            break
        else:
            N += 1
    return N
print(find())