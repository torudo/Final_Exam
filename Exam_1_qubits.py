import random
import numpy as np
import matplotlib.pyplot as plt


# Author: Tobias Rudolph
# Date: 10.07.2020

dim = 6

# null    = np.array([1,0] , dtype=float)
# one     = np.array([0,1] , dtype=float)
# print(null.T,one.transpose())

null = np.zeros(dim, dtype=float)
null[0] =1
one = np.zeros(dim, dtype=float)
one[1] = 1
#print(null,one)


#a = np.diag(np.sqrt(np.linspace(0, dim+1, dim)))
a = np.diag(np.linspace(0, dim-1, dim))
a = np.sqrt(a)
a = np.vstack((a[1:], a[0]))
#switchs lines
#a[[-1,0]] = a[[0,-1]]

print(a)
print("\n")


# print(np.arange(3))
# print(np.linspace(0,3,4))


#a = np.delete(a, (0), axis=0)
#print(a)
