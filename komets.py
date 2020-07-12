# import matplotlib.pyplot as plt
# import random
# import numpy as np
# from matplotlib.pyplot import plot, draw, show
# from scipy.fftpack import fft,ifft
# from scipy.linalg import expm, sinm, cosm
# from scipy import linalg
# from scipy.sparse import diags
# from scipy.sparse import coo_matrix, block_diag
# import csv
# import os
# from math import *
# from cmath import *
# from pde import *

import numpy as np
from pylab import *

def RungeKutta(t, ht, y, n, Func):
    #----------------------------------------------------------------------------
    # Propagates the solution y of a system of 1st order ODEs
    # y’[i] = f[i](t,y[]), i = 1..n
    # from t to t+ht using the 4th order Runge-Kutta method
    # Calls: Func(t, y, f) - RHS of ODEs
    #----------------------------------------------------------------------------
    f1 = [0]*(n+1); f2 = [0]*(n+1) # RHS of ODEs
    f3 = [0]*(n+1); f4 = [0]*(n+1)
    yt = [0]*(n+1) # predicted solution
    ht2 = ht/2e0
    Func(t,y,f1) # RHS at t
    for i in range(1,n+1): yt[i] = y[i] + ht2*f1[i]
    Func(t+ht2,yt,f2) # RHS at t+ht/2
    for i in range(1,n+1): yt[i] = y[i] + ht2*f2[i]
    Func(t+ht2,yt,f3) # RHS at t+ht/2
    for i in range(1,n+1): yt[i] = y[i] + ht *f3[i]
    Func(t+ht,yt,f4) # RHS at t+ht
    h6 = ht/6e0 # propagate solution
    for i in range(1,n+1): y[i] += h6*(f1[i] + 2*(f2[i] + f3[i]) + f4[i])


# Physical values
M = 1.99e30   # kg
R = 2e11      # m
v0mag = 3e4   # m/s
G = 6.673e-11 # mˆ3\,kgˆ-1 sˆ-2
# Initial conditions
r0 = R*array([1,0])

v0 = v0mag*array([0,1])
# Numerical values
time = 60*60*24*365*5 ## s
dt = 100 # s
# Setup Simulation
n = int(ceil(time/dt))
r = zeros((n,2),float)
v = zeros((n,2),float)
t = zeros((n,1),float)
r[0] = r0 # vectors
v[0] = v0 # vectors
GM = G*M
# Calculation loop
for i in range(n-1):
    rr = norm(r[i,:])
    a = -GM*r[i]/rr**3
    v[i+1] = v[i] + dt*a
    r[i+1] = r[i] + dt*v[i+1]
    t[i+1] = t[i] + dt
plot(r[:,0],r[:,1])
xlabel('x [m]'); ylabel('y [m]')#; axis equal
show()
