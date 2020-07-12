# Angular motion of a nonlinear pendulum by the Runge-Kutta method
# u" = -g/l * sin(u) - k * u’, u(0) = u0, u’(0) = u0’
from math import *
from ode import *
import matplotlib.pyplot as plt

#============================================================================
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



g = 9.81e0 # gravitational acceleration
def Func(t, u, f): # RHS of 1st order ODEs
    f[1] = u[2] # u[1] = u, u[2] = u’
    f[2] = -g/l * sin(u[1]) - k * u[2]

# main
l = 1e0 # pendulum length
k = 0e0 # velocity coefficient
u0 = 0.5e0*pi # initial displacement
du0 = 0e0 # initial derivative
tmax = 20e0 # time span
ht = 0.001e0 # time step size
n = 2 # number of 1st order ODEs
u = [0]*(n+1) # solution components
#out = open("pendulum.txt","w") # open output file
#out.write(" t u du\n")
t = 0e0
u[1] = u0; u[2] = du0 # initial values
#out.write(("{0:10.5f}{1:10.5f}{2:10.5f}\n").format(t,u[1],u[2]))
nT = 0 # number of half-periods
t1 = t2 = 0e0 # bounding solution zeros
us = u[1] # save solution

t_plot = []
u1_plot = []
u2_plot = []

while (t+ht <= tmax): # propagation loop
    RungeKutta(t,ht,u,n,Func)
    t += ht
    if (u[1]*us < 0e0): # count solution passages through zero
        if (t1 == 0): t1 = t # initial zero
        else: t2 = t; nT += 1 # final zero
    us = u[1] # save solution
    #out.write(("{0:10.5f}{1:10.5f}{2:10.5f}\n").format(t,u[1],u[2]))
    t_plot.append(t)
    u1_plot.append(u[1])
    u2_plot.append(u[2])

T = 2e0*(t2-t1) / nT # calculated period
T0 = 2e0*pi*sqrt(l/g) # harmonic period
#print("u0 = {0:7.5f} T/T0 = {1:7.5f}".format(u0,T/T0))
#out.close()

plt.plot(t_plot,u1_plot)
plt.plot(t_plot,u2_plot)
plt.show()
