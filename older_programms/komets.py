
import numpy as np
from pylab import *


# Physical values
#M = 1.99e30   # kg
#R = 2e11      # m
#v0mag = 3e4   # m/s
#G = 6.673e-11 # mˆ3\,kgˆ-1 sˆ-2
# Initial conditions
#GM = G*M
#AU = 149597870700 #m
#YR =
GM = 4*np.pi**2 #*AU**3/YR**2
v0mag = np.sqrt(GM/1)#pi/2.

r0 = array([0,1])
v0 = -v0mag*array([1,0])
# Numerical values
#time = 60*60*24*365*5 ## s
time = 1 ## s
dt = 0.001 # s
# Setup Simulation
n = int(ceil(time/dt))+1
r = zeros((n,2),float)
v = zeros((n,2),float)
t = zeros((n,1),float)
r[0] = r0 # vectors
v[0] = v0 # vectors
# Calculation loop
for i in range(n-1):
    rr = norm(r[i,:])
    a = -GM*r[i]/rr**3
    v[i+1] = v[i] + dt*a
    r[i+1] = r[i] + dt*v[i+1]
    t[i+1] = t[i] + dt
plot(r[:,0],r[:,1])
xlabel('x [m]'); ylabel('y [m]')#; axis equal
plot(0,0,'o',color='black', markersize=10)
show()
