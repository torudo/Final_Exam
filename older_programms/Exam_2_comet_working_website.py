import random
import numpy as np
from numpy import *
from pylab import *

import matplotlib.pyplot as plt

# Author: Tobias Rudolph
# Date: 17.07.2020


YR = 1
AU = 1
GM = 4*np.pi**2 *AU**3/YR**2

r0mag = 1
v0mag =  np.sqrt(GM/AU)

r0 = r0mag*array([1,0])
v0 = v0mag*array([0,1])

time = 1.54 ## s
dt = 0.0001
# Setup Simulation
n = int(ceil(time/dt)) #+5000
print(n)

r = zeros((n,2),float)
v = zeros((n,2),float)
t = zeros((n,1),float)

E = np.zeros(n)  # Total energy/mass
r[0] = r0 # vectors
v[0] = v0 # vectors


#Set common figure parameters:
newparams = {
    'figure.figsize': (16, 5), 'axes.grid': True,
    'lines.linewidth': 1.5, 'font.size': 19, 'lines.markersize' : 10,
    'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'}
plt.rcParams.update(newparams)

T = 365.25*24*3600  # Orbital period [s]
R = 152.10e9        # Aphelion distance [m]
G = 6.673e-11       # Gravitational constant [N (m/kg)^2]
M = 1.9891e30       # Solar mass [kg]
m = 5.9726e24       # Earth mass [kg]

C = (G*M*T**2)/(R**3)


# Initial conditions
X0 = 1
U0 = 0
Y0 = 0
V0 = 29.29e3*T/R

t_max = 1    # t_max=1 Corresponds to one single complete orbit
dt = 0.0001
N = int(t_max/dt)-1  # Number of time steps




def F(X_, Y_, U_, V_): # dX/dtau
    return U_

def G(X_, Y_, U_, V_): # dY/dtau
    return V_

def H(X_, Y_, U_, V_): # dU/dtau
    return -C * (  X_/( (np.sqrt(X_**2 + Y_**2) )**3)  )

def I(X_, Y_, U_, V_): # dV/dtau
    return -C * (  Y_/( (np.sqrt(X_**2 + Y_**2) )**3)  )

X_4RK = np.zeros(N)
Y_4RK = np.zeros(N)
U_4RK = np.zeros(N)
V_4RK = np.zeros(N)
E_4RK = np.zeros(N-1)  # Total energy/mass

X_4RK[0] = X0
V_4RK[0] = V0

for n in range(N-1):

    k_x1 = dt * F( X_4RK[n], Y_4RK[n], U_4RK[n], V_4RK[n] )
    k_y1 = dt * G( X_4RK[n], Y_4RK[n], U_4RK[n], V_4RK[n] )
    k_u1 = dt * H( X_4RK[n], Y_4RK[n], U_4RK[n], V_4RK[n] )
    k_v1 = dt * I( X_4RK[n], Y_4RK[n], U_4RK[n], V_4RK[n] )

    k_x2 = dt * F( X_4RK[n] + k_x1/2, Y_4RK[n] + k_y1/2, U_4RK[n] + k_u1/2, V_4RK[n] + k_v1/2 )
    k_y2 = dt * G( X_4RK[n] + k_x1/2, Y_4RK[n] + k_y1/2, U_4RK[n] + k_u1/2, V_4RK[n] + k_v1/2 )
    k_u2 = dt * H( X_4RK[n] + k_x1/2, Y_4RK[n] + k_y1/2, U_4RK[n] + k_u1/2, V_4RK[n] + k_v1/2 )
    k_v2 = dt * I( X_4RK[n] + k_x1/2, Y_4RK[n] + k_y1/2, U_4RK[n] + k_u1/2, V_4RK[n] + k_v1/2 )

    k_x3 = dt * F( X_4RK[n] + k_x2/2, Y_4RK[n] + k_y2/2, U_4RK[n] + k_u2/2, V_4RK[n] + k_v2/2 )
    k_y3 = dt * G( X_4RK[n] + k_x2/2, Y_4RK[n] + k_y2/2, U_4RK[n] + k_u2/2, V_4RK[n] + k_v2/2 )
    k_u3 = dt * H( X_4RK[n] + k_x2/2, Y_4RK[n] + k_y2/2, U_4RK[n] + k_u2/2, V_4RK[n] + k_v2/2 )
    k_v3 = dt * I( X_4RK[n] + k_x2/2, Y_4RK[n] + k_y2/2, U_4RK[n] + k_u2/2, V_4RK[n] + k_v2/2 )

    k_x4 = dt * F( X_4RK[n] + k_x3, Y_4RK[n] + k_y3, U_4RK[n] + k_u3, V_4RK[n] + k_v3 )
    k_y4 = dt * G( X_4RK[n] + k_x3, Y_4RK[n] + k_y3, U_4RK[n] + k_u3, V_4RK[n] + k_v3 )
    k_u4 = dt * H( X_4RK[n] + k_x3, Y_4RK[n] + k_y3, U_4RK[n] + k_u3, V_4RK[n] + k_v3 )
    k_v4 = dt * I( X_4RK[n] + k_x3, Y_4RK[n] + k_y3, U_4RK[n] + k_u3, V_4RK[n] + k_v3 )

    X_4RK[n+1] = X_4RK[n] + k_x1/6 + k_x2/3 + k_x3/3 + k_x4/6
    Y_4RK[n+1] = Y_4RK[n] + k_y1/6 + k_y2/3 + k_y3/3 + k_y4/6
    U_4RK[n+1] = U_4RK[n] + k_u1/6 + k_u2/3 + k_u3/3 + k_u4/6
    V_4RK[n+1] = V_4RK[n] + k_v1/6 + k_v2/3 + k_v3/3 + k_v4/6

    E_4RK[n] = 0.5*(U_4RK[n+1]**2+V_4RK[n+1]**2)-C/np.sqrt(X_4RK[n+1]**2 + Y_4RK[n+1]**2)


# If t_max was set to exactly one period, it will be interesting
# to investigate whether or not the orbit is closed (planet returns
# to its starting position)
if(t_max==1):
    # Find offset
    print("\nSmall offset indicates closed orbit")
    print("Offset in 'x': %0.3e - %0.7e = %0.7e" % (X_4RK[0], X_4RK[-1], X_4RK[N-1]-X_4RK[0]))
    print("Offset in 'y': %0.3e - %0.7e = %0.7e" % (Y_4RK[0], Y_4RK[-1], Y_4RK[N-1]-Y_4RK[0]))
    print("Total offset: %0.3e" % np.sqrt((X_4RK[0]-X_4RK[-1])**2+(Y_4RK[0]-Y_4RK[-1])**2))

    # Find perihelion seperation:
    r_perihelion = abs(min(Y_4RK))
    print("\nThe parahelion seperation is %0.3f, compared to 0.967." % r_perihelion)

plt.figure()
plt.title('4th order Runge-Kutta')
plt.plot(X_4RK, Y_4RK, 'g', [0], [0], 'ro')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.grid()

plt.figure()
plt.title('Energy per unit mass')
plt.plot(E_4RK)
plt.ylabel("Energy")
plt.xlabel(r"$n$")
plt.grid()
plt.show()
