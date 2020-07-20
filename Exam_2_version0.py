import random
import numpy as np
from numpy import *
from pylab import *

import matplotlib.pyplot as plt

# Author: Tobias Rudolph
# Date: 17.07.2020

#other constants
T = 365.25*24*3600  # Orbital period [s]
R = 152.10e9        # Aphelion distance [m]
G = 6.673e-11       # Gravitational constant [N (m/kg)^2]
M = 1.9891e30       # Solar mass [kg]
m = 5.9726e24       # Earth mass [kg]

YR = 1 #2*np.pi* np.sqrt(GM)
AU = 1
GM = 4*np.pi**2 *AU**3/YR**2
e = 0.9

r0mag = 1*(1-e)
v0mag =  np.sqrt(GM/1* (1+e)/(1-e))
#v0mag =  3*np.pi/2 * AU/YR

r0 = r0mag*array([1,0])
v0 = v0mag*array([0,1])

time = 10 ## s
dt = 0.0005
#dt = 0.05
# Setup Simulation
n = int(ceil(time/dt)) + 1 #+5000
print(n)

r = zeros((n,2),float)
v = zeros((n,2),float)
t = zeros((n,1),float)

r[0] = r0 # vectors
v[0] = v0 # vectors

C = 1

E = np.zeros(n)  # Total energy/mass
E[0] = 0.5*(norm(v[0])**2) - C/norm(r[0])
E_0 = 0.5*(norm(v[0])**2) - C/norm(r[0])


def acc(r, rr):
    return -GM*r/rr**3

def RungeKutta_secondtry(n,r, v, t, acc, E):
    for i in range(0, n-1):
        rr = norm(r[i,:])
        k1v = acc(r[i], rr)                         #use RK4 method
        k1r = v[i]
        k2v = acc(r[i] + (dt/2) * k1r, rr)
        k2r = v[i] + (dt/2) * k1v
        k3v = acc(r[i] + (dt/2) * k2r, rr)
        k3r =     v[i] + (dt/2) * k2v
        k4v = acc(r[i] +  dt    * k3r, rr)
        k4r = v[i] + dt * k3v

        r[i+1] = r[i] + dt/6.*(k1r + 2*k2r + 2*k3r + k4r)
        v[i+1] = v[i] + dt/6.*(k1v + 2*k2v + 2*k3v + k4v)

        t[i+1] = t[i] + dt
        E[i] = 0.5*(norm(v[i+1])**2) - C/norm(r[i+1])
        #print(E[i])
    return t, r, v, E


def RungeKutta_update(n, r, v):
    for i in range(0, n-1):
        rr = norm(r[i,:])
        k1 = dt*v[i]
        #l1 = dt*getacc(x[i], v[i], t[i])
        l1 = dt*acc(r[i], rr)
        k2 = dt/2.*(v[i] + l1/2.)
        l2 = dt/2.*acc(r[i] + k1/2, rr) #, v[i]+l1/2, t[i]+dt/2)
        k3 = dt/2.*(v[i] + l2/2.)
        l3 = dt/2.*acc(r[i] + k2/2, rr) #, v[i]+l2/2, t[i]+dt/2)
        k4 = dt*(v[i] + l3)
        l4 = dt*acc(r[i] + k3, rr) #, v[i]+l3, t[i]+tau)

        r[i+1] = r[i] + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
        v[i+1] = v[i] + 1/6.*(l1 + 2*l2 + 2*l3 + l4)
        t[i+1] = t[i] + dt
    return t, r, v


def PlotTrajectory(x,y):
    ax = plt.gca()
    ax.cla()
    plt.plot(x, y)
    circle1 = plt.Circle((0, 0), 1, linestyle=":", color='grey', fill=False, alpha= 0.5)
    ax.add_artist(circle1)
    plt.plot(0,0,'o',color='black', markersize=10)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    axs = 2
    #xlim(-axs,axs)
    #ylim(-axs,axs)

def main():


    #update(n,r,v)
    #RungeKutta_update(n,r,v)
    RungeKutta_secondtry(n,r,v,t,acc,E)
    PlotTrajectory(r[:,0],r[:,1])
    #plt.show()
    #print(r)

    if(time==1):
        # Find offset
        print("\nSmall offset indicates closed orbit")
        print("Offset in 'x': %0.3e - %0.9e = %0.7e" % (r[0,0],r[-1,0], r[-1,0]-r[0,0]))
        print("Offset in 'y': %0.3e - %0.9e = %0.7e" % (r[0,1],r[-1,1], r[-1,1]-r[0,1]))
        print("Total offset: %0.3e" % np.sqrt( (r[0,0]-r[-1,0])**2 + (r[0,1]-r[-1,1])**2) )

        # Find perihelion seperation:
        r_perihelion = abs(min(r[:,0]))
        print("\nThe perihelion seperation is %0.3f, compared to 0.967." % r_perihelion)

    draw()

    plt.figure()
    plt.title('Energy per unit mass')

    plt.ylabel("Energy")
    plt.xlabel(r"$n$")
    plt.grid()
    for i,value in enumerate(r):
        E[i] = 0.5*(norm(v[i])**2) - C/norm(value)
    #print(E)
    plot(t,E/E_0)
    show()




main()
