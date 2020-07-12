import random
import numpy as np
import matplotlib.pyplot as plt

# Author: Tobias Rudolph
# Date: 10.07.2020

#Global variables:
AU = 1
YR = 1
GM = 4*np.pi**2 *AU**3/YR**2
v0 = np.pi/2 *AU/YR #tangentialgesch.

tau = 0.1* YR
threshold = 0.001
S_1 = 0.9
S_2 = 1.3

#For ii) and iii)
def PlotTrajectory(t_start, x0, v0, tau, N, k, Q, gamma, omega):
    t, x, v = RungeKutta(getacc_pendulum, N, Q, x0, v0)

    x = ((x+np.pi) % (2*np.pi)) - np.pi

    print('t_tot = ' + str(N-1) + '*tau = ' + str(t[-1]))

    plt.plot(t, x)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()
