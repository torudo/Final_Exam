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

def getacc_pendulum(x, v, t, Q):
    """
    The function to evaluate, here the driven pendulum
    """
    k=1
    omega = 2/3.
    gamma = 0.5
    return -k*np.sin(x)-gamma*v+Q*np.sin(omega*t)

def help_taucalc(tau, epsilon, delta):
    return tau * (epsilon/delta)**(1./5.)

def RungeKutta(getacc, tau, N, Q, x0, v0):
    """
    This is the actual RungeKutta method as explained in the lecture
    """
    t = np.arange(0,N)*tau
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

    for i in range(0, N-1):
        k1 = tau*v[i]
        l1 = tau*getacc(x[i], v[i], t[i], Q)
        k2 = tau*(v[i]+l1/2.)
        l2 = tau*getacc(x[i]+k1/2, v[i]+l1/2, t[i]+tau/2, Q)
        k3 = tau*(v[i]+l2/2.)
        l3 = tau*getacc(x[i]+k2/2, v[i]+l2/2, t[i]+tau/2, Q)
        k4 = tau*(v[i]+l3)
        l4 = tau*getacc(x[i]+k3, v[i]+l3, t[i]+tau, Q)

        x[i+1] = x[i] + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
        v[i+1] = v[i] + 1/6.*(l1 + 2*l2 + 2*l3 + l4)
    return t, x, v


def adaptive_RungeKutta(getacc,tau, N, Q, x0, v0):
    """
    This is the actual RungeKutta method as explained in the lecture
    """
    t = np.zeros(N)
    x = np.zeros(N)
    v = np.zeros(N)

    x[0] = x0
    v[0] = v0

    print("Wertevergelich\nx_step dh_step_x delta tau\n")
    for i in range(0, N-1):

        k1 = tau*v[i]
        l1 = tau*getacc(x[i], v[i], t[i], Q)

        k2 = tau*(v[i]+l1/2.)
        l2 = tau*getacc(x[i]+k1/2, v[i]+l1/2, t[i]+tau/2, Q)

        k3 = tau*(v[i]+l2/2.)
        l3 = tau*getacc(x[i]+k2/2, v[i]+l2/2, t[i]+tau/2, Q)

        k4 = tau*(v[i]+l3)
        l4 = tau*getacc(x[i]+k3, v[i]+l3, t[i]+tau, Q)

        x_step = x[i] + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
        v_step = v[i] + 1/6.*(l1 + 2*l2 + 2*l3 + l4)
        # k2 = tau*(v[i]+l1/4.)
        # l2 = tau*getacc(x[i]+k1/4, v[i]+l1/4, t[i]+tau/4, Q)
        # k3 = tau*(v[i]+l2/4.)
        # l3 = tau*getacc(x[i]+k2/4, v[i]+l2/4, t[i]+tau/4, Q)
        # k4 = tau*(v[i]+l3/2)
        # l4 = tau*getacc(x[i]+k3/2, v[i]+l3/2, t[i]+tau/2, Q)

        k1 = tau/2.*v[i]
        l1 = tau/2.*getacc(x[i], v[i], t[i], Q)
        k2 = tau/2.*(v[i] + l1/4.)
        l2 = tau/2.*getacc(x[i] + k1/4., v[i]+l1/4., t[i]+tau/4., Q)
        k3 = tau/2.*(v[i] + l2/4.)
        l3 = tau/2.*getacc(x[i] + k2/4., v[i]+l2/4., t[i]+tau/4., Q)
        k4 = tau/2.*(v[i] + l3/2.)
        l4 = tau/2.*getacc(x[i] + k3/2., v[i]+l3/2., t[i]+tau/2., Q)
        half_step_x = x[i] + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
        half_step_v = v[i] + 1/6.*(l1 + 2*l2 + 2*l3 + l4)

        k1 = tau/2.*half_step_v
        l1 = tau/2.*getacc(half_step_x, half_step_v, t[i], Q)
        k2 = tau/2.*(half_step_v+l1/4.)
        l2 = tau/2.*getacc(half_step_x+k1/4., half_step_v+l1/4., t[i]+tau/4., Q)
        k3 = tau/2.*(half_step_v+l2/4.)
        l3 = tau/2.*getacc(half_step_x+k2/4., half_step_v+l2/4., t[i]+tau/4., Q)
        k4 = tau/2.*(half_step_v+l3/2.)
        l4 = tau/2.*getacc(half_step_x + k3/2., half_step_v+l3/2., t[i]+tau/2., Q)
        dh_step_x = half_step_x + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
        dh_step_v = half_step_v + 1/6.*(l1 + 2*l2 + 2*l3 + l4)

        delta = abs(x_step - dh_step_x) /( 0.5*(abs(x_step) + abs(dh_step_x)) )

        print("{0:.5}   {1:.5}  {2:.5}  {3:2.3}".format(x_step,dh_step_x,delta,tau))
        if delta > threshold:
            #decreas tau and do again
            x[i+1] = x_step
            v[i+1] = v_step
            #print("grööößer!")
        elif delta < threshold:
            x[i+1] = dh_step_x
            v[i+1] = dh_step_v
            #print("kllleeeeiner!")
            #increase tau for next timestep
        newtau = help_taucalc(tau, threshold, delta)
        tau = min(max(S_1*newtau, tau/S_2),S_2*tau)
        t[i+1] = t[i] + tau


#        x[i+1] = x[i] + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
#        v[i+1] = v[i] + 1/6.*(l1 + 2*l2 + 2*l3 + l4)


    return t, x, v

#For i)
def Test20(t_start, x0, v0, tau, N, k, Q, gamma, omega):
    N = 4
    t, x, v = adaptive_RungeKutta(getacc_pendulum, tau, N, Q, x0, v0)
    print('\nResults for Q = ' + str(Q) + ' with ' + str(N-1) + ' time steps:')
    print('x[' + str(N-1) + '] = {:.8f}'.format(x[-1]) )
    print('v[' + str(N-1) + '] = {:.8f}'.format(v[-1]) )
    t, x, v = RungeKutta(getacc_pendulum, tau, N, Q, x0, v0)
    print('Results for Q = ' + str(Q) + ' with ' + str(N-1) + ' time steps:')
    print('x[' + str(N-1) + '] = {:.8f}'.format(x[-1]) )
    print('v[' + str(N-1) + '] = {:.8f}'.format(v[-1]) )


#For ii) and iii)
def PlotTrajectory(t_start, x0, v0, tau, N, k, Q, gamma, omega):
    t, x, v = RungeKutta(getacc_pendulum, N, Q, x0, v0)

    x = ((x+np.pi) % (2*np.pi)) - np.pi

    print('t_tot = ' + str(N-1) + '*tau = ' + str(t[-1]))

    plt.plot(t, x)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()


if __name__ == '__main__':

    #start values
    x0 = 1
    v0 = 0 # v0 = dxdt(0)

    k = 1
    omega = 2/3.
    gamma = 0.5

    T0 = 2*np.pi/omega
    tau = T0/200
    print("tau = {}".format(tau))

    Test20(0, x0, v0, tau, 21, k, 0.5, gamma, omega)
#    PlotTrajectory(0, x0, v0, tau, 8001, k, 0.9, gamma, omega)
#    PlotTrajectory(0, x0, v0, tau, 8001, k, 1.1, gamma, omega)
