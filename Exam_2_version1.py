import random
import numpy as np
from numpy import *
from pylab import *

import matplotlib.pyplot as plt

#Set common figure parameters:
#newparams = {
#     'figure.figsize': (5, 5)}#, 'axes.grid': True,
#     'lines.linewidth': 1.5, 'font.size': 19, 'lines.markersize' : 10,
#     'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'}
#plt.rcParams.update(newparams)


# Author: Tobias Rudolph
# Date: 17.07.2020

#Global Constants:
T = 365.25*24*3600  # Orbital period [s]
R = 152.10e9        # Aphelion distance [m]
G = 6.673e-11       # Gravitational constant [N (m/kg)^2]
M = 1.9891e30       # Solar mass [kg]
m = 5.9726e24       # Earth mass [kg]


YR = 1 #2*np.pi* np.sqrt(GM)
AU = 1
GM = 4*np.pi**2 *AU**3/YR**2

threshold = 1.e-3 #0.001
#threshold = 1.e-7 #0.001
S1 = 0.9
S2 = 1.3


## Zwischenspeicher version
#def RK4_singlestep2(X0, V0, t, dt, rhs):
    # x = X0[0]
    # y = X0[1]
    # u = V0[0]
    # v = V0[1]
    #
    # # get the RHS at several points
    # k1x, k1y, l1u, l1v = rhs([x,y], [u,v])
    #
    # k2x, k2y, l2u, l2v = \
    #     rhs([x+0.5*dt*k1x,y+0.5*dt*k1y],
    #         [u+0.5*dt*l1u,v+0.5*dt*l1v])
    #
    # k3x, k3y, l3u, l3v = \
    #     rhs([x+0.5*dt*k2x,y+0.5*dt*k2y],
    #         [u+0.5*dt*l2u,v+0.5*dt*l2v])
    #
    # k4x, k4y, l4u, l4v = \
    #     rhs([x+dt*k3x,y+dt*k3y],
    #         [u+dt*l3u,v+dt*l3v])
    #
    # # advance
    # unew = u + (dt/6.0)*(l1u + 2.0*l2u + 2.0*l3u + l4u)
    # vnew = v + (dt/6.0)*(l1v + 2.0*l2v + 2.0*l3v + l4v)
    #
    # xnew = x + (dt/6.0)*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    # ynew = y + (dt/6.0)*(k1y + 2.0*k2y + 2.0*k3y + k4y)
    #
    # return [xnew, ynew], [unew, vnew], 1, 1
## Am liebsten in Vektor schreibweise umändern
#def RK4_singlestep(X0, V0, t, dt, rhs):
    # x = X0[0]
    # y = X0[1]
    # u = V0[0]
    # v = V0[1]
    #
    # k1x, k1y, l1u, l1v = rhs([x,y], [u,v])
    #
    # k2x, k2y, l2u, l2v = \
    #     rhs([x+0.5*dt*k1x,y+0.5*dt*k1y],
    #         [u+0.5*dt*l1u,v+0.5*dt*l1v])
    #
    # k3x, k3y, l3u, l3v = \
    #     rhs([x+0.5*dt*k2x,y+0.5*dt*k2y],
    #         [u+0.5*dt*l2u,v+0.5*dt*l2v])
    #
    # k4x, k4y, l4u, l4v = \
    #     rhs([x+dt*k3x,y+dt*k3y],
    #         [u+dt*l3u,v+dt*l3v])
    # unew = u + (dt/6.0)*(l1u + 2.0*l2u + 2.0*l3u + l4u)
    # vnew = v + (dt/6.0)*(l1v + 2.0*l2v + 2.0*l3v + l4v)
    # xnew = x + (dt/6.0)*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    # ynew = y + (dt/6.0)*(k1y + 2.0*k2y + 2.0*k3y + k4y)
    # return [xnew, ynew], [unew, vnew], 1

# def help_taucalc(tau, epsilon, delta):
#     return tau * (epsilon/delta)**(1./5.)

class Orbit(object):
    """ Orbit object which holds the information about:
        r = position
        v = velocitiy
    """

    def __init__(self, r0 = 1, v0 = np.sqrt(GM), tmax = 1, tau = 0.1):
        self.tmax = tmax
        self.dt = tau
        self.err = threshold
        # For the RK4 without adaptive steps
        self.n = int(ceil(self.tmax/self.dt)) + 1 #+5000
        #print("Number of Timesteps: %d \n" %(self.n))

        self.r = np.array([np.zeros(2)])
        self.v = np.array([np.zeros(2)])
        self.t = np.array([np.zeros(1)])

        # Initialisation
        self.r[0] = r0
        self.v[0] = v0

        # container for SMaxis and eccentricity
        self.a = 0
        self.e = 0

    def acc_vec(self, r):
        rr = math.sqrt(r[0]**2 + r[1]**2)
        return -GM*r / rr**3

    def RKStep_vec(self, r0, v0, t, dt, acc):
        r = r0
        v = v0

        k1 = dt*v
        l1 = dt*self.acc_vec(r)

        k2 = dt*(v + l1/2.)
        l2 = dt*self.acc_vec(r + k1/2.)

        k3 = dt*(v + l2/2.)
        l3 = dt*self.acc_vec(r + k2/2.)

        k4 = dt*(v + l3)
        l4 = dt*self.acc_vec(r + k3)

        r_new = r + 1/6.*(k1 + 2*k2 + 2*k3 + k4)
        v_new = v + 1/6.*(l1 + 2*l2 + 2*l3 + l4)

        t_new = t + dt
        return r_new, v_new, t_new


# NOt adaptive version of the RK
    def RungeKutta(self, n, r, v, t, dt, acc, E):
        #t = 0.0
        #ResultsStep = []
        for i in range(0, n-1):
            #print(i)

            #r[i+1], v[i+1], t[i+1], E[i] = self.RKStep(i, n, r[i], v[i], t[i], dt, acc, E[i])

            ResultsStep = self.RKStep(self.r[i], self.v[i], self.t[i], dt, acc, E)
            self.r = np.append(self.r , [ResultsStep[0]] ,axis= 0)
            self.v = np.append(self.v , [ResultsStep[1]] ,axis= 0)
            self.t = np.append(self.t , [ResultsStep[2]] ,axis= 0)
            self.E = np.append(E , ResultsStep[3])
        return r, v, t, E

    def RungeKutta_adaptive(self, dt, err, tmax):
        r = self.r[0]
        v = self.v[0]
        t = 0.0
        dt_new = dt
        n_reset = 0

        while t < tmax:
            if self.err > 0.0:
                rel_error = 1.e10
                n_try = 0
                while rel_error > self.err:
                    dt = dt_new
                    if t+dt > tmax:
                        dt = tmax-t
                    # Running two half steps
                    rtemp, vtemp, ttemp =\
                        self.RKStep_vec(r, v, t, 0.5*dt, self.acc_vec)
                    rnew, vnew, tnew =\
                        self.RKStep_vec(rtemp, vtemp, ttemp, 0.5*dt, self.acc_vec)
                    # single step to compare with
                    rsingstep, vsingstep, tsingstep =\
                        self.RKStep_vec(r, v, t, dt, self.acc_vec)
                    # New error
                    rel_error = max(np.abs((rnew[0]-rsingstep[0])/rnew[0]),
                                    np.abs((rnew[1]-rsingstep[1])/rnew[1]),
                                    np.abs((vnew[0]-vsingstep[0])/vnew[0]),
                                                np.abs((vnew[1]-vsingstep[1])/vnew[1]))

                    dt_est = dt*abs(self.err/rel_error)**(1./5.)
                    dt_new = min(max(S1*dt_est, dt/S2), S2*dt)
                    n_try += 1
                if n_try > 1:
                    # n_try = 1 if we took only a single try at the step
                    n_reset += (n_try-1)
            else:
                if t + dt > tmax:
                    dt = tmax-t
                rnew, vnew, tnew =\
                    self.RKStep_vec(r, v, t, dt, self.acc)
            # successful step
            t += dt

            self.r = np.append(self.r ,[rnew] ,axis= 0)
            self.v = np.append(self.v ,[vnew] ,axis= 0)
            self.t = np.append(self.t ,t)

            r = rnew; v = vnew
        print("resets",n_reset)
        return r, v, t


####################################
# Plotting

    def PlotSystem(self,x=0,y=0):
        ax = plt.gca()
        ax.cla()
        circle1 = plt.Circle((0, 0), 1, linestyle=":", color='grey', fill=False, alpha= 0.5)
        ax.add_artist(circle1)
        plt.plot(0,0,'o',color='black', markersize=10)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        axs = 2
        #xlim(-axs,axs)
        #ylim(-axs,axs)
        plt.grid(linestyle=':',alpha=0.2)

        ax.minorticks_on()
        ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
        ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)

        #filename = "Pulse_first_case{0:.3}.pdf".format(omega)
        #plt.savefig(filename, bbox_inches='tight')
        plt.draw()

    def PlotOrbit(self,x,y):
        plt.plot(x, y, '.', linewidth=0, markersize=1)
        plt.plot(x, y)
        plt.plot(x[-1],y[-1],'o',color='blue', markersize=3)
        #plt.plot(x[2299],y[2299],'o',color='blue', markersize=3)

    def EnergyPlot(self):
        self.E = np.ones(len(self.r))
        for i in range(0,len(self.r)):
            self.E[i] = 0.5*(norm(self.v[i])**2) - GM/norm(self.r[i])

        plt.figure()
        plt.title('Energy per unit mass')
        plt.ylabel(r"E / $\mathrm{E}_0$")
        plt.xlabel(r"$t$")
        plt.grid()
        # for i,value in enumerate(r):
        #     E[i] = 0.5*(norm(v[i])**2) - C/norm(value)
        plt.plot(self.t, self.E/self.E[0])
        #plt.plot(self.t, self.E)
        #plt.plot(np.linspace(0,len(self.r)-1,len(self.r)), self.E, '.')

        #plt.ylim(0.999999,1.000002)
        plt.draw()

    def CalcExcentricity(self, r):
        """Calculates Eccentricitie follwing the geometric properties """
        norm_list = []
        for i in range(len(self.r)):
            norm_list.append(norm(self.r[i]))
        max_r = max(norm_list)
        min_r = min(norm_list)
        e = (max_r- min_r)/(max_r + min_r)
        print("Eccentricity ala rmaxmin: {0:2.3f}".format(e))
        a = (abs(max_r)+abs(min_r))/2.
        print("A = {0:1.8f}".format(a) )
        return a

    def Calc_Extentricity_Vector(self, r0 , v0):
        # appending a 3. dimension for the cross product
        r = np.append(r0,0)
        v = np.append(v0,0)
        L = np.cross(r,v)
        e_vec = (np.cross(v, L))/GM - r/ norm(r)
        e = norm(e_vec)
        print("Another way e = {0:.3f}".format(e))


    def semimajor_axis(self):
        x_min = min(self.r[:,0])
        x_max = max(self.r[:,0])
        a = (abs(x_max)+abs(x_min))/2.
        print("Semimajor axis",a)
        return a

    def displacement(self, r):
        #indexx = abs(r[1: ,1] - 0.).argmin()
        shift = 1
        indexx = abs(r[shift: ,1] - 0.).argmin() + shift
        #print("Das Listenelement mit der geringsten Abweichung ist:",r[:,1].flatten()[indexx],"Listenindex:",indexx)
        print("\nSmall offset indicates closed orbit")
        print("Offset in 'x': %0.3e - %0.9e = %0.7e" % (r[0,0],r[indexx,0], r[indexx,0]-r[0,0]))
        print("Offset in 'y': %0.3e - %0.9e = %0.7e" % (r[0,1],r[indexx,1], r[indexx,1]-r[0,1]))
        print("Total offset: %0.3e" % np.sqrt( (r[0,0]-r[indexx,0])**2 + (r[0,1]-r[indexx,1])**2) )
        #r_perihelion = abs(min(r[:,0]))
        #print("\nThe perihelion seperation is %0.3f, compared to 0.967." % r_perihelion)

# These are stil todo:
    def timestepPlot():
        plt.figure()
        plt.title('dt geplottet')
        plt.ylabel(r"dt")
        plt.xlabel(r"$t$")
        plt.grid()
        # for i,value in enumerate(r):
        #     E[i] = 0.5*(norm(v[i])**2) - C/norm(value)
        #plt.scatter(self.t, self.E/self.E[0])
        show()


    def Output():
        pass


################################################################################
################################# End of Class #################################
################################################################################


def KeplersThird_Problemplot(a,T):
    ax = plt.gca()
    ax.cla()
    plt.plot(np.linspace(0,10,10),np.linspace(0,10,10)*GM/(4*np.pi**2), alpha = 0.7)
    plt.scatter(a,T, color = 'black')
    plt.xlim(0.13,0.19)
    plt.ylim(0.13,0.19)

    plt.xlabel(r'$A$')
    plt.ylabel(r'$T$')
    plt.grid(linestyle=':',alpha=0.2)

    ax.minorticks_on()
    ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
    ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)

    plt.show()

def main():
    params = {
        'test' : 2,
        'testa' : 100
    }
    ############### First Orbit
    v0mag = np.pi/2 * AU/YR
    r0mag = 1
    #v0mag =  np.sqrt(GM/1* (1+e)/(1-e))
    r0 = r0mag*array([1,0])
    v0 = v0mag*array([0,1])
    #t = np.sqrt((0.5161)**3)
    t = 0.37116165063121087
    #t = 0.8
    newOrbit = Orbit(r0, v0, t, 0.01)
    newOrbit.RungeKutta_adaptive(newOrbit.dt, newOrbit.err, newOrbit.tmax)
    #E = 0
    #(self, n, r, v, t, dt, acc, E):
    #newOrbit.RungeKutta(2001, newOrbit.r, newOrbit.v,newOrbit.tmax, newOrbit.dt, acc, E)
    #print("nr of steps",len(newOrbit.r)-1)


    ############### Second Orbit
    v0mag = 1.3*np.pi/2 * AU/YR
    r0mag = 1
    #v0mag =  np.sqrt(GM/1* (1+e)/(1-e))
    r0 = r0mag*array([1,0])
    v0 = v0mag*array([0,1])
    #t = np.sqrt((0.525)**3)
    t = 0.3831331692562119
    #t = 20
    newOrbit2 = Orbit(r0, v0, t)
    newOrbit2.RungeKutta_adaptive(newOrbit2.dt, newOrbit2.err, newOrbit2.tmax)
    #print("nr of steps",len(newOrbit.r)-1)


    ############### First Orbit
    v0mag = 2.*np.pi/2 * AU/YR
    r0mag = 1
    #v0mag =  np.sqrt(GM/1* (1+e)/(1-e))
    r0 = r0mag*array([1,0])
    v0 = v0mag*array([0,1])
    #t = np.sqrt((0.572)**3)
    t = 0.4318659210832704
    #t = 20
    newOrbit3 = Orbit(r0, v0, t)
    newOrbit3.RungeKutta_adaptive(newOrbit3.dt, newOrbit3.err, newOrbit3.tmax)
    #print("nr of steps",len(newOrbit.r)-1)


    '''    # Controll  Orbit
    e = 0.75
    e = 0
    r0mag = 1*(1+e)
    v0mag =  np.sqrt(GM/1* (1-e)/(1+e))
    r0 = r0mag*array([1,0])
    v0 = v0mag*array([0,1])
    newOrbit_Ellips = Orbit(r0, v0, 1)
    newOrbit_Ellips.RungeKutta_adaptive(newOrbit_Ellips.dt, newOrbit_Ellips.err, newOrbit_Ellips.tmax)
    '''

    newOrbit.PlotSystem()
    newOrbit.PlotOrbit(newOrbit.r[:,0],newOrbit.r[:,1])
    newOrbit2.PlotOrbit(newOrbit2.r[:,0],newOrbit2.r[:,1])
    newOrbit3.PlotOrbit(newOrbit3.r[:,0],newOrbit3.r[:,1])
    #newOrbit_Ellips.PlotOrbit(newOrbit_Ellips.r[:,0],newOrbit_Ellips.r[:,1])

    plt.show()


    # For the Excentisity !
    newOrbit2.Calc_Extentricity_Vector(newOrbit.r[-2],newOrbit.v[-2])
    newOrbit.Calc_Extentricity_Vector(newOrbit2.r[-2],newOrbit2.v[-2])
    newOrbit3.Calc_Extentricity_Vector(newOrbit3.r[-2],newOrbit3.v[-2])
    #newOrbit.displacement(newOrbit.r)
    newOrbit.CalcExcentricity(newOrbit.r)
    newOrbit2.CalcExcentricity(newOrbit2.r)
    newOrbit3.CalcExcentricity(newOrbit3.r)


# To create the KEpler plot!!
    a0 = newOrbit.semimajor_axis()
    a1 = newOrbit2.semimajor_axis()
    a2 = newOrbit3.semimajor_axis()
    print("AAA",a0)
    #aas = np.array([0.51646,0.52,0.57])**3
    aas = np.array([a0, a1, a2])**3
    print(np.sqrt((a0)**3),np.sqrt((a1)**3),np.sqrt((a2)**3))
    Tts = np.array([0.37116165063121087, 0.3831331692562119, 0.4318659210832704])**2
    #KeplersThird_Problemplot(aas, Tts)
###########################################################################################
    #Energie
    #newOrbit.EnergyPlot()
    #newOrbit2.EnergyPlot()
    #plt.show()

main()
