import random
import numpy as np
from numpy import *
from pylab import *

import matplotlib.pyplot as plt

#Set common figure parameters:
newparams = {
     'figure.figsize': (5, 5)}#, 'axes.grid': True,
#     'lines.linewidth': 1.5, 'font.size': 19, 'lines.markersize' : 10,
#     'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'}
plt.rcParams.update(newparams)


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

threshold = 1.e-5 #0.00001
S1 = 0.9
S2 = 1.3

#Alte version vom Zettel
def RKStep(i, n, r, v, t, dt, acc, E):
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
    #delta = abs(x_step - dh_step_x) /( 0.5*(abs(x_step) + abs(dh_step_x)) )
    epsilon_rel = fabs( )
    return t, r, v, E
# Zwischenspeicher version
def RK4_singlestep2(X0, V0, t, dt, rhs):
    x = X0[0]
    y = X0[1]
    u = V0[0]
    v = V0[1]

    # get the RHS at several points
    k1x, k1y, l1u, l1v = rhs([x,y], [u,v])

    k2x, k2y, l2u, l2v = \
        rhs([x+0.5*dt*k1x,y+0.5*dt*k1y],
            [u+0.5*dt*l1u,v+0.5*dt*l1v])

    k3x, k3y, l3u, l3v = \
        rhs([x+0.5*dt*k2x,y+0.5*dt*k2y],
            [u+0.5*dt*l2u,v+0.5*dt*l2v])

    k4x, k4y, l4u, l4v = \
        rhs([x+dt*k3x,y+dt*k3y],
            [u+dt*l3u,v+dt*l3v])

    # advance
    unew = u + (dt/6.0)*(l1u + 2.0*l2u + 2.0*l3u + l4u)
    vnew = v + (dt/6.0)*(l1v + 2.0*l2v + 2.0*l3v + l4v)

    xnew = x + (dt/6.0)*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    ynew = y + (dt/6.0)*(k1y + 2.0*k2y + 2.0*k3y + k4y)

    return [xnew, ynew], [unew, vnew], 1, 1

# Am liebsten in Vektor schreibweise umändern
def RK4_singlestep(X0, V0, t, dt, rhs):
    x = X0[0]
    y = X0[1]
    u = V0[0]
    v = V0[1]

    k1x, k1y, l1u, l1v = rhs([x,y], [u,v])

    k2x, k2y, l2u, l2v = \
        rhs([x+0.5*dt*k1x,y+0.5*dt*k1y],
            [u+0.5*dt*l1u,v+0.5*dt*l1v])

    k3x, k3y, l3u, l3v = \
        rhs([x+0.5*dt*k2x,y+0.5*dt*k2y],
            [u+0.5*dt*l2u,v+0.5*dt*l2v])

    k4x, k4y, l4u, l4v = \
        rhs([x+dt*k3x,y+dt*k3y],
            [u+dt*l3u,v+dt*l3v])
    unew = u + (dt/6.0)*(l1u + 2.0*l2u + 2.0*l3u + l4u)
    vnew = v + (dt/6.0)*(l1v + 2.0*l2v + 2.0*l3v + l4v)
    xnew = x + (dt/6.0)*(k1x + 2.0*k2x + 2.0*k3x + k4x)
    ynew = y + (dt/6.0)*(k1y + 2.0*k2y + 2.0*k3y + k4y)
    return [xnew, ynew], [unew, vnew], 1, 1


def help_taucalc(tau, epsilon, delta):
    return tau * (epsilon/delta)**(1./5.)

# def acc(r, rr):
#     return -GM*r / rr**3


class Orbit(object):
    """ Orbit object which holds the information about:
        r = position
        v = velocitiy
        time =
    """
    def __init__(self, r0 = 1, v0 = np.sqrt(GM), tmax = 0.37, tau = 0.01):
        self.tmax = tmax
        self.dt = tau
        self.err = threshold
        # For the RK4 without adaptive steps
        self.n = int(ceil(self.tmax/self.dt)) + 1 #+5000
        #print("Number of Timesteps: %d \n" %(self.n))

        self.r = np.array([np.zeros(2)])
        self.v = np.array([np.zeros(2)])
        self.t = np.array([np.zeros(1)])
        self.E = np.zeros(1)

        self.r[0] = r0 # vectors
        self.v[0] = v0 # vectors

        self.E[0] = 0.5*(norm(self.v[0])**2) - GM/norm(self.r[0])

    def rhs(self, X, V):
        # current radius
        r = math.sqrt(X[0]**2 + X[1]**2)
        #print(r)
        # position
        xdot = V[0]
        ydot = V[1]

        # velocity
        udot = -GM*X[0]/r**3
        vdot = -GM*X[1]/r**3

        return xdot, ydot, udot, vdot



    def RKStep2(self, r, v, t, dt, acc, E):
        #rr = norm(r)
        rr = math.sqrt(r[0]**2 + r[1]**2)
        #print(rr)
        k1v = acc(r, rr)                         #use RK4 method
        k1r = v

        k2v = acc(r + (dt/2) * k1r, rr)
        k2r = v + (dt/2) * k1v

        k3v = acc(r + (dt/2) * k2r, rr)
        k3r = v + (dt/2) * k2v

        k4v = acc(r +  dt    * k3r, rr)
        k4r = v + dt * k3v

        r_new = r + dt/6.*(k1r + 2*k2r + 2*k3r + k4r)
        v_new = v + dt/6.*(k1v + 2*k2v + 2*k3v + k4v)

        t_new = t + dt
        E_new = 0.5*(norm(v_new)**2) - GM/norm(r_new)

        #print(E[i])
        #delta = abs(x_step - dh_step_x) /( 0.5*(abs(x_step) + abs(dh_step_x)) )
    #    epsilon_rel = fabs( )
        return r_new, v_new, t_new, E_new


    def RKStep(self, r, v, t, dt, acc, E):
        #rr = norm(r)
        rr = math.sqrt(r[0]**2 + r[1]**2)
        #print(rr)
        x = r[0]
        y = r[1]
        u = v[0]
        v = v[1]

        k1x = dt*u
        k1y = dt*v
        l1u = dt*acc(x, rr)                         #use RK4 method
        l1v = dt*acc(y, rr)                         #use RK4 method

        k2x = dt*(u + l1u/2.)
        k2y = dt*(v + l1v/2.)
        l2u = dt*acc(x + k1x/2., rr)
        l2v = dt*acc(y + k1y/2., rr)

        k3x = dt*(u + l2u/2.)
        k3y = dt*(v + l2v/2.)
        l3u = dt*acc(x + k2x/2., rr)
        l3v = dt*acc(y + k2y/2., rr)

        k4x = dt*(u + l3u)
        k4y = dt*(v + l3v)
        l4u = dt*acc(x + k3x, rr)
        l4v = dt*acc(y + k3y, rr)


        x_new = x + 1/6.*(k1x + 2*k2x + 2*k3x + k4x)
        y_new = y + 1/6.*(k1y + 2*k2y + 2*k3y + k4y)
        u_new = u + 1/6.*(l1u + 2*l2u + 2*l3u + l4u)
        v_new = v + 1/6.*(l1v + 2*l2v + 2*l3v + l4v)

        t_new = t + dt
        E_new = 0#0.5*(norm(u_new)**2) - GM/norm(x_new)


        #print(E[i])
        #delta = abs(x_step - dh_step_x) /( 0.5*(abs(x_step) + abs(dh_step_x)) )
    #    epsilon_rel = fabs( )
        #return r_new, v_new, t_new, E_new
        return [x_new,y_new], [u_new,v_new], t_new, E_new




    def RungeKutta(self, n, r, v, t, dt, acc, E):
        #t = 0.0
        #ResultsStep = []
        for i in range(0, n-1):
            #print(i)

            #r[i+1], v[i+1], t[i+1], E[i] = self.RKStep(i, n, r[i], v[i], t[i], dt, acc, E[i])

            ResultsStep = self.RKStep(self.r[i], self.v[i], self.t[i], dt, acc, self.E[i])
            self.r = np.append(self.r , [ResultsStep[0]] ,axis= 0)
            self.v = np.append(self.v , [ResultsStep[1]] ,axis= 0)
            self.t = np.append(self.t , [ResultsStep[2]] ,axis= 0)
            self.E = np.append(self.E , ResultsStep[3])
        return r, v, t, E


    def RungeKutta_adaptive(self, dt, err, tmax):
        r = self.r[0]
        v = self.v[0]
        t = 0.0
        E = self.E[0]

        err = threshold
        dt_new = dt
        n_reset = 0

        while t < tmax:
            #print(t)
            #print(dt)
            if err > 0.0:
                rel_error = 1.e10
                n_try = 0
                while rel_error > err:
                    dt = dt_new
                    #print(dt)
                    #print(err)
                    #print(rel_error)
                    if t+dt > tmax:
                        dt = tmax-t
                    #print("R",r,v)
                    rtemp, vtemp, ttemp, Etemp =\
                        RK4_singlestep(r, v, t, 0.5*dt, self.rhs)#, E)
                        #self.RKStep(r, v, t, 0.5*dt, acc, E)
                    rnew, vnew, tnew, Enew =\
                        RK4_singlestep(rtemp, vtemp, ttemp, 0.5*dt, self.rhs)#, Etemp)
                        #self.RKStep(rtemp, vtemp, ttemp, 0.5*dt, acc, Etemp)

                    rsingstep, vsingstep, tsingstep, Esingstep =\
                        RK4_singlestep(r, v, t, dt, self.rhs)#, E)
                        #self.RKStep(r, v, t, dt, acc, E)
                    #print(rnew-rsingstep)
                    # Checken ob max auch beim arry funktioniert
                    # also dimensionen checken, sonst r[][0] etc..
                    #print(np.abs((rnew-rsingstep)/rnew))

                    rel_error = max(np.abs((rnew[0]-rsingstep[0])/rnew[0]),
                                    np.abs((rnew[1]-rsingstep[1])/rnew[1]),
                                    np.abs((vnew[0]-vsingstep[0])/vnew[0]),
                                    np.abs((vnew[1]-vsingstep[1])/vnew[1]))
                    #print(rel_error)
                    #                abs((unew-rsingstep)/unew),
                    #                abs((vnew-rsingstep)/vnew))
                    #print(rnew)
                    dt_est = dt*abs(err/rel_error)**0.2
                    dt_new = min(max(S1*dt_est, dt/S2), S2*dt)
                    n_try += 1
                if n_try > 1:
                    # n_try = 1 if we took only a single try at the step
                    n_reset += (n_try-1)
            else:
                if t + dt > tmax:
                    dt = tmax-t
                rnew, vnew, tnew, E_new =\
                    RK4_singlestep(r, v, t, dt, self.rhs)#, E)
                    #self.RKStep(r, v, t, dt, acc, E)
            # successful step
            t += dt

            self.r = np.append(self.r ,[rnew] ,axis= 0)
            self.v = np.append(self.v ,[vnew] ,axis= 0)
            self.t = np.append(self.t ,t)
            self.E = np.append(self.E ,Enew )

            #r[i+1], v[i+1], t[i+1], E[i] = self.RKStep(i, n, r[i], v[i], t[i], dt, acc, E[i])
            #RKStep(i,n,r, v, t, dt/2, acc, E)
            #RKStep(i,n,r, v, t, dt/2, acc, E)
            r = rnew; v = vnew
        print("resets",n_reset)
        return r, v, t, E



    def RungeKutta_adaptive_2(self, dt, err, tmax):
        r = self.r[0]
        v = self.v[0]
        t = 0.0
        E = self.E[0]

        err = threshold
        dt_new = dt
        n_reset = 0

        while t < tmax:
            #print(t)
            #print(dt)
            if err > 0.0:
                rel_error = 1.e10
                n_try = 0
                while rel_error > err:
                    dt = dt_new
                    #print(dt)
                    #print(err)
                    #print(rel_error)
                    if t+dt > tmax:
                        dt = tmax-t
                    #print("R",r,v)
                    rtemp, vtemp, ttemp, Etemp =\
                        self.RKStep(r, v, t, 0.5*dt, acc, E)
                    print("test",rtemp)
                    #print("test",rtemp)
                    rnew, vnew, tnew, Enew =\
                        self.RKStep(rtemp, vtemp, ttemp, 0.5*dt, acc, Etemp)

                    print("NEWest",rnew)
                    rsingstep, vsingstep, tsingstep, Esingstep =\
                        self.RKStep(r, v, t, dt, acc, E)
                    #print(rnew-rsingstep)
                    # Checken ob max auch beim arry funktioniert
                    # also dimensionen checken, sonst r[][0] etc..
                    #print(np.abs((rnew-rsingstep)/rnew))

                    rel_error = max(np.abs((rnew[0]-rsingstep[0])/rnew[0]),
                                    np.abs((rnew[1]-rsingstep[1])/rnew[1]),
                                    np.abs((vnew[0]-vsingstep[0])/vnew[0]),
                                    np.abs((vnew[1]-vsingstep[1])/vnew[1]))
                    #print(rel_error)
                    #                abs((unew-rsingstep)/unew),
                    #                abs((vnew-rsingstep)/vnew))
                    #print(rnew)
                    dt_est = dt*abs(err/rel_error)**0.2
                    dt_new = min(max(S1*dt_est, dt/S2), S2*dt)
                    n_try += 1
                if n_try > 1:
                    # n_try = 1 if we took only a single try at the step
                    n_reset += (n_try-1)
            else:
                if t + dt > tmax:
                    dt = tmax-t
                rnew, vnew, tnew, E_new =\
                    self.RKStep(r, v, t, dt, acc, E)
            # successful step
            t += dt

            self.r = np.append(self.r ,[rnew] ,axis= 0)
            self.v = np.append(self.v ,[vnew] ,axis= 0)
            self.t = np.append(self.t ,t)
            self.E = np.append(self.E ,Enew )

            #r[i+1], v[i+1], t[i+1], E[i] = self.RKStep(i, n, r[i], v[i], t[i], dt, acc, E[i])
            #RKStep(i,n,r, v, t, dt/2, acc, E)
            #RKStep(i,n,r, v, t, dt/2, acc, E)
            r = rnew; v = vnew
        print("resets",n_reset)
        return r, v, t, E



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
        plt.figure()
        plt.title('Energy per unit mass')
        plt.ylabel(r"E / $\mathrm{E}_0$")
        plt.xlabel(r"$t$")
        plt.grid()
        # for i,value in enumerate(r):
        #     E[i] = 0.5*(norm(v[i])**2) - C/norm(value)
        plot(self.t, self.E/self.E[0])
        show()

    def CalcExcentricity(self, r):
        #x_max = max(self.r[:,0])
        #y_max = max(self.r[:,1])
        #x_min = min(self.r[:,0])
        #y_min = min(self.r[:,1])
        #print(  x_max,
        #        y_max,
        #        x_min,
        #        y_min)
        #e = (x_max-x_min)/(x_max+x_min)
        #print("eccentricity is: {0:2.2f}".format(e))
        #e = np.sqrt(1 - y_max**2/x_max**2)
        #print("eccentricity is: {0:2.2f}".format(e))
        norm_list = []
        for i in range(len(self.r)):
            norm_list.append(norm(self.r[i]))
        max_r = max(norm_list)
        min_r = min(norm_list)
        e = (max_r- min_r)/(max_r + min_r)
        print("Eccentricity ala rmaxmin: {0:2.3f}".format(e))

    def Calc_Extentricity_Vector(self, r0 , v0):
        r = np.append(r0,0)
        v = np.append(v0,0)
        #print(r)
        L = np.cross(r,v)
        e_vec = (np.cross(v, L))/GM - r/ norm(r)
        e = norm(e_vec)
        print("Another way e = {0:.3f}".format(e))


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


def main():
    params = {
        'test' : 2,
        'testa' : 100
    }
    e = 0
    v0mag =  2.2*np.pi/2 * AU/YR
    r0mag = 1*(1-e)
    #v0mag =  np.sqrt(GM/1* (1+e)/(1-e))
    r0 = r0mag*array([1,0])
    v0 = v0mag*array([0,1])
    t = 0.37
    newOrbit = Orbit(r0, v0, 1)
    newOrbit.RungeKutta_adaptive(newOrbit.dt, newOrbit.err, newOrbit.tmax)
    print("nr of steps",len(newOrbit.r)-1)

    e = 0.75
    e = 0
    r0mag = 1*(1+e)
    v0mag =  np.sqrt(GM/1* (1-e)/(1+e))
    r0 = r0mag*array([1,0])
    v0 = v0mag*array([0,1])

    newOrbit_Ellips = Orbit(r0, v0, 1)
    newOrbit_Ellips.RungeKutta_adaptive(newOrbit_Ellips.dt, newOrbit_Ellips.err, newOrbit_Ellips.tmax)
    #newOrbit2.RungeKutta(newOrbit2.n, newOrbit2.r, newOrbit2.v, newOrbit2.t, newOrbit2.dt, acc, newOrbit2.E)[0]
    #newOrbit2.RungeKutta_adaptive_2(newOrbit.dt, newOrbit.err, newOrbit.tmax)[0]
    #newOrbit.CalcExcentricity()

    # For the Excentisity !
    newOrbit.Calc_Extentricity_Vector(newOrbit.r[-2],newOrbit.v[-2])
    #newOrbit.displacement(newOrbit.r)
    newOrbit.CalcExcentricity(newOrbit.r)
    #Energie
    #newOrbit.EnergyPlot()
    #newOrbit2.EnergyPlot()

    newOrbit.PlotSystem()
    newOrbit.PlotOrbit(newOrbit.r[:,0],newOrbit.r[:,1])
    newOrbit_Ellips.PlotOrbit(newOrbit_Ellips.r[:,0],newOrbit_Ellips.r[:,1])
    plt.show()



main()
