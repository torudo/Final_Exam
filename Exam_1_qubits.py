import cmath
from scipy.linalg import expm, sinm, cosm
import random
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


# Author: Tobias Rudolph
# Date: 10.07.2020

#Global list_variables

omega   = 2.0 * 2.*np.pi
alpha   = -0.100 * 2* np.pi


def Probability(x):
    return np.absolute(x)**2


class Qubit(object):
    def __init__(self, tau, T_Sim, dim):
        self.dim        = dim
        self.tau        = tau
        self.T_Sim      = T_Sim

        self.N          = int(ceil(self.T_Sim/self.tau))

        self.time_array  = np.linspace(0,self.N,self.N+1) * self.tau

        self.a          = np.diag(np.linspace(0, self.dim-1, self.dim))
        self.a          = np.sqrt(self.a)
        self.a          = np.vstack((self.a[1:], self.a[0]))

        self.a_degga    = np.conj(self.a.T)

        self.I          = np.diag(np.linspace(1, self.dim, self.dim))

        self.sigma_x    = array([[0, 1],    [1, 0]])
        self.sigma_y    = array([[0, -1j],  [1j, 0]])
        self.sigma_z    = array([[1, 0],    [0, -1]])

        #self.Psi        = np.zeros((self.dim,self.dim), dtype = complex)
        self.B          = np.zeros((dim,dim), dtype=complex)  #Container of vectors

        self.X = []
        self.Y = []
        self.Z = []

        self.H_Trans    = omega * dot(self.a_degga, self.a) \
                            + alpha/2. * dot(self.a_degga, self.a) \
                            * ( dot(self.a_degga, self.a) - self.I)
        self.expH_Trans = expm(-1j * self.tau * self.H_Trans)

        self.H_Drive    = np.zeros((self.dim,self.dim), dtype = complex)
        self.U          = np.zeros((self.dim,self.dim), dtype = complex)

        self.Pulse_0    = np.pi/2.
        self.beta       = np.pi/self.T_Sim
        self.Pulse      = 0

    def Initialize(self):
        for i in range(self.dim):
            self.B[i][i] = 1
        #for task 1
        #self.Psi_0 = (self.B[0] + self.B[1] ) /np.sqrt(2.)
        self.Psi_0 = self.B[0]
        self.Psi = self.Psi_0


    def PrintAll(self):
        print("a\n",self.a,"\nadegga\n",self.a_degga,"\nI\n",self.I)
        print(self.B[0])
        print(self.B[1])
        print(self.Psi_0)



    def CalcExpH1(self, t):
        arg  = self.tau*self.Pulse/2.
        c = np.cos(arg)
        s = np.sin(arg)
        mat =  np.zeros((self.dim,self.dim), dtype = complex)
        for i in range(self.dim - 1):
            if i % 2 == 0:
                mat[i, i] = c
                mat[i+1, i+1] = c
                mat[i+1, i] = complex(0, s)
                mat[i, i+1] = complex(0, s)
        mat[-1, -1] = 1
        #print('ExpK1 = ', mat)
        return mat


    def CalcExpH2(self, t):
        arg = self.tau*self.Pulse/2.
        c = np.cos(arg)
        s = np.sin(arg)
        mat =  np.zeros((self.dim,self.dim), dtype = complex)
        for i in range(self.dim - 1):
            if i % 2 == 1:
                mat[i, i] = c
                mat[i+1, i+1] = c
                mat[i+1, i] = complex(0, s)
                mat[i, i+1] = complex(0, s)
        mat[0, 0] = 1
        #print('ExpK2 = ', mat)
        return mat


    def CalcTimeoperator(self, t):
        self.Pulse = self.Pulse_0 * self.beta * np.sin(self.beta * t) * np.cos(omega * t)
        self.H_Drive = self.Pulse*(self.a_degga + self.a)
        self.expH_Drive = expm(-1j * self.tau/2. * self.H_Drive)
        #self.U = np.dot(self.expH_Drive, np.dot(self.expH_Trans, self.expH_Drive))

        ExpH1 = self.CalcExpH1(t)
        ExpH2 = self.CalcExpH2(t)
        self.U = np.matmul(ExpH1, np.matmul(ExpH2, np.matmul(self.expH_Trans, np.matmul(ExpH2, ExpH1))))
        #print(self.U)

    def RunStep(self, t):
        self.CalcTimeoperator(t)
        self.Psi = np.dot(self.U, self.Psi)
        #self.X.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_x),self.Psi) )
        #self.Y.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_y),self.Psi) )
        #self.Z.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_z),self.Psi) )
        #print(self.X)
        #self.X.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_x),self.Psi) )
        #self.Y.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_y),self.Psi) )
        self.Z.append( np.dot(np.dot((np.conj(self.Psi[0]),np.conj(self.Psi[1])),self.sigma_z),(self.Psi[0],self.Psi[1])) )
        #print(self.X)


    def TimeEvaluation(self):
        self.Initialize()

        #for timestep in range(0, self.N):
        for timestep in self.time_array:
            self.RunStep(timestep)
            print(timestep)

        #self.OutputToFile()
        #self.Plot_Part_one()
        self.PlotOnlyZ()

    def Plot_Part_one(self):
        fig, ax = plt.subplots()
        plt.plot(self.time_array, self.X, label=r"$\langle X\rangle(t)$")
        plt.plot(self.time_array, self.Y, label=r"$\langle Y\rangle(t)$")
        plt.plot(self.time_array, self.Z, label=r"$\langle Z\rangle(t)$")

        #plt.xlim(-0.1,5.1)
        #plt.ylim(-1.5,1.5)
        plt.ylabel(r"$ \langle A \rangle(t)$")
        plt.xlabel(r"$t$")

        plt.legend(ncol=3)
        plt.grid(linestyle=':',alpha=0.2)

        ax.minorticks_on()
        ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
        ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)

        filename = "Pulse_first_case{0:.3}.pdf".format(omega)
        plt.savefig(filename, bbox_inches='tight')

        plt.show()


    def PlotOnlyZ(self):
        fig, ax = plt.subplots()
        #plt.plot(self.time_array, self.X, label=r"$\langle X\rangle(t)$")
        #plt.plot(self.time_array, self.Y, label=r"$\langle Y\rangle(t)$")
        plt.plot(self.time_array, self.Z, label=r"$\langle Z\rangle(t)$")

        #plt.xlim(-0.1,5.1)
        plt.ylim(-1.1,1.1)
        plt.ylabel(r"$ \langle A \rangle(t)$")
        plt.xlabel(r"$t$")

        plt.legend(ncol=3)
        plt.grid(linestyle=':',alpha=0.2)

        ax.minorticks_on()
        ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
        ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)

        filename = "Pulse_first_case{0:.3}.pdf".format(omega)
        plt.savefig(filename, bbox_inches='tight')

        plt.show()


    def OutputToFile(self):
        folder = "."
        #folder = ".\ohne_Wall"
        fname = folder + "\QSim_Pulse_{0:4.2f}.txt".format(self.T_Sim)
        out = open(fname,"w")
        out.write("#time\tX\tY\tZ\n")
        for i in range(0, len(self.time_array)):
            out.write("{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t \n".\
            format(self.time_array[i], self.X[i].real, self.Y[i].real, self.Z[i].real ) )
        out.close

################################################################################
################################# End of Class #################################
################################################################################



def main():
    params = {
        'dim' : 2,
        'T_Sim' : 100,
        'tau' : 0.001
    }

    newQubit = Qubit(**params)
    newQubit.Initialize()
    newQubit.TimeEvaluation()


    newQubit.CalcTimeoperator(0)
    print(newQubit.a)
    print(newQubit.a_degga)
    print(newQubit.H_Drive)
    #newQubit.PrintAll()
    print("\n")
    print(newQubit.CalcExpH1())
    print(newQubit.CalcExpH2())
    print("\n")



    print("\n")
    #testmat = array([[0,1],[1,0]])
    testmat = array([[0,1,0],[1,0,1],[0,1,0]])
    print(expm(-1j*0.001*testmat))
    vgltest = array([[cos(1), 1j*sin(1)],[1j*sin(1),cos(1)]])
    print(vgltest)



main()
