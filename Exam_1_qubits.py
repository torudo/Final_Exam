import cmath
from scipy.linalg import expm, sinm, cosm
import random
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


# Author: Tobias Rudolph
# Date: 10.07.2020


def Probability(x):
    return np.absolute(x)**2


class Qubit(object):
    """
    The quibit class contains the variables for the simulation of the
    timeevolution
    Takes omega and alpha and divides by 2pi
    """
    def __init__(self, tau, T_Sim, dim, omega, alpha):
        self.dim        = dim               # dimension
        self.tau        = tau               # time step
        self.T_Sim      = T_Sim             # time at which simulation ends
        # Time Grid vor simulation
        self.N          = int(ceil(self.T_Sim/self.tau))
        self.time_array = np.linspace(0,self.N,self.N+1) *self.tau
        # Creates the Matrix for ladder operator
        self.a          = np.diag(np.linspace(0, self.dim-1, self.dim))
        self.a          = np.sqrt(self.a)
        self.a          = np.vstack((self.a[1:], self.a[0]))
        self.a_degga    = np.conj(self.a.T)
        adegga_a        = np.dot(self.a_degga, self.a)
        self.I          = np.identity(self.dim) # Identity used in H_Trans

        self.omega      = omega*2*np.pi     # Parameter for H_Transmon
        self.alpha      = alpha*2*np.pi     # Parameter for H_Transmon
        # Calculates the Transmon Hamiltionion
        self.H_Trans    = self.omega * adegga_a + self.alpha/2. * ( dot(adegga_a, adegga_a)-dot(adegga_a,self.I))
        # Defines the exponential of it for the TimeOperatoor
        self.expH_Trans = expm(-1j * self.tau * self.H_Trans)

        # Container of the states
        self.B          = np.zeros((dim,dim), dtype=complex)  #Container of vectors
        # Pauli Matrixes
        self.sigma_x    = array([[0, 1],    [1, 0]])
        self.sigma_y    = array([[0, -1j],  [1j, 0]])
        self.sigma_z    = array([[1, 0],    [0, -1]])
        # List for the time evolution of pauli matrizes
        self.X = []
        self.Y = []
        self.Z = []

        # Analytic for 2D !
        #self.expH_Trans = np.array([[1., 0.], [0., np.exp(complex(0, -self.tau*self.omega))]])
        #self.expH_Trans = np.array([[1,0],[0,np.exp(-1j*omega)]])

        # Container for the Drive Hamiltonian
        self.H_Drive    = np.zeros((self.dim,self.dim), dtype = complex)
        self.U          = np.zeros((self.dim,self.dim), dtype = complex)

        self.Pulse_0    = np.pi/2.
        self.beta       = np.pi/self.T_Sim
        self.Pulse      = np.pi/2.

        self.leakage = []

    def Initialize(self):
        # Define the set of eigenvektors
        for i in range(self.dim):
            self.B[i][i] = 1
        #for task 1
        #self.Psi_0 = (self.B[0] + self.B[1] ) /np.sqrt(2.)
        #for taks 2
        #self.Psi_0 = self.B[0]
        #for task 3
        self.Psi_0 = self.B[1]
        self.Psi = self.Psi_0


    def CalcExpH1(self, t):
        mat =  np.zeros((self.dim,self.dim), dtype = complex)
        for i in range(self.dim - 1):
            if i % 2 == 0:
                # Sqrt factor from a adegga
                arg  = self.Pulse/2.*self.tau * np.sqrt(i+1)
                c = np.cos(arg)
                s = np.sin(arg)
                mat[i, i] = c
                mat[i+1, i+1] = c
                mat[i+1, i] = complex(0, s)
                mat[i, i+1] = complex(0, s)
        mat[-1, -1] = 1
        #print('ExpK1 = ', mat)
        return mat


    def CalcExpH2(self, t):
        arg = self.Pulse*self.tau/2.
        mat =  np.zeros((self.dim,self.dim), dtype = complex)
        for i in range(self.dim - 1):
            if i % 2 == 1:
                arg *= np.sqrt(i+1)
                c = np.cos(arg)
                s = np.sin(arg)
                mat[i, i] = c
                mat[i+1, i+1] = c
                mat[i+1, i] = complex(0, s)
                mat[i, i+1] = complex(0, s)
        mat[0, 0] = 1
        if self.dim == 2:
            mat[-1, -1] = 1
        ##print('ExpK2 = ', mat)
        return mat

    def PulseFunction(self, t):
        return self.Pulse_0 * self.beta * np.sin(self.beta * t) * np.cos(self.omega * t)

    def CalcTimeoperator(self, t, Hamil):
        self.Pulse = self.PulseFunction(t)
        # here the H matrixes have to be calculated
        ExpH1 = self.CalcExpH1(t)
        ExpH2 = self.CalcExpH2(t)
        self.U = np.matmul(ExpH1, np.matmul(ExpH2, np.matmul(self.expH_Trans, np.matmul(ExpH2, ExpH1))))
        return self.U
        #print(self.U)

    def CalcTimeoperator_2D(self, t, Hamil):
        Hamil = self.expH_Trans
        P = self.PulseFunction(t)
        a = np.cos(self.tau*P/2.)
        b = complex(0, np.sin(self.tau*P/2.))
        ExpB = np.array([[a, b], [b, a]], dtype = np.complex)
        print("Die H MAtrixen richtig!!!\n",ExpB )
        print("\n Space \n")

        return np.matmul(ExpB, np.matmul(Hamil, ExpB))

    def CalcTimeoperator_3D(self, t, Hamil):
        Hamil = self.expH_Trans
        P = self.PulseFunction(t)
        a = np.cos(self.tau*P/2.)
        b = complex(0, np.sin(self.tau*P/2.))

        ExpH1 = np.array([[a, b, 0], [b, a, 0], [0, 0, 1]], dtype = np.complex)
        ExpH2 = np.array([[1, 0, 0], [0, a, b], [0, b, a]], dtype = np.complex)
        # print("Die H MAtrixen richtig!!!\n",ExpH1, "\n",ExpH2 )
        # print("\n Space \n")
        return np.matmul(ExpH1, np.matmul(ExpH2, np.matmul(self.expH_Trans, np.matmul(ExpH2, ExpH1))))

    def RunStep(self, t):
        self.U = self.CalcTimeoperator(t, self.expH_Trans)
        #self.U = self.CalcTimeoperator_3D(t, self.expH_Trans)
        # Timeevolution of spi for one step
        self.Psi = np.matmul(self.U, self.Psi)
        #self.Psi = U.dot(self.Psi) # not in use

        # Cacluates the time evolution of the pauli matrix
        #self.X.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_x),self.Psi) )
        #self.Y.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_y),self.Psi) )
        #self.Z.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_z),self.Psi) )
        #self.Z.append( np.matmul(np.matmul(np.conj(self.Psi.T),self.sigma_z),self.Psi) )

        #print(self.X)
        # Another way to calculate the Z pauli matrix
        #self.X.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_x),self.Psi) )
        #self.Y.append( np.dot(np.dot(np.conj(self.Psi.T),self.sigma_y),self.Psi) )
        self.Z.append( np.dot(np.dot((np.conj(self.Psi[0]),np.conj(self.Psi[1])),self.sigma_z),(self.Psi[0],self.Psi[1])) )
        #print(self.X)


    def TimeEvaluation(self):
        # Initialize the system
        self.Initialize()

        # runs in time over the time array (until T_Sim)
        for timestep in self.time_array:
             self.RunStep(timestep)
             self.leakage.append(self.ComputeLeakage())
        #     #print(timestep)

        #Saves the data or plots it
        #self.OutputToFile()

        #self.Plot_Part_one()

        #self.PlotOnlyZ()


    def Plot_Pulseform(self):
        fig, ax = plt.subplots()
        #plt.plot(self.time_array, self.X, label=r"$\langle X\rangle(t)$")
        #plt.plot(self.time_array, self.Y, label=r"$\langle Y\rangle(t)$")
        plt.plot(self.time_array, self.PulseFunction(self.time_array) , label=r"Pulse()")

        #plt.xlim(-0.1,5.1)
        #plt.ylim(-1.1,1.1)
        plt.ylabel(r"$ \langle A \rangle(t)$")
        plt.xlabel(r"$t$")

        plt.legend(ncol=3)
        plt.grid(linestyle=':',alpha=0.2)

        ax.minorticks_on()
        ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
        ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)

        #filename = "Pulse_first_case{0:.3}.pdf".format(omega)
        #plt.savefig(filename, bbox_inches='tight')

        plt.show()

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

        filename = "Pulse_first_case{0:.3}.pdf".format(self.omega)
        plt.savefig(filename, bbox_inches='tight')

        plt.show()

    def PlotOnlyZ_enviroment(self):
        fig, ax = plt.subplots()
        #plt.plot(self.time_array, self.X, label=r"$\langle X\rangle(t)$")
        #plt.plot(self.time_array, self.Y, label=r"$\langle Y\rangle(t)$")
        #plt.plot(self.time_array, self.Z, label=r"$\langle Z\rangle(t)$")
        #plt.xlim(-0.1,5.1)
        plt.ylim(-1.1,1.1)
        plt.ylabel(r"$ \langle Z \rangle(t)$")
        plt.xlabel(r"$t$")

        plt.grid(linestyle=':',alpha=0.2)

        ax.minorticks_on()
        ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
        ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)



    def PrintAll(self):
        print("a\n",self.a,"\nadegga\n",self.a_degga,"\nI\n",self.I)
        print(self.B[0])
        print(self.B[1])
        print(self.Psi_0)

    def ComputeLeakage(self):
        p0 = Probability(self.Psi[0])
        p1 = Probability(self.Psi[1])
        return (1.-p0-p1)
#    def calculate_Propability

    def PlotLeakage(self):
        fig, ax = plt.subplots()
        plt.plot(self.time_array, self.leakage, label=r"$\alpha / 2\pi ={}$".format(self.alpha))
        #plt.xlim(-0.1,5.1)
        #plt.ylim(-1.5,1.5)
        plt.ylabel(r"$1-p_0(t)-p_1(t)$")
        plt.xlabel(r"$t$")

        plt.legend()#ncol=3)
        plt.grid(linestyle=':',alpha=0.2)

        ax.minorticks_on()
        ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
        ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)

        filename = "leakage_test{0:.3}.pdf".format(self.alpha)
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

    ###################
    # Part One 1 ##
    params2D = {
        'dim' : 2,
        'T_Sim' : 100,
        'tau' : 0.001,
        'omega': 2.0
    }

    params10D = {
        'dim' : 10,
        'T_Sim' : 100,
        'tau' : 0.001,
        'omega': 2.0,
        'alpha': -0.5
    }

    newQubit = Qubit(**params10D)
    newQubit.Initialize()
    newQubit.TimeEvaluation()

    #newQubit.PlotLeakage()
    newQubit.PlotOnlyZ_enviroment()

    #newQubit10D = Qubit(**params10D)
    #newQubit10D.Initialize()
    #newQubit10D.TimeEvaluation()

    plt.plot(newQubit.time_array, newQubit.Z,'.', markersize=0.5, label=r"2 D: $\langle Z\rangle(t)$")
    #plt.plot(newQubit10D.time_array, newQubit10D.Z,'.', label=r"10 D: $\langle Z\rangle(t)$")
    #plt.legend()
    #filename = "Pulse_first_case{0:.3}.pdf".format(self.omega)
    #plt.savefig(filename, bbox_inches='tight')
    plt.show()


    ##DEBUGGING
    # t = 0
    # print("\n This not")
    # print(newQubit.CalcTimeoperator(t,newQubit.expH_Trans))
    # print("This one is the right")
    # #U = newQubit.CalcTimeoperator_3D(t,newQubit.expH_Trans )
    # U = newQubit.CalcTimeoperator_2D(t,newQubit.expH_Trans )
    # print(U)
    #
    # print("This is H1 and H2 in OLd way!!!!!!!!!!!!!!!!")
    # print(newQubit.CalcExpH1(t))
    # print(newQubit.CalcExpH2(t))
    # print("\n")



    #print(newQubit.a)
    #print(newQubit.a_degga)
    #print(newQubit.H_Drive)

    #newQubit.PrintAll()



    #newQubit.Plot_Pulseform()

    print("\n")
    #testmat = array([[0,1],[1,0]])
    # testmat = array([[0,1,0],[1,0,1],[0,1,0]])
    # print(expm(-1j*0.001*testmat))
    # vgltest = array([[cos(1), 1j*sin(1)],[1j*sin(1),cos(1)]])
    # print(vgltest)


    # test = np.array([[0,1],[1,0]])
    # print(expm(-1j*test))
    # print("\n")
    # test = np.array([[1,0],[0,1]])
    # print(expm(-1j*test))
    # print("\n")
    # arg = -1
    # matrix_2d = np.array([[np.cos(arg), 1j*sin(arg)],[1j*sin(arg),cos(arg)]])
    # print(matrix_2d)

main()
