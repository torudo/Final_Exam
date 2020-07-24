import numpy as np
import matplotlib.pyplot as plt


class NMR:

    def __init__(self, t_1, t_2, n, res):
        self.res        = res
        self.N          = n
        self.timelength = int(n*res)
        self.time       = np.linspace(0, self.N-1, self.N)
        self.f_0        = 4
        self.f_1        = 1./4.

        self.B_0        = 2.*np.pi*self.f_0
        self.h          = np.pi*2.*self.f_1
        self.gamma      = 1.
        self.w_0        = self.B_0

        self.T_1        = t_1
        self.T_2        = t_2

        self.B          = np.zeros(3, dtype = np.float)

        self.M_x        = []
        self.M_y        = []
        self.M_z        = []


    def CalcB(self, t, phase):
        self.B[0] = self.h * np.cos(self.w_0 * t + phase)
        self.B[1] = -self.h * np.sin(self.w_0 * t + phase)
        self.B[2] = self.B_0

    def InitM(self, m_x, m_y, m_z):
        self.M_x.append(m_x)
        self.M_y.append(m_y)
        self.M_z.append(m_z)

    def CalcExpC(self, tau):
        mat = np.zeros((3, 3), dtype = np.float)
        mat[0, 0] = np.exp(-tau*self.T_2/2.)
        mat[1, 1] = np.exp(-tau*self.T_2/2.)
        mat[2, 2] = np.exp(-tau*self.T_1/2.)
        return mat

    def CalcExpB(self, tau, loc_time):

        self.CalcB(loc_time, np.pi/2.)
        print(self.B[0], self.B[1], self.B[2],"\n")
        mat = np.zeros((3, 3), dtype = np.float)
        omega_sq = self.B[0]**2 + self.B[1]**2 + self.B[2]**2
        omega = np.sqrt(omega_sq)

        mat[0, 0] = self.B[0]**2 + (self.B[1]**2 + self.B[2]**2)*np.cos(omega * tau)
        mat[1, 1] = self.B[1]**2 + (self.B[0]**2 + self.B[2]**2)*np.cos(omega * tau)
        mat[2, 2] = self.B[2]**2 + (self.B[0]**2 + self.B[1]**2)*np.cos(omega * tau)

        mat[0, 1] = self.B[0]*self.B[1]*(1-np.cos(omega * tau)) + omega*self.B[2]*np.sin(omega * tau)
        mat[0, 2] = self.B[0]*self.B[2]*(1-np.cos(omega * tau)) - omega*self.B[1]*np.sin(omega * tau)
        mat[1, 0] = self.B[0]*self.B[1]*(1-np.cos(omega * tau)) - omega*self.B[2]*np.sin(omega * tau)
        mat[1, 2] = self.B[1]*self.B[2]*(1-np.cos(omega * tau)) + omega*self.B[0]*np.sin(omega * tau)
        mat[2, 0] = self.B[0]*self.B[2]*(1-np.cos(omega * tau)) + omega*self.B[1]*np.sin(omega * tau)
        mat[2, 1] = self.B[1]*self.B[2]*(1-np.cos(omega * tau)) - omega*self.B[0]*np.sin(omega * tau)

        return mat/omega_sq

    def CalcUpdateMatrix(self,i):

        mat = np.matmul(self.CalcExpC(self.res/2.), np.matmul(self.CalcExpB(self.res/2., i ),
                        self.CalcExpC(self.res/2.)))
        print("Time {}".format(i) )
        return mat

    def Update(self, time):
        mat = self.CalcUpdateMatrix(time)
        M_updated = np.zeros(3, dtype = np.float)
        M_old = np.array([self.M_x[-1], self.M_y[-1], self.M_z[-1]])
        M_updated = mat.dot(M_old)

        self.M_x.append(M_updated[0])
        self.M_y.append(M_updated[1])
        self.M_z.append(M_updated[2])

    def PlotMag(self):
        plt.plot(self.time*self.res, self.M_x, label = r'$M_x(t)$')
        plt.plot(self.time*self.res, self.M_y, label = r'$M_y(t)$')
        plt.plot(self.time*self.res, self.M_z, label = r'$M_z(t)$')
        #plt.xlim(0,self.timelength)
        #plt.ylim(-1.1,1.1)
        plt.legend()
        plt.title(r"1/$T_1$ ={0}, 1/$T_2$ ={1}, $M(0)$={2} ".format(self.T_1,self.T_2,(self.M_x[0], self.M_y[0], self.M_z[0]) ) )
        plt.xlabel(r"$t$")
        plt.savefig('NMR_T1-{0}_T2-{1}_Minit-{2}{3}{4}_tau_{5}_t_{6}.pdf'.format(self.T_1,self.T_2,self.M_x[0], self.M_y[0], self.M_z[0],self.res,self.timelength), bbox_inches='tight')
        plt.show()

    def Run(self):
        dt = 0
        for i in range(self.N-1):
            self.Update(dt)
            print(np.array([self.M_x[i], self.M_y[i], self.M_z[i]]))
            dt += 0.5*self.res

        print(np.array([self.M_x[-1], self.M_y[-1], self.M_z[-1]]))
        self.PlotMag()

    def Run_in_Time(self):
        t = 10
        tau = 0.01
        dt = 0
        for i in range(0, int(t/tau)-1):
            self.Update(dt)
            print(np.array([self.M_x[i], self.M_y[i], self.M_z[i]]))
            dt += tau*0.5

        print(np.array([self.M_x[-1], self.M_y[-1], self.M_z[-1]]))
        self.PlotMag()


def main():
    t_1 = 1
    t_2 = 0
    t = 100

    NewExp = NMR(t_1, t_2, 10*t, 1/t)
    NewExp.InitM(0, 0, 1)
    NewExp.Run()

main()
