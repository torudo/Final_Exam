import cmath
from scipy.linalg import expm, sinm, cosm
import random
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

print(np.linspace(0,10,101))
#
# # Author: Tobias Rudolph
# # Date: 10.07.2020
#
# def Probability(x):
#     return np.absolute(x)**2
#
# dim = 2
# #omega = 2.
# omega = 2.0* 2.*np.pi
# alpha = -0.100 * 2* np.pi
#
# tau = 0.001
# T_Simulation = 100
#
#
# null = np.zeros(dim, dtype=complex)
# null[0] = 1
# one = np.zeros(dim, dtype = complex)
# one[1] = 1
# #....
#
# state_plus = (null + one ) /np.sqrt(2.)
# print(null,one,state_plus)
#
# # null    = np.array([1,0] , dtype=float)
# # one     = np.array([0,1] , dtype=float)
# # print(null.T,one.transpose())
#
# #print(null,one)
#
#
# #a = np.diag(np.sqrt(np.linspace(0, dim+1, dim)))
# a = np.diag(np.linspace(0, dim-1, dim))
# a = np.sqrt(a)
# a = np.vstack((a[1:], a[0]))
# #switchs lines
# #a[[-1,0]] = a[[0,-1]]
#
# #print(a)
# #print(a.T)
# #print("\n")
#
# a_degga = np.conj(a.T)
# #print(dot(a_degga,a))
#
# I = np.diag(np.linspace(1, dim-1, dim))
#
#
# H_Trans = omega* dot(a_degga, a) + alpha/2.*dot(a_degga, a)*( dot(a_degga, a)-I)
# #print(H_Trans)
#
#
# n_steps = int(ceil(T_Simulation/tau))
# #print(n_steps)
#
# sigma_x = array([[0, 1],[1, 0]])
# sigma_y = array([[0, -1j],[1j, 0]])
# sigma_z = array([[1, 0],[0, -1]])
#
# X = []
# Y = []
# Z = []
#
# psi_0   = state_plus
# psi     = psi_0
# Pulse_0 = np.pi/2.
# beta    = np.pi/T_Simulation
#
#
#
#
#
# for i in range(0, n_steps):
#     #H_Pulse = np.zeros((dim,dim), dtype=float)
#     Pulse = Pulse_0*beta*np.sin(beta * i)*np.cos(omega * i)
#     H_Pulse = Pulse*(a_degga + a)
#     #print(H_Pulse)
#     #Psi(t)= exp(i tau H) psi(0)
#     print(i)
#     U = np.dot(expm(-1j*tau/2.*H_Pulse),np.dot(expm(-1j*tau*H_Trans),expm(-1j*tau/2.*H_Pulse)))
#
#     psi = np.dot(U,psi)
#     #print(psi)
#     X.append( np.dot(np.dot(np.conj(psi.T),sigma_x),psi) )
#     Y.append( np.dot(np.dot(np.conj(psi.T),sigma_y),psi) )
#     Z.append( np.dot(np.dot(np.conj(psi.T),sigma_z),psi) )
#
#
# # print(np.arange(3))
# # print(np.linspace(0,3,4))
#
#
# #a = np.delete(a, (0), axis=0)
# #print(a)
# t = np.linspace(0,n_steps,n_steps)*tau
#
# # plt.plot(t, Probability(X), label="<X>(t)")
# # plt.plot(t, Probability(Y), label="<Y>(t)")
# # plt.plot(t, Probability(Z), label="<Z>(t)")
# fig, ax = plt.subplots()
#
# plt.plot(t, X, label=r"$\langle X\rangle(t)$")
# plt.plot(t, Y, label=r"$\langle Y\rangle(t)$")
# plt.plot(t, Z, label=r"$\langle Z\rangle(t)$")
#
# plt.xlim(-0.1,5.1)
# plt.ylim(-1.5,1.5)
# plt.ylabel(r"$ \langle A \rangle(t)$")
# plt.xlabel(r"$t$")
#
# plt.legend(ncol=3)
# plt.grid(linestyle=':',alpha=0.2)
#
# ax.minorticks_on()
# ax.tick_params(which='major', direction='in')#,direction='inout', length=10, width=2,)
# ax.tick_params(which='minor', direction='in',color = '#0f0f0f50')#, length=5, width=2,)
#
# filename = "first_case{0:.3}.pdf".format(omega)
# plt.savefig(filename, bbox_inches='tight')
#
# plt.show()
