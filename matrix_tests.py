import cmath
from scipy.linalg import expm, sinm, cosm
import random
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

self.H_Trans    = self.omega * dot(self.a_degga, self.a) \
                    + self.alpha/2. * dot(self.a_degga, self.a) \
                    * ( dot(self.a_degga, self.a) - self.I)

# Defines the exponential of it for the TimeOperatoor
self.expH_Trans = expm(-1j * self.tau * self.H_Trans)
self.expH_Trans = np.array([[1., 0.], [0., np.exp(complex(0, -self.tau*self.omega))]])
