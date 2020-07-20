# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time


def odesolver12(f, t, y, h):
    """Calculate the next step of an initial value problem (IVP) of
    an ODE with a RHS described by f, with an order 1 approx.
    (Eulers Method) and an order 2 approx. (Midpoint Rule). This is
    the simplest embedded RK pair.
    Parameters:
        f: function. RHS of ODE.
        t: float. Current time.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 1 approx.
        w: float. Order 2 approx.
    """
    s1 = f(t, y)
    s2 = f(t+h, y+h*s1)
    w = y + h*s1
    q = y + h/2.0*(s1+s2)
    return w, q

def odesolver23(f, t, y, h):
    """Calculate the next step of an IVP of an ODE with a RHS
    described by f, with an order 2 approx. (Explicit Trapezoid
    Method) and an order 3 approx. (third order RK).
    Parameters:
        f: function. RHS of ODE.
        t: float. Current time.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(t, y)
    s2 = f(t+h, y+h*s1)
    s3 = f(t+h/2.0, y+h*(s1+s2)/4.0)
    w = y + h*(s1+s2)/2.0
    q = y + h*(s1+4.0*s3+s2)/6.0
    return w, q

def odesolver45(f, t, y, h):
    """Calculate the next step of an IVP of an ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        t: float. Current time.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(t, y)
    s2 = f(t+h/4.0, y+h*s1/4.0)
    s3 = f(t+3.0*h/8.0, y+3.0*h*s1/32.0+9.0*h*s2/32.0)
    s4 = f(t+12.0*h/13.0, y+1932.0*h*s1/2197.0-7200.0*h*s2/2197.0+7296.0*h*s3/2197.0)
    s5 = f(t+h, y+439.0*h*s1/216.0-8.0*h*s2+3680.0*h*s3/513.0-845.0*h*s4/4104.0)
    s6 = f(t+h/2.0, y-8.0*h*s1/27.0+2*h*s2-3544.0*h*s3/2565+1859.0*h*s4/4104.0-11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0+1408.0*s3/2565.0+2197.0*s4/4104.0-s5/5.0)
    q = y + h*(16.0*s1/135.0+6656.0*s3/12825.0+28561.0*s4/56430.0-9.0*s5/50.0+2.0*s6/55.0)
    return w, q


def RHS1(t, y):
    return t - 2*t*y

def analytical(t, y):
    return 0.5*(1-np.exp(-t**2))

def rk_adaptive(ode, rhs, y0=0.0, t0=0.0, TOL=1e-04, theta=1e-02, tmax=1.0):
    """Perform an adaptive RK method.
    Parameters:
        ode:   function. ODE solver.
        rhs:   function. RHS of ODE.
        y0:    float, optional. Initial position.
        t0:    float, optional. Initial time.
        TOL:   float, optional. Tolerance of relative error.
        theta: float, optional. "Protective" constant.
        tmax:  float, optional. End of calculation interval.
    Returns:
        y:     list. Position.
        t:     list. Time.
        i:     int. Number of iterations
    """

    # Allocate lists to store position and time and set
    # initial conditions.
    y = []
    t = []
    y.append(y0)
    t.append(t0)

    # Set initial step size and declare iteration integer
    h = 1.0
    i = 0

    while (t[i] < tmax):
        # Get two different approximations
        w, q = ode(rhs, t[i], y[i], h)
        # Estimate error
        e = abs((w-q)/max(w, theta))
        # If e larger thant TOL, decrease step length
        if (e > TOL):
            h = 0.8*(TOL*e)**(1/5)*h
            # Get two new approximations
            w, q = ode(rhs, t[i], y[i], h)
            # Estimate new error
            e = abs((w-q)/max(w, theta))
            # If e still larger than TOL, halve step length until False
            while (e > TOL):
                h = h/2.0
                # New approximations
                w, q = ode(rhs, t[i], y[i], h)
                # New error estimate
                e = abs((w-q)/max(w, theta))
        # Store highest order approximation as next y-value
        y.append(q)
        # Store current time + step size as next time
        t.append(t[i] + h)
        # Increment step number
        i += 1
        # Check if e is too small, if so, double step size
        if (e < 0.1*TOL):
            h = h*2.0

    return y, t, i

pos, times, iterations = rk_adaptive(odesolver12, RHS1, y0=0.0, t0=0.0, TOL=1e-02, theta=1e-03, tmax=1.0)


# Declare plot and plot results
plt.figure()

# Analytical plot
ta = np.linspace(0, times[iterations])
ya = analytical(ta, 0)
plt.plot(ta, ya , label='Analytical solution.')

# RK1/2 plot
plt.plot(times, pos, '-o' , label='RK1/2. %i steps.' % (iterations+1))
pos2, times2, iterations2 = rk_adaptive(odesolver23, RHS1, y0=0.0, t0=0.0, TOL=1e-02, theta=1e-03, tmax=1.0)
plt.plot(times2, pos2, '-o' , label='RK1/2. %i steps.' % (iterations2+1))
pos3, times3, iterations3 = rk_adaptive(odesolver45, RHS1, y0=0.0, t0=0.0, TOL=1e-02, theta=1e-03, tmax=1.0)
plt.plot(times3, pos3, '-o' , label='RK1/2. %i steps.' % (iterations3+1))
plt.ylabel(r'$y(t)$')
plt.xlabel(r'$t$')
plt.legend(loc="best");
plt.show()
