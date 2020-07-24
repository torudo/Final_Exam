def RKStep(self, r0, v0, t, dt, acc, E):
    #rr = norm(r)
    #print(rr)
    x = r0[0]
    y = r0[1]

    u = v0[0]
    v = v0[1]

    k1x = dt*u
    k1y = dt*v
    l1u = dt*acc(x,y)                         #use RK4 method
    l1v = dt*acc(y,x)                         #use RK4 method

    k2x = dt*(u + l1u/2.)
    k2y = dt*(v + l1v/2.)
    l2u = dt*acc(x + k1x/2., y + k1y/2.)
    l2v = dt*acc(y + k1y/2., x + k1x/2.)

    k3x = dt*(u + l2u/2.)
    k3y = dt*(v + l2v/2.)
    l3u = dt*acc(x + k2x/2., y+ k2y/2.)
    l3v = dt*acc(y + k2y/2., x+ k2x/2.)

    k4x = dt*(u + l3u)
    k4y = dt*(v + l3v)
    l4u = dt*acc(x + k3x, y + k3y)
    l4v = dt*acc(y + k3y, x+ k3x)


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
    return [x_new,y_new], [u_new,v_new], t_new


    # Beschleunigungen der RK
        def acc(self, x0, y0):
            r = math.sqrt(x0**2 + y0**2)
            return -GM*x0 / r**3





                def rhs(self, X, V):
                    r = math.sqrt(X[0]**2 + X[1]**2)
                    xdot = V[0]
                    ydot = V[1]
                    udot = -GM*X[0]/r**3
                    vdot = -GM*X[1]/r**3
                    return xdot, ydot, udot, vdot



                    def RungeKutta_adaptive(self, dt, err, tmax):
                        #start values
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
                                        RK4_singlestep(r, v, t, 0.5*dt, self.rhs)#, E)

                                    rnew, vnew, tnew =\
                                        RK4_singlestep(rtemp, vtemp, ttemp, 0.5*dt, self.rhs)#, Etemp)
                                    # single step to compare with
                                    rsingstep, vsingstep, tsingstep =\
                                        RK4_singlestep(r, v, t, dt, self.rhs)#, E)
                                    # New error
                                    rel_error = max(np.abs((rnew[0]-rsingstep[0])/rnew[0]),
                                                    np.abs((rnew[1]-rsingstep[1])/rnew[1]),
                                                    np.abs((vnew[0]-vsingstep[0])/vnew[0]),
                                                    np.abs((vnew[1]-vsingstep[1])/vnew[1]))

                                    #dt_est = dt*abs(self.err/rel_error)**(1./5.)
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
                                    RK4_singlestep(r, v, t, dt, self.rhs)#, E)
                            # Finally we made a step
                            t += dt

                            self.r = np.append(self.r ,[rnew] ,axis= 0)
                            self.v = np.append(self.v ,[vnew] ,axis= 0)
                            self.t = np.append(self.t ,t)

                            r = rnew; v = vnew
                        print("resets",n_reset)
                        return r, v, t
