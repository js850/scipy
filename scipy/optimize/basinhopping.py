# Original Author: Jacob Stevenson 2012


import numpy as np
import _minimize
from optimize import Result, _check_unknown_options

# TODO:

# Basin Hopping

"""
basin hopping algorithm

The important components of basin hopping are:

    step taking algorithm:
        The default for this should be random displacement with adaptive step size.

        The user should be able to specify their own step taking algorithm.  question:  how do we combine
        adaptive step size with user defined step taking?

    optimizer:
        This is the routine which does local minimization.  The default should be scipy lbfgs algorithm
        
        The user should be able to specify their own minimizer.

    accept test:
        The default will be metropolis criterion with temperature T=1.

        This is where we can implement bounds on the problem.

        Should be able to add their own accept or reject test?

    Storage:
        A class for storing the best structure found

        possibly also the best N structures found?
"""

class _Storage(object):
    def __init__(self, x, f):
        self._add(x, f)
    def _add(self, x, f):
        self.x = np.copy(x)
        self.f = f
    def insert(self, x, f):
        if f < self.f:
            self._add(x, f)
    def get_lowest(self):
        return self.x, self.f

class _BasinHopping(object):
    def __init__(self, x0, minimizer, step_taking, accept_tests, iprint=1 ):
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        print "accept_tests", accept_tests
        self.iprint = iprint

        self.nstep = 0
        self.takestep_report = True #should we report the result of the step to takestep?

        #do initial minimization
        res = minimizer(self.x)
        self.x = np.copy(res.x)
        self.energy = res.fun
        print "basinhopping step %d: energy %g" % (self.nstep, self.energy)

        #initialize storage class
        self.storage = _Storage(self.x, self.energy)

    def _monte_carlo_step(self):
        #take a random step
        #make a copy of x because the step_taking algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        #do a local minimization
        res = self.minimizer(x_after_step)
        x_after_quench = res.x
        energy_after_quench = res.fun
        if not res.success:
            print "warning: basinhoppping: minimize failure"

        #accept the move based on self.accept_tests
        #if any one of accept test is false, than reject the step
        accept = True
        for test in self.accept_tests:
            if not test(enew=energy_after_quench, xnew=x_after_quench, eold=self.energy, xold=self.x):
                accept = False
                break

        #Report the result of the acceptance test to the take step class.  This
        #is for adaptive step taking
        if self.takestep_report:
            self.step_taking.report(accept)

        return x_after_quench, energy_after_quench, accept

    def one_cycle(self):
        self.nstep += 1

        xtrial, etrial, accept = self._monte_carlo_step()

        eold = self.energy
        if accept:
            self.energy = etrial
            self.x = np.copy(xtrial)
            self.storage.insert( self.x, self.energy )

        if self.iprint > 0:
            if self.nstep % self.iprint == 0:
                self.print_report(etrial, accept)

    def print_report(self, etrial, accept):
        xlowest, elowest = self.storage.get_lowest()
        #print self.storage.get_lowest()
        #print self.storage.get_lowest()[1]
        print "basinhopping step %d: energy %g trial_energy %g accepted %d lowest_energy %g" % (self.nstep, self.energy, etrial, accept, elowest)

class AdaptiveStepsize(object):
    def __init__( self, takestep, accept_rate=0.5, interval = 50, factor = 0.9, verbose=True ):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = True

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)
    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            #We're accepting too many steps.  This generally means we're trapped in a basin.
            #Take bigger steps 
            self.takestep.stepsize /= self.factor
        else:
            #We're not accepting enough steps.  Take smaller steps 
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print "adaptive stepsize: acceptance rate %f target %f new stepsize %g old stepsize %g" % (accept_rate, self.target_accept_rate, self.takestep.stepsize, old_stepsize)
    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1

        if self.nstep % self.interval == 0:
            self._adjust_step_size()

        return self.takestep(x)

    def report(self, accept):
        if accept:
            self.naccept += 1


class RandomDisplacement(object):
    def __init__(self, stepsize = 0.5):
        self.stepsize = stepsize
    def __call__(self, x):
        x += np.random.uniform(-self.stepsize, self.stepsize, np.shape(x) )
        return x

class _MinimizerWrapper(object):
    """
    wrap a minimizer function as a minimizer class
    """
    def __init__(self, minimizer, func=None, **kwargs):
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs
    def __call__(self, x0):
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)


class _Metropolis(object):
    def __init__(self, T):
        """
        Metropolis acceptance criterion
        """
        self.beta = 1.0 / T
    def accept_reject(self, enew, eold):
        w=min(1.0, np.exp(-(enew - eold) * self.beta ) )
        rand = np.random.rand()
        return w >= rand
    def __call__(self, **kwargs):
        """
        enew and eold are manditory in kwargs
        """
        return self.accept_reject(kwargs["enew"], kwargs["eold"])




def basinhopping(x0, func=None, args=(), optimizer=None,
        minimizer=None, minimizer_kwargs=dict(),
           maxiter=10000, T=1.0, stepsize=0.5, interval=50,
           iprint=1):
    """Minimize a function using the basin hopping algorithm

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    func : callable ``f(x, *args)``, optional
        Function to be optimized.  Either func or minimizer must be passed
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (Jacobian, Hessian).
    minimizer : callable ``minimizer(x0, **minimizer_kwargs)``, optional
        Use this minizer rather than the default.  If the minimizer is given
        then, func is not used.  basinhopping will get the function values from
        the output of minimizer
    minimizer_kwargs : tuple, optional
        Extra arguments to be passed to the minimizer.  If argument minimizer
        is specified, then it is passed to that, else it is passed to the default
        scipy.optimize.minimize().  See scipy.optimize.minimize() for details.
        If the default minimzer is used, some important options could be

            method : the method to use in minimizations
            jac : specify the jacobian for gradient minimizations
            hess : specify the hessian for hessian based minimizations
            tol : tolerance
            
    maxiter : integer, optional
        The maximum number of basin hopping iterations
    T : float, optional
        The ``temperature`` parameter for the accept or reject criterion.
    stepsize : float, optional
        initial stepsize for use in the random displacement
    interval : integer, optional
        interval for how often to update the stepsize
    iprint : integer, optional
        The interval at which to print status information.  iprint < 0 for a
        silent run


    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.
        Important attributes are: ``x`` the solution array, and ``message``
        which describes the cause of the termination. See `Result` for a
        description of other attributes.

    Notes
    -----
    Basin hopping is a random algorithm which attemps to find the global
    minimum of a smooth scalar function of one or more variables.  The algorith
    was originally described by David Wales http://www-wales.ch.cam.ac.uk/
    The algorithm is iterative with each iteration composed of the following
    steps

    1) random displacement of the coordinates

    2) local minimization

    3) accept or reject the new coordinates based on the minimized function
    value.


    """
    x0 = np.array(x0)

    #set up minimizer
    if minimizer is None and func is None:
        raise ValueError("minimizer and func cannot both be None")
    if callable(minimizer):
        wrapped_minimizer = _MinimizerWrapper(minimizer, **minimizer_kwargs)
    else:
        #use default
        #minimizer_kwargs = dict()
        #minimizer_kwargs["method"] = "L-BFGS-B"
        #minimizer_kwargs["jac"] = True
        if len(args) > 0:
            #should we be worried about overwriting?
            minimizer_kwargs["args"] = args
        wrapped_minimizer = _MinimizerWrapper(_minimize.minimize, func, **minimizer_kwargs)
        
    #set up step taking algorithm
    if True:
        #use default
        displace = RandomDisplacement(stepsize=stepsize)
        verbose = iprint > 0
        step_taking = AdaptiveStepsize(displace, interval=interval)

    #set up accept tests
    if True:
        ##use default
        metropolis = _Metropolis(T) 
        accept_tests = [ metropolis ]

    bh = _BasinHopping(x0, wrapped_minimizer, step_taking, accept_tests, iprint=iprint)

    for i in range(maxiter):
        bh.one_cycle()

    return bh.storage.get_lowest()

    if False:
        if full_output:
            return res['x'], res['fun'], res['T'], res['nfev'], res['nit'], \
                res['accept'], res['status']
        else:
            return res['x'], res['status']


if __name__ == "__main__":
    from numpy import cos, sin
    if False:
        from pygmin.potentials.lj import LJ
        pot = LJ()
        x0 = np.random.uniform(-1,1,3*38)
        kwargs={ "method": "L-BFGS-B", "jac": True }
        ret = basinhopping(x0, func=pot.getEnergyGradient, minimizer_kwargs=kwargs, maxiter=100)
        print ret



    if False:
        def func(x):
            f =  cos(14.5*x-0.3) + (x+0.2)*x
            df = np.array(-14.5*sin(14.5*x-0.3) + 2.*x + 0.2)
            return f, df
        # minimum expected at ~-0.195
        kwargs={ "method": "L-BFGS-B", "jac": True } #, "options":{"disp":True} }
        x0 = np.array(1.0)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200, iprint=10)
        print "minimum expected at ~", -0.195
        print ret

    if False:
        # minimum expected at ~[-0.195, -0.1]
        def func(x):
            f = cos(14.5*x[0]-0.3) + (x[1]+0.2)*x[1] + (x[0]+0.2)*x[0]
            return f
        kwargs={ "method": "L-BFGS-B"} #, "options":{"disp":True} }
        x0 = np.array([1.0,1.])
        import scipy.optimize
        scipy.optimize.minimize(func, x0, **kwargs)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=900, iprint=1)
        print "minimum expected at ~", [-0.195, -0.1]
        print ret

    if True:
        #try a function with much higher barriers between the local minima.
        def func(x):
            f =  5.*cos(14.5*x-0.3) + 2.*(x+0.2)*x
            df = np.array(-5.*14.5*sin(14.5*x-0.3) + 2.*(2.*x + 0.2))
            return f, df
        # minimum expected at ~-0.195
        kwargs={ "method": "L-BFGS-B", "jac": True } #, "options":{"disp":True} }
        x0 = np.array(1.0)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200, iprint=10)
        print "minimum expected at ~", -0.1956
        print ret


