#Original Author: Jacob Stevenson 2012
#the basinhopping global optimization algorithm

__all__ = ['basinhopping']

import numpy as np
from numpy import cos, sin
import scipy.optimize


#
#notes for the basin hopping algorithm
#
#The important components of basin hopping are:
#
#    step taking algorithm:
#        The default for this should be random displacement with adaptive step
#        size.
#
#        The user should be able to specify their own step taking algorithm.
#        question:  how do we combine adaptive step size with user defined step
#        taking?
#
#    optimizer:
#        This is the routine which does local minimization.  The default should
#        be scipy lbfgs algorithm
#        
#        The user should be able to specify their own minimizer.
#
#    accept test:
#        The default will be metropolis criterion with temperature T=1.
#
#        This is where we can implement bounds on the problem.
#
#        Should be able to add their own accept or reject test?
#
#    Storage:
#        A class for storing the best structure found
#
#        possibly also the best N structures found?

# todo: 
#   make the step taking algorithm passable?

class _Storage(object):
    def __init__(self, x, f):
        """
        Class used to store the lowest energy structure
        """
        self._add(x, f)
    def _add(self, x, f):
        self.x = np.copy(x)
        self.f = f
    def insert(self, x, f):
        if f < self.f:
            self._add(x, f)
            return True
        else:
            return False
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
        self.takestep_report = True

        #do initial minimization
        res = minimizer(self.x)
        self.x = np.copy(res.x)
        self.energy = res.fun
        print "basinhopping step %d: energy %g" % (self.nstep, self.energy)

        #initialize storage class
        self.storage = _Storage(self.x, self.energy)
        
        #initialize return object
        self.res = scipy.optimize.Result()
        if hasattr(res, "nfev"):
            self.res.nfev = res.nfev
        if hasattr(res, "njev"):
            self.res.njev = res.njev
        if hasattr(res, "nhev"):
            self.res.nhev = res.nhev


    def _monte_carlo_step(self):
        #Take a random step.  Make a copy of x because the step_taking
        #algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        #do a local minimization
        res = self.minimizer(x_after_step)
        x_after_quench = res.x
        energy_after_quench = res.fun
        if hasattr(res, "success"):
            if not res.success:
                print "warning: basinhoppping: minimize failure"
        if hasattr(res, "nfev"):
            self.res.nfev += res.nfev
        if hasattr(res, "njev"):
            self.res.njev += res.njev
        if hasattr(res, "nhev"):
            self.res.nhev += res.nhev


        #accept the move based on self.accept_tests
        #if any one of accept test is false, than reject the step
        accept = True
        for test in self.accept_tests:
            if not test(enew=energy_after_quench, xnew=x_after_quench,
                    eold=self.energy, xold=self.x):
                accept = False
                break

        #Report the result of the acceptance test to the take step class.  This
        #is for adaptive step taking
        if self.takestep_report:
            self.step_taking.report(accept)

        return x_after_quench, energy_after_quench, accept

    def one_cycle(self):
        self.nstep += 1
        newmin = False

        xtrial, etrial, accept = self._monte_carlo_step()

        eold = self.energy
        if accept:
            self.energy = etrial
            self.x = np.copy(xtrial)
            newmin = self.storage.insert( self.x, self.energy )

        if newmin and self.iprint > 0:
            print "found new global minimum on step %d with function value %g" \
                    % (self.nstep, self.energy)
        if self.iprint > 0:
            if self.nstep % self.iprint == 0:
                self.print_report(etrial, accept)

        return newmin

    def print_report(self, etrial, accept):
        xlowest, elowest = self.storage.get_lowest()
        print "basinhopping step %d: energy %g trial_f %g accepted %d lowest_f %g" \
                % (self.nstep, self.energy, etrial, accept, elowest)

class AdaptiveStepsize(object):
    def __init__( self, takestep, accept_rate=0.5, interval = 50, factor = 0.9,
            verbose=True ):
        """
        Class to implement adaptive stepsize.  The step size used by class
        takestep is modified to ensure the true acceptance rate is as close as
        possible to the target.

        .. versionadded:: 0.13.0

        Parameters
        ----------
        takestep : callable
            The step taking routine.  Must contain modifiable attribute
            takestep.stepsize
        accept_rate : float, optional
            The target step acceptance rate
        interval : integer, optional
            Interval for how often to update the stepsize
        factor : float, optional
            The step size is multiplied or divided by this factor upon each
            update. 
        verbose : bool, optional
            Print information about each update

        """
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
            #We're accepting too many steps.  This generally means we're
            #trapped in a basin.  Take bigger steps 
            self.takestep.stepsize /= self.factor
        else:
            #We're not accepting enough steps.  Take smaller steps 
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print "adaptive stepsize: acceptance rate %f target %f new stepsize %g old stepsize %g" \
                    % (accept_rate, self.target_accept_rate,
                            self.takestep.stepsize, old_stepsize)
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
    """
    Add a random displacement of maximum size, stepsize, to the coordinates

    update x inplace
    """
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




def basinhopping(x0, func=None, args=(), optimizer=None, minimizer=None,
        minimizer_kwargs=dict(), maxiter=10000, T=1.0, stepsize=0.5,
        interval=50, iprint=-1, niter_success=None):
    """
    Find the global minimum of a function using the basin hopping algorithm

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    func : callable ``f(x, *args)``, optional
        Function to be optimized.  Either func or minimizer must be passed
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (Jacobian, Hessian).
    minimizer : callable ``minimizer(x0, **minimizer_kwargs)``, optional
        Use this minizer rather than the default.  If the minimizer is given
        then func is not used.  basinhopping() will get the function values
        from the output of minimizer.  The output must be an object with
        attributes x and fun reporting the minimized coordinates and function
        value
    minimizer_kwargs : tuple, optional
        Extra arguments to be passed to the minimizer.  If argument minimizer
        is specified, then it is passed to that, else it is passed to the
        default scipy.optimize.minimize().  See scipy.optimize.minimize() for
        details.  If the default minimzer is used, some important options could
        be

            method - the minimization method
            jac - specify the Jacobian for gradient minimizations
            hess - specify the Hessian for Hessian based minimizations
            tol - tolerance
            
    maxiter : integer, optional
        The maximum number of basin hopping iterations
    T : float, optional
        The ``temperature`` parameter for the accept or reject criterion.
        Higher ``temperatures`` mean that larger jumps in function value will
        be accepted
    stepsize : float, optional
        initial stepsize for use in the random displacement.
    interval : integer, optional
        interval for how often to update the stepsize
    iprint : integer, optional
        The interval at which to print status information.  iprint < 0 for a
        silent run
    niter_success : integer, optional
        Stop the run if the global minimum candidate remains the same for this
        number of iterations.


    Returns
    -------
    res : Result
        The optimization result represented as a ``Result`` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination. See `Result` for a description of other
        attributes.

    Notes
    -----
    Basin hopping is a random algorithm which attempts to find the global
    minimum of a smooth scalar function of one or more variables.  The
    algorithm was originally described by David Wales
    http://www-wales.ch.cam.ac.uk/ .  The algorithm is iterative with each
    iteration composed of the following steps

    1) random displacement of the coordinates

    2) local minimization

    3) accept or reject the new coordinates based on the minimized function
    value.

    This global minimization method has been shown to be extremely efficient on
    a wide variety of problems in physics and chemistry.  It is especially
    efficient when the function has many minima separated by large barriers.
    See the Cambridge Cluster Database http://www-wales.ch.cam.ac.uk/CCD.html
    for database of molecular systems that have been optimized primarily using
    basin hopping.  This database includes minimization problems exceeding
    300 degrees of freedom.

    For global minimization problems there's no general way to know that you've
    found the global solution.  The standard way is to run the algorithm until
    the lowest minimum found stops changing.

    References
    ----------
    .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
        Cambridge, UK
    .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
        the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
        110 Atoms.  Journal of Physical Chemistry A, 1997, 101 (28), pp
        5111-5116


    """
    x0 = np.array(x0)

    #set up minimizer
    if minimizer is None and func is None:
        raise ValueError("minimizer and func cannot both be None")
    if callable(minimizer):
        wrapped_minimizer = _MinimizerWrapper(minimizer, **minimizer_kwargs)
    else:
        #use default
        if len(args) > 0:
            #should we be worried about overwriting?
            minimizer_kwargs["args"] = args
        wrapped_minimizer = _MinimizerWrapper(
                scipy.optimize.minimize, func, **minimizer_kwargs)
        
    #set up step taking algorithm
    if True:
        #use default
        displace = RandomDisplacement(stepsize=stepsize)
        verbose = iprint > 0
        step_taking = AdaptiveStepsize(displace, interval=interval, 
                verbose=verbose)

    #set up accept tests
    if True:
        ##use default
        metropolis = _Metropolis(T) 
        accept_tests = [ metropolis ]

    if niter_success is None:
        niter_success = maxiter + 2

    bh = _BasinHopping(x0, wrapped_minimizer, step_taking, accept_tests,
            iprint=iprint)

    #start main iteration loop
    count = 0
    message = ["maximum iterations reached"]
    for i in range(maxiter):
        newmin = bh.one_cycle()
        count += 1
        if newmin:
            count = 0
        elif count > niter_success:
            message = ["success condition satisfied"]
            break

    #finished.

    lowest = bh.storage.get_lowest()
    res = bh.res
    res.x = np.copy(lowest[0])
    res.fun = lowest[1]
    return res



if __name__ == "__main__":

    if True:
        print ""
        print ""
        print "minimize a 1d function with gradient"
        def func(x):
            f =  cos(14.5*x-0.3) + (x+0.2)*x
            df = np.array(-14.5*sin(14.5*x-0.3) + 2.*x + 0.2)
            return f, df
        # minimum expected at ~-0.195
        kwargs={ "method": "L-BFGS-B", "jac": True }
        x0 = np.array(1.0)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200,
                iprint=10)
        print "minimum expected at ~", -0.195
        print ret

    if True:
        print ""
        print ""
        print "minimize a 2d function without gradient"
        # minimum expected at ~[-0.195, -0.1]
        def func(x):
            f = cos(14.5*x[0]-0.3) + (x[1]+0.2)*x[1] + (x[0]+0.2)*x[0]
            return f
        kwargs={ "method": "L-BFGS-B"}
        x0 = np.array([1.0,1.])
        scipy.optimize.minimize(func, x0, **kwargs)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200,
                iprint=10)
        print "minimum expected at ~", [-0.195, -0.1]
        print ret

    if True:
        print ""
        print ""
        print "minimize a 1d function with large barriers"
        #try a function with much higher barriers between the local minima.
        def func(x):
            f =  5.*cos(14.5*x-0.3) + 2.*(x+0.2)*x
            df = np.array(-5.*14.5*sin(14.5*x-0.3) + 2.*(2.*x + 0.2))
            return f, df
        # minimum expected at ~-0.195
        kwargs={ "method": "L-BFGS-B", "jac": True }
        x0 = np.array(1.0)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200,
                iprint=10)
        print "minimum expected at ~", -0.1956
        print ret


