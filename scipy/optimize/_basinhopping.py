"""
Original Author: Jacob Stevenson 2012

basinhopping: The basinhopping global optimization algorithm
"""

__all__ = ['basinhopping', 'basinhopping_advanced']

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
    def __init__(self, x0, minimizer, step_taking, accept_tests, callback=None,
                 iprint=1):
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.iprint = iprint

        self.nstep = 0
        self.takestep_report = True

        #do initial minimization
        minres = minimizer(self.x)
        self.x = np.copy(minres.x)
        self.energy = minres.fun
        if self.iprint > 0:
            print "basinhopping step %d: energy %g" % (self.nstep, self.energy)

        #initialize storage class
        self.callback = callback
        self.storage = _Storage(self.x, self.energy)

        #initialize return object
        self.res = scipy.optimize.Result()
        if hasattr(minres, "nfev"):
            self.res.nfev = minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev = minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev = minres.nhev

    def _monte_carlo_step(self):
        #Take a random step.  Make a copy of x because the step_taking
        #algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        #do a local minimization
        minres = self.minimizer(x_after_step)
        x_after_quench = minres.x
        energy_after_quench = minres.fun
        if hasattr(minres, "success"):
            if not minres.success:
                print "warning: basinhoppping: minimize failure"
        if hasattr(minres, "nfev"):
            self.res.nfev += minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev += minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev += minres.nhev

        #accept the move based on self.accept_tests f any test is false, than
        #reject the step, except if ny test returns the special value, the
        #string 'force accept'.  In this ccase we accept the step regardless.
        #This can be used to forcefully escape from a local minima if normal
        #basin hopping steps are not sufficient.
        accept = True
        for test in self.accept_tests:
            testres = test(f_new=energy_after_quench, x_new=x_after_quench,
                           f_old=self.energy, x_old=self.x)
            if isinstance(testres, bool):
                if not testres:
                    accept = False
            elif isinstance(testres, str):
                if testres == "force accept":
                    accept = True
                    break
                else:
                    raise ValueError(
                        "accept test must return bool or string 'force accept'. Type is",
                        type(testres)
                    )
            else:
                raise ValueError(
                    "accept test must return bool or string 'force accept'. Type is",
                    type(testres))

        #Report the result of the acceptance test to the take step class.  This
        #is for adaptive step taking
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(accept, f_new=energy_after_quench,
                                    x_new=x_after_quench, f_old=self.energy,
                                    x_old=self.x)

        return x_after_quench, energy_after_quench, accept

    def one_cycle(self):
        self.nstep += 1
        newmin = False

        xtrial, energy_trial, accept = self._monte_carlo_step()

        energy_old = self.energy
        if accept:
            self.energy = energy_trial
            self.x = np.copy(xtrial)
            newmin = self.storage.insert(self.x, self.energy)

        if callable(self.callback):
            #should we pass acopy of x?
            self.callback(self.x, self.energy, accept)

        if newmin and self.iprint > 0:
            print "found new global minimum on step %d with function value %g" \
                  % (self.nstep, self.energy)
        if self.iprint > 0:
            if self.nstep % self.iprint == 0:
                self.print_report(energy_trial, accept)

        return newmin

    def print_report(self, energy_trial, accept):
        xlowest, energy_lowest = self.storage.get_lowest()
        print "basinhopping step %d: energy %g trial_f %g accepted %d lowest_f %g" \
              % (self.nstep, self.energy, energy_trial, accept, energy_lowest)


class _AdaptiveStepsize(object):
    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
                 verbose=True):
        """
        Class to implement adaptive stepsize.  This class wraps the step taking
        class and modifies the stepsize to ensure the true acceptance rate is
        as close as possible to the target.

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
        self.verbose = verbose

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

    def report(self, accept, **kwargs):
        if accept:
            self.naccept += 1


class _RandomDisplacement(object):
    """
    Add a random displacement of maximum size, stepsize, to the coordinates

    update x inplace
    """
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        x += np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
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

    def accept_reject(self, energy_new, energy_old):
        w = min(1.0, np.exp(-(energy_new - energy_old) * self.beta))
        rand = np.random.rand()
        return w >= rand

    def __call__(self, **kwargs):
        """
        f_new and f_old are manditory in kwargs
        """
        return bool(self.accept_reject(kwargs["f_new"],
                    kwargs["f_old"]))


def basinhopping_advanced(x0, func=None, optimizer=None, minimizer=None,
                          minimizer_kwargs=dict(), take_step=None,
                          accept_test=None, callback=None, maxiter=10000,
                          T=1.0, stepsize=0.5, interval=50, disp=False,
                          niter_success=None):
    """
    Find the global minimum of a function using the basin hopping algorithm

    .. versionadded:: 0.12.0

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    func : callable ``f(x, *args)``, optional
        Function to be optimized.  Either func or minimizer must be passed.
        Use minimizer_kwargs to specify args.
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
            args - tuple, optional
                Extra arguments passed to the objective function (func) and its
                derivatives (Jacobian, Hessian). See description for func
                above.
            jac - specify the Jacobian for gradient minimizations
            hess - specify the Hessian for Hessian based minimizations
            tol - tolerance

    take_step : callable ``take_step(x)``, optional
        Replace the default step taking routine with this routine.
        The default step taking routine is a random displacement of the
        coordinates, but other step taking algorithms may be better for some
        systems.  take_step can optionally have two attributes

            take_step.stepsize - float
            take_step.report - callable, ``report(accept, f_new=f_new,
                                                  x_new=x_new, f_old=f_old,
                                                  x_old=x_old)``

        The function take_step.report() is called after each cycle and can be
        used to adaptively improve the routine.  In the above, f_new, x_new,
        f_old, and x_old are the new and old function value and coordinates,
        and accept is bool holding whether or not the new coordinates were
        accepted.  If take_step.report is not present and take_step.stepsize
        is, basinhopping will adjust take_step.stepsize in order to optimize
        the global minimum search.
    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold,
                                          x_old=x_old)``, optional
        Define a test which will be used to judge whether or not to accept
        the step.  This will be used in addition to the Metropolis test based
        on ``temperature`` T.  The acceptable return values are True, False, or
        "force accept".  If the latter, then this will overide any other tests
        in order to accept the step.  This can be used, for example, to
        forcefully escape from a local minimum that basinhopping is trapped in.
    callback : callable, ``callback(x, f, accept)``, optional
        Add a callback function which will be called each time a minima is
        accepted.  This can be used, for example, to save the lowest N minima
        found.
    maxiter : integer, optional
        The maximum number of basin hopping iterations
    T : float, optional
        The ``temperature`` parameter for the accept or reject criterion.
        Higher ``temperatures`` mean that larger jumps in function value will
        be accepted.  For best results T should be comparable to the separation
        (in function value) between local minima.
    stepsize : float, optional
        initial stepsize for use in the random displacement.
    interval : integer, optional
        interval for how often to update the stepsize
    disp : bool, optional
        Set to True to print status messages
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
    algorithm was originally described by David Wales and Jonathan Doye
    http://www-wales.ch.cam.ac.uk/ .  The algorithm is iterative with each
    iteration composed of the following steps

    1) random displacement of the coordinates

    2) local minimization

    3) accept or reject the new coordinates based on the minimized function
       value.

    The acceptance test is based on the Metropolis criterion of standard Monte
    Carlo integration.  This global minimization method has been shown to be
    extremely efficient on a wide variety of problems in physics and chemistry.
    It is especially efficient when the function has many minima separated by
    large barriers.  See the Cambridge Cluster Database
    http://www-wales.ch.cam.ac.uk/CCD.html for database of molecular systems
    that have been optimized primarily using basin hopping.  This database
    includes minimization problems exceeding 300 degrees of freedom.

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

    Examples
    --------
    The following example is a one dimensional minimization problem,  with many
    local minima superimposed on a parabola.

    >>> func = lambda x: cos(14.5 * x - 0.3) + (x + 0.2) * x
    >>> x0=[1.]

    Basinhopping, internally, uses a local minimization algorithm.  We will use
    the parameter minimizer_kwargs to tell basinhopping which algorithm to use
    and how to set up that minimizer.  This parameter will be passed to
    scipy.optimze.minimize()

    >>> minimizer_kwargs = {"method": "BFGS"}
    >>> ret = basinhopping(x0, func, minimizer_kwargs=minimizer_kwargs,
    ...                    maxiter=200)
    >>> print "global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun)
    global minimum: x = -0.1951, f(x0) = -1.0009

    Next consider a two dimensional minimization problem. Also, this time we
    will use gradient information to significantly speed up the search.

    >>> def func2d(x):
    ...     f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
    ...                                                         0.2) * x[0]
    ...     df = np.zeros(2)
    ...     df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    ...     df[1] = 2. * x[1] + 0.2
    ...     return f, df

    We'll also use a different minimizer, just for fun.  Also we must tell the
    minimzer that our function returns both energy and gradient (jacobian)
    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    >>> x0 = [1.0, 1.0]
    >>> ret = basinhopping_advanced(x0, func2d,
    ...                             minimizer_kwargs=minimizer_kwargs,
    ...                             maxiter=200)
    >>> print "global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun)
    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109


    Here is an example using a custom step taking routine.  Imagine you want
    the first coordinate to take larger steps then the rest of the coordinates.
    This can be implemented like so

    >>> class MyTakeStep(object):
    ...    def __init__(self, stepsize=0.5):
    ...        self.stepsize = stepsize
    ...    def __call__(self, x):
    ...        s = self.stepsize
    ...        x[0] += np.random.uniform(-2.*s, 2.*s)
    ...        x[1:] += np.random.uniform(-s, s, x[1:].shape)
    ...        return x

    Since MyTakeStep.stepsize exists, but MyTakeStep.report() doesn't,
    basinhopping_advanced will adjust the magnitude of stepsize to optimize the
    search.  We'll use the same 2d function as before

    >>> mytakestep = MyTakeStep()
    >>> ret = basinhopping_advanced(x0, func2d,
    ...                             minimizer_kwargs=minimizer_kwargs,
    ...                             maxiter=200, take_step=mytakestep)
    >>> print "global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun)
    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109


    Now let's do an example using a custom callback function which prints the
    value of every minimum found

    >>> def print_fun(x, f, accepted):
    ...         print "at minima %.4f accepted %d" % (f, int(accepted))

    We'll run it for only 10 basinhopping steps this time.

    >>> np.random.seed(1)
    >>> ret = basinhopping_advanced(x0, func2d,
    ...                             minimizer_kwargs=minimizer_kwargs,
    ...                             maxiter=10, callback=print_fun)
    at minima 0.4159 accepted 1
    at minima -0.9073 accepted 1
    at minima -0.1021 accepted 1
    at minima -0.1021 accepted 1
    at minima 0.9102 accepted 1
    at minima 0.9102 accepted 1
    at minima 0.9102 accepted 0
    at minima -0.1021 accepted 1
    at minima -1.0109 accepted 1
    at minima -1.0109 accepted 1

    The minima at -1.0109 is actually the global minimum, found already
    on the 4th iteration


    Now let's implement bounds on the problem using a custom accept_test

    >>> class MyBounds(object):
    ...     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
    ...         self.xmax = np.array(xmax)
    ...         self.xmin = np.array(xmin)
    ...     def __call__(self, **kwargs):
    ...         x = kwargs["x_new"]
    ...         tmax = bool(np.all(x <= self.xmax))
    ...         tmin = bool(np.all(x >= self.xmin))
    ...         return tmax and tmin

    >>> mybounds = MyBounds()
    >>> ret = basinhopping_advanced(x0, func2d,
    ...                             minimizer_kwargs=minimizer_kwargs,
    ...                             maxiter=10, accept_test=mybounds)


    """
    x0 = np.array(x0)

    #turn printing on or off
    if disp:
        iprint = 1
    else:
        iprint = -1

    #set up minimizer
    if minimizer is None and func is None:
        raise ValueError("minimizer and func cannot both be None")
    if callable(minimizer):
        wrapped_minimizer = _MinimizerWrapper(minimizer, **minimizer_kwargs)
    else:
        #use default
        wrapped_minimizer = _MinimizerWrapper(scipy.optimize.minimize, func,
                                              **minimizer_kwargs)

    #set up step taking algorithm
    verbose = iprint > 0
    if take_step is not None:
        if not callable(take_step):
            raise ValueError("take_step must be callable")
        # if take_step.stepsize exists, but take_step.report() doesn't, then
        # then use _AdaptiveStepsize to control take_step.stepsize
        if hasattr(take_step, "stepsize") and not hasattr(take_step, "report"):
            mytake_step = _AdaptiveStepsize(take_step, interval=interval,
                                            verbose=verbose)
        else:
            mytake_step = take_step
    else:
        #use default
        displace = _RandomDisplacement(stepsize=stepsize)
        mytake_step = _AdaptiveStepsize(displace, interval=interval,
                                        verbose=verbose)

    #set up accept tests
    if accept_test is not None:
        if not callable(accept_test):
            raise ValueError("accept_test must be callable")
        accept_tests = [accept_test]
    else:
        accept_tests = []
    ##use default
    metropolis = _Metropolis(T)
    accept_tests.append(metropolis)

    if niter_success is None:
        niter_success = maxiter + 2

    bh = _BasinHopping(x0, wrapped_minimizer, mytake_step, accept_tests,
                       callback=callback, iprint=iprint)

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

    #prepare return object
    lowest = bh.storage.get_lowest()
    res = bh.res
    res.x = np.copy(lowest[0])
    res.fun = lowest[1]
    res.message = message
    return res


def basinhopping(x0, func=None, optimizer=None, minimizer=None,
                 minimizer_kwargs=dict(), maxiter=10000, T=1.0, stepsize=0.5,
                 interval=50, disp=False, niter_success=None):
    """
    Find the global minimum of a function using the basin hopping algorithm

    .. versionadded:: 0.12.0

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    func : callable ``f(x, *args)``, optional
        Function to be optimized.  Either func or minimizer must be passed.
        Use minimizer_kwargs to specify args.
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
            args - tuple, optional
                Extra arguments passed to the objective function and its
                derivatives (Jacobian, Hessian). See description for func
                above.
            jac - specify the Jacobian for gradient minimizations
            hess - specify the Hessian for Hessian based minimizations
            tol - tolerance

    maxiter : integer, optional
        The maximum number of basin hopping iterations
    T : float, optional
        The ``temperature`` parameter for the accept or reject criterion.
        Higher ``temperatures`` mean that larger jumps in function value will
        be accepted.  For best results T should be comparable to the separation
        (in function value) between local minima.
    stepsize : float, optional
        initial stepsize for use in the random displacement.
    interval : integer, optional
        interval for how often to update the stepsize
    disp : bool, optional
        Set to True to print status messages
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


    Examples
    --------
    The following example is a one dimensional minimization problem,  with many
    local minima superimposed on a parabola.

    >>> func = lambda x: cos(14.5 * x - 0.3) + (x + 0.2) * x
    >>> x0=[1.]

    Basinhoppin, internally, uses a local minimization algorithm.  We will use
    the parameter minimizer_kwargs to tell basinhopping which algorithm to use
    and how to set up that minimizer.  This parameter will be passed to
    scipy.optimze.minimize()

    >>> minimizer_kwargs = {"method": "BFGS"}
    >>> ret = basinhopping(x0, func, minimizer_kwargs=minimizer_kwargs,
    ...                    maxiter=200)
    >>> print "global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun)
    global minimum: x = -0.1951, f(x0) = -1.0009

    Next consider a two dimensional minimization problem. Also, this time we
    will use gradient information to significantly speed up the search.

    >>> def func2d(x):
    ...     f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
    ...                                                         0.2) * x[0]
    ...     df = np.zeros(2)
    ...     df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    ...     df[1] = 2. * x[1] + 0.2
    ...     return f, df

    We'll also use a different minimizer, just for fun.  Also we must tell the
    minimzer that our function returns both energy and gradient (jacobian)
    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    >>> x0 = [1.0, 1.0]
    >>> ret = basinhopping(x0, func2d, minimizer_kwargs=minimizer_kwargs,
    ...                    maxiter=200)
    >>> print "global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun)
    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109

    """
    x0 = np.array(x0)

    #turn printing on or off
    if disp:
        iprint = 1
    else:
        iprint = -1

    #set up minimizer
    if minimizer is None and func is None:
        raise ValueError("minimizer and func cannot both be None")
    if callable(minimizer):
        wrapped_minimizer = _MinimizerWrapper(minimizer, **minimizer_kwargs)
    else:
        #use default
        wrapped_minimizer = _MinimizerWrapper(scipy.optimize.minimize, func,
                                              **minimizer_kwargs)

    #set up step taking algorithm
    if True:
        #use default
        displace = _RandomDisplacement(stepsize=stepsize)
        verbose = iprint > 0
        step_taking = _AdaptiveStepsize(displace, interval=interval,
                                        verbose=verbose)

    #set up accept tests
    if True:
        ##use default
        metropolis = _Metropolis(T)
        accept_tests = [metropolis]

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

    #prepare return object
    lowest = bh.storage.get_lowest()
    res = bh.res
    res.x = np.copy(lowest[0])
    res.fun = lowest[1]
    res.message = message
    return res


if __name__ == "__main__":
    if True:
        print ""
        print ""
        print "minimize a 1d function with gradient"

        def func(x):
            f = cos(14.5 * x - 0.3) + (x + 0.2) * x
            df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2. * x + 0.2)
            return f, df

        # minimum expected at ~-0.195
        kwargs = {"method": "L-BFGS-B", "jac": True}
        x0 = np.array(1.0)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200,
                           disp=False)
        print "minimum expected at ~", -0.195
        print ret

    if True:
        print ""
        print ""
        print "minimize a 2d function without gradient"
        # minimum expected at ~[-0.195, -0.1]

        def func(x):
            f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
                                                                0.2) * x[0]
            return f

        kwargs = {"method": "L-BFGS-B"}
        x0 = np.array([1.0, 1.])
        scipy.optimize.minimize(func, x0, **kwargs)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200,
                           disp=False)
        print "minimum expected at ~", [-0.195, -0.1]
        print ret

    if True:
        print ""
        print ""
        print "minimize a 1d function with large barriers"
        #try a function with much higher barriers between the local minima.

        def func(x):
            f = 5. * cos(14.5 * x - 0.3) + 2. * (x + 0.2) * x
            df = np.array(-5. * 14.5 * sin(14.5 * x - 0.3) + 2. * (2. * x +
                                                                   0.2))
            return f, df

        # minimum expected at ~-0.195
        kwargs = {"method": "L-BFGS-B", "jac": True}
        x0 = np.array(1.0)
        ret = basinhopping(x0, func, minimizer_kwargs=kwargs, maxiter=200,
                           disp=False)
        print "minimum expected at ~", -0.1956
        print ret

    if False:
        func = lambda x: cos(14.5 * x - 0.3) + (x + 0.2) * x
        x0 = [1.]
        ret = basinhopping(x0, func, maxiter=200, disp=False)
        print "minimum expected at ~", -0.195
        print ret

    if True:
        print ""
        print ""
        print "try a harder 2d problem"

        def func2d(x):
            f = (cos(14.5 * x[0] - 0.3) + (x[0] + 0.2) * x[0] +
                 cos(14.5 * x[1] - 0.3) + (x[1] + 0.2) * x[1] + x[0] * x[1])
            df = np.zeros(2)
            df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2 + x[1]
            df[1] = -14.5 * sin(14.5 * x[1] - 0.3) + 2. * x[1] + 0.2 + x[0]
            return f, df

        kwargs = {"method": "L-BFGS-B", "jac": True}
        x0 = np.array([1.0, 1.0])
        ret = basinhopping(x0, func2d, minimizer_kwargs=kwargs, maxiter=200,
                           disp=True)
        print "minimum expected at ~", [-0.19415263, -0.19415263]
        print ret
