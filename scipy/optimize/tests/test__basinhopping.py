"""
Unit tests for the basin hopping global minimization algorithm.
"""
from __future__ import division, print_function, absolute_import
import copy

from numpy.testing import *
from numpy.testing.utils import *
import numpy as np
from numpy import cos, sin

from scipy.optimize import basinhopping, minimize
from scipy.optimize._basinhopping import Storage, RandomDisplacement, \
    Metropolis, AdaptiveStepsize


def func1d(x):
    f = cos(14.5 * x - 0.3) + (x + 0.2) * x
    df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2. * x + 0.2)
    return f, df


def func1d_nograd(x):
    f = cos(14.5 * x - 0.3) + (x + 0.2) * x
    df = np.array(-14.5 * sin(14.5 * x - 0.3) + 2. * x + 0.2)
    return f, df


def func2d_nograd(x):
    f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    return f


def func2d(x):
    f = cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
    df = np.zeros(2)
    df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    df[1] = 2. * x[1] + 0.2
    return f, df


class Minimizer(object):
    def __init__(self, func, **kwargs):
        self.kwargs = kwargs
        self.func = func

    def __call__(self, x0, **newkwargs):
        #combine the two kwargs
        kwargs = dict(list(newkwargs.items()) + list(self.kwargs.items()))
        res = minimize(self.func, x0, **kwargs)
        return res


class MyTakeStep1(RandomDisplacement):
    """use a copy of displace, but have it set a special parameter to
    make sure it's actually being used."""
    def __init__(self):
        self.been_called = False
        super(MyTakeStep1, self).__init__()

    def __call__(self, x):
        self.been_called = True
        return super(MyTakeStep1, self).__call__(x)


def myTakeStep2(x):
    """redo RandomDisplacement in function form without the attribute stepsize
    to make sure still everything works ok
    """
    s = 0.5
    x += np.random.uniform(-s, s, np.shape(x))
    return x


class MyAcceptTest(object):
    """pass a custom accept test

    This does nothing but make sure it's being used and ensure all the
    possible return values are accepted
    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0

    def __call__(self, **kwargs):
        self.been_called = True
        self.ncalls += 1
        if self.ncalls == 1:
            return False
        elif self.ncalls == 2:
            return 'force accept'
        else:
            return True


class MyCallBack(object):
    """pass a custom callback function

    This makes sure it's being used.  It also returns True after 10
    steps to ensure that it's stopping early.

    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0

    def __call__(self, x, f, accepted):
        self.been_called = True
        self.ncalls += 1
        if self.ncalls == 10:
            return True


class TestBasinHopping(TestCase):
    """ Tests for basinhopping """
    def setUp(self):
        """ Tests setup.

        run tests based on the 1-D and 2-D functions described above.  These
        are the same functions as used in the anneal algorithm with some
        gradients added.
        """
        self.x0 = (1.0, [1.0, 1.0])
        self.sol = (-0.195, np.array([-0.195, -0.1]))
        self.upper = (3., [3., 3.])
        self.lower = (-3., [-3., -3.])
        self.tol = 3  # number of decimal places

        self.niter = 100
        self.disp = False

        # fix random seed
        np.random.seed(1234)

        self.kwargs = {"method": "L-BFGS-B", "jac": True}
        self.kwargs_nograd = {"method": "L-BFGS-B"}

    def test_TypeError(self):
        # test the TypeErrors are raised on bad input
        i = 1
        # if take_step is passed, it must be callable
        assert_raises(TypeError, basinhopping, func2d, self.x0[i],
                          take_step=1)
        # if accept_test is passed, it must be callable
        assert_raises(TypeError, basinhopping, func2d, self.x0[i],
                          accept_test=1)
        # accept_test must return bool or string "force_accept"

        def bad_accept_test1(*args, **kwargs):
            return 1

        def bad_accept_test2(*args, **kwargs):
            return "not force_accept"
        assert_raises(ValueError, basinhopping, func2d, self.x0[i],
                          minimizer_kwargs=self.kwargs,
                          accept_test=bad_accept_test1)
        assert_raises(ValueError, basinhopping, func2d, self.x0[i],
                          minimizer_kwargs=self.kwargs,
                          accept_test=bad_accept_test2)

    def test_1d_grad(self):
        # test 1d minimizations with gradient
        i = 0
        res = basinhopping(func1d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_2d(self):
        # test 2d minimizations with gradient
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)
        assert_(res.nfev > 0)

    def test_njev(self):
        # test njev is returned correctly
        i = 1
        minimizer_kwargs = self.kwargs.copy()
        # L-BFGS-B doesn't use njev, but BFGS does
        minimizer_kwargs["method"] = "BFGS"
        res = basinhopping(func2d, self.x0[i],
                           minimizer_kwargs=minimizer_kwargs, niter=self.niter,
                           disp=self.disp)
        assert_(res.nfev > 0)
        assert_equal(res.nfev, res.njev)

    def test_2d_nograd(self):
        # test 2d minimizations without gradient
        i = 1
        res = basinhopping(func2d_nograd, self.x0[i],
                           minimizer_kwargs=self.kwargs_nograd,
                           niter=self.niter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_all_minimizers(self):
        # test 2d minimizations with gradient
        i = 1
        methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                   'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
        minimizer_kwargs = copy.copy(self.kwargs)
        for method in methods:
            minimizer_kwargs["method"] = method
            res = basinhopping(func2d, self.x0[i],
                               minimizer_kwargs=self.kwargs,
                               niter=self.niter, disp=self.disp)
            assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_pass_takestep(self):
        # test that passing a custom takestep works
        # also test that the stepsize is being adjusted
        takestep = MyTakeStep1()
        initial_step_size = takestep.stepsize
        i = 1
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp,
                           take_step=takestep)
        assert_almost_equal(res.x, self.sol[i], self.tol)
        assert_(takestep.been_called)
        # make sure that the built in adaptive step size has been used
        assert_(initial_step_size != takestep.stepsize)

    def test_pass_simple_takestep(self):
        # test that passing a custom takestep without attribute stepsize
        takestep = myTakeStep2
        i = 1
        res = basinhopping(func2d_nograd, self.x0[i],
                           minimizer_kwargs=self.kwargs_nograd,
                           niter=self.niter, disp=self.disp,
                           take_step=takestep)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_pass_accept_test(self):
        # test passing a custom accept test
        # makes sure it's being used and ensures all the possible return values
        # are accepted.
        accept_test = MyAcceptTest()
        i = 1
        # there's no point in running it more than a few steps.
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=10, disp=self.disp, accept_test=accept_test)
        assert_(accept_test.been_called)

    def test_pass_callback(self):
        # test passing a custom callback function
        # This makes sure it's being used.  It also returns True after 10 steps
        # to ensure that it's stopping early.
        callback = MyCallBack()
        i = 1
        # there's no point in running it more than a few steps.
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=30, disp=self.disp, callback=callback)
        assert_(callback.been_called)
        assert_("callback" in res.message[0])
        assert_equal(res.nit, 10)

    def test_minimizer_fail(self):
        # test if a minimizer fails
        i = 1
        self.kwargs["options"] = dict(maxiter=0)
        self.niter = 10
        self.disp = True
        res = basinhopping(func2d, self.x0[i], minimizer_kwargs=self.kwargs,
                           niter=self.niter, disp=self.disp)
        # the number of failed minimizations should be the number of
        # iterations + 1
        assert_equal(res.nit + 1, res.minimization_failures)


class Test_Storage(TestCase):
    def setUp(self):
        self.x0 = np.array(1)
        self.f0 = 0
        self.storage = Storage(self.x0, self.f0)

    def test_higher_f_rejected(self):
        ret = self.storage.update(self.x0 + 1, self.f0 + 1)
        x, f = self.storage.get_lowest()
        assert_equal(self.x0, x)
        assert_equal(self.f0, f)
        assert_(not ret)

    def test_lower_f_accepted(self):
        ret = self.storage.update(self.x0 + 1, self.f0 - 1)
        x, f = self.storage.get_lowest()
        assert_(self.x0 != x)
        assert_(self.f0 != f)
        assert_(ret)


class Test_RandomDisplacement(TestCase):
    def setUp(self):
        self.stepsize = 1.0
        self.displace = RandomDisplacement(stepsize=self.stepsize)
        self.N = 300000
        self.x0 = np.zeros([self.N])

    def test_random(self):
        # the mean should be 0
        # the variance should be (2*stepsize)**2 / 12
        # note these tests are random, they will fail from time to time
        x = self.displace(self.x0)
        v = (2. * self.stepsize) ** 2 / 12
        assert_almost_equal(np.mean(x), 0., 1)
        assert_almost_equal(np.var(x), v, 1)


class Test_Metropolis(TestCase):
    def setUp(self):
        self.T = 2.
        self.met = Metropolis(self.T)

    def test_boolean_return(self):
        # the return must be a bool.  else an error will be raised in
        # basinhopping
        ret = self.met(f_new=0., f_old=1.)
        assert isinstance(ret, bool)

    def test_lower_f_accepted(self):
        assert_(self.met(f_new=0., f_old=1.))

    def test_KeyError(self):
        # should raise KeyError if kwargs f_old or f_new is not passed
        assert_raises(KeyError, self.met, f_old=1.)
        assert_raises(KeyError, self.met, f_new=1.)

    def test_accept(self):
        # test that steps are randomly accepted for f_new > f_old
        one_accept = False
        one_reject = False
        for i in range(1000):
            if one_accept and one_reject:
                break
            ret = self.met(f_new=1., f_old=0.5)
            if ret:
                one_accept = True
            else:
                one_reject = True
        assert_(one_accept)
        assert_(one_reject)


class Test_AdaptiveStepsize(TestCase):
    def setUp(self):
        self.stepsize = 1.
        self.ts = RandomDisplacement(stepsize=self.stepsize)
        self.target_accept_rate = 0.5
        self.takestep = AdaptiveStepsize(takestep=self.ts, verbose=False,
                                          accept_rate=self.target_accept_rate)

    def test_adaptive_increase(self):
        # if few steps are rejected, the stepsize should increase
        x = 0.
        self.takestep(x)
        self.takestep.report(False)
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(True)
        assert_(self.ts.stepsize > self.stepsize)

    def test_adaptive_decrease(self):
        # if few steps are rejected, the stepsize should increase
        x = 0.
        self.takestep(x)
        self.takestep.report(True)
        for i in range(self.takestep.interval):
            self.takestep(x)
            self.takestep.report(False)
        assert_(self.ts.stepsize < self.stepsize)

    def test_all_accepted(self):
        # test that everything works OK if all steps were accepted
        x = 0.
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(True)
        assert_(self.ts.stepsize > self.stepsize)

    def test_all_rejected(self):
        # test that everything works OK if all steps were rejected
        x = 0.
        for i in range(self.takestep.interval + 1):
            self.takestep(x)
            self.takestep.report(False)
        assert_(self.ts.stepsize < self.stepsize)


if __name__ == "__main__":
    run_module_suite()
