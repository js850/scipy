"""
Unit tests for the basin hopping global minimization algorithm.
"""
import copy

from numpy.testing import TestCase, run_module_suite, \
    assert_almost_equal, assert_, dec
import numpy as np
from numpy import cos, sin

from scipy.optimize import basinhopping, minimize


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
        kwargs = dict(newkwargs.items() + self.kwargs.items())
        res = minimize(self.func, x0, **kwargs)
        return res


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

        self.maxiter = 100
        self.disp = False

        # fix random seed
        np.random.seed(1234)

        self.kwargs = {"method": "L-BFGS-B", "jac": True}
        self.kwargs_nograd = {"method": "L-BFGS-B"}

    def test_1d_grad(self):
        """test 1d minimizations with gradient"""
        i = 0
        res = basinhopping(self.x0[i], func1d, minimizer_kwargs=self.kwargs,
                           maxiter=self.maxiter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_2d(self):
        """test 2d minimizations with gradient"""
        i = 1
        res = basinhopping(self.x0[i], func2d, minimizer_kwargs=self.kwargs,
                           maxiter=self.maxiter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_2d_nograd(self):
        """test 2d minimizations without gradient"""
        i = 1
        res = basinhopping(self.x0[i], func2d_nograd,
                           minimizer_kwargs=self.kwargs_nograd,
                           maxiter=self.maxiter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_pass_minimizer(self):
        """test 2d minimizations with user defined minimizer"""
        i = 1
        minimizer = Minimizer(func2d, **self.kwargs)
        res = basinhopping(self.x0[i], minimizer=minimizer,
                           maxiter=self.maxiter, disp=self.disp)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    def test_all_minimizers(self):
        """test 2d minimizations with gradient"""
        i = 1
        methods = [ 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
        minimizer_kwargs = copy.copy(self.kwargs)
        for method in methods:
            minimizer_kwargs["method"] = method
            res = basinhopping(self.x0[i], func2d, minimizer_kwargs=self.kwargs,
                               maxiter=self.maxiter, disp=self.disp)
            assert_almost_equal(res.x, self.sol[i], self.tol)

if __name__ == "__main__":
    run_module_suite()
