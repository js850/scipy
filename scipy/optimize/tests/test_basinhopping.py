"""
Unit tests for the basin hopping global minimization algorithm.
"""

from numpy.testing import TestCase, run_module_suite, \
    assert_almost_equal, assert_, dec

import numpy as np
from numpy import cos, sin

from scipy.optimize import basinhopping, minimize

def func1d(x):
    f =  cos(14.5*x-0.3) + (x+0.2)*x
    df = np.array(-14.5*sin(14.5*x-0.3) + 2.*x + 0.2)
    return f, df
def func1d_nograd(x):
    f =  cos(14.5*x-0.3) + (x+0.2)*x
    df = np.array(-14.5*sin(14.5*x-0.3) + 2.*x + 0.2)
    return f, df
def func2d_nograd(x):
    f = cos(14.5*x[0]-0.3) + (x[1]+0.2)*x[1] + (x[0]+0.2)*x[0]
    return f
def func2d(x):
    f = cos(14.5*x[0]-0.3) + (x[1]+0.2)*x[1] + (x[0]+0.2)*x[0]
    df = np.zeros(2)
    df[0] = -14.5*sin(14.5*x[0]-0.3) + 2.*x[0] + 0.2
    df[1] = 2.*x[1] + 0.2
    return f, df

class Minimizer(object):
    def __init__(self, func, **kwargs):
        self.kwargs = kwargs
        self.func = func
    def __call__(self, x0, **newkwargs):
        #combine the two kwargs
        kwargs = dict( newkwargs.items() + self.kwargs.items() )
        res = minimize(self.func, x0, **kwargs)
        return res


class TestAnneal(TestCase):
    """ Tests for anneal """
    def setUp(self):
        """ Tests setup.
        """
        self.x0 = (1.0, [1.0, 1.0])
        self.sol = (-0.195, np.array([-0.195, -0.1]))
        self.upper = (3., [3., 3.])
        self.lower = (-3., [-3., -3.])
        self.tol = 3 #number of decimal places


        # 'fast' and 'cauchy' succeed with maxiter=1000 but 'boltzmann'
        # exits with status=3 until very high values. Keep this value
        # reasonable though.
        self.maxiter = 100
        self.iprint = -1

        # fix random seed
        np.random.seed(1234)

        self.kwargs={ "method": "L-BFGS-B", "jac": True } #, "options":{"disp":True} }
        self.kwargs_nograd={ "method": "L-BFGS-B" } #, "options":{"disp":True} }


    @dec.slow
    def test_1d_grad(self, use_wrapper=False):
        i = 0
        res = basinhopping(self.x0[i], func1d, minimizer_kwargs=self.kwargs, maxiter=self.maxiter, iprint=self.iprint)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    @dec.slow
    def test_2d(self, use_wrapper=False):
        i = 1
        res = basinhopping(self.x0[i], func2d, minimizer_kwargs=self.kwargs, maxiter=self.maxiter, iprint=self.iprint)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    @dec.slow
    def test_2d_nograd(self, use_wrapper=False):
        i = 1
        res = basinhopping(self.x0[i], func2d_nograd, minimizer_kwargs=self.kwargs_nograd, maxiter=self.maxiter, iprint=self.iprint)
        assert_almost_equal(res.x, self.sol[i], self.tol)

    @dec.slow
    def test_pass_minimizer(self, use_wrapper=False):
        i=1
        minimizer = Minimizer(func2d, **self.kwargs)
        res = basinhopping(self.x0[i], minimizer=minimizer, maxiter=self.maxiter, iprint=self.iprint)
        assert_almost_equal(res.x, self.sol[i], self.tol)



if __name__ == "__main__":
    run_module_suite()
