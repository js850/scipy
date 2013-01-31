"""
benchmarks for the basinhopping global optimization algorithm

these functions were taken from 

http://en.wikipedia.org/wiki/Test_functions_for_optimization

I have chosen functions that seemed appropriate for basinhopping.  These were
functions that were continuous and (mostly) smooth, and had many competing local
minima.

There is also the Lennard Jones potential, which was not taken from the above
website, but is a common potential in the fields of physics and chemistry.  The 
global minimum energies are taken from the cambridge cluster database
http://www-wales.ch.cam.ac.uk/CCD.html

"""
from numpy import exp, cos, sqrt, pi, sin
import numpy as np

from scipy.optimize import basinhopping, anneal

class BenchmarkSystem(object):
    Etol = 1e-4
    def __init__(self, potential):
        self.pot = potential
        
    def get_random_configuration(self):
        if hasattr(self.pot, "get_random_configuration"):
            return self.pot.get_random_configuration()
        xmin, xmax = self.pot.xmin, self.pot.xmax
        x = np.random.uniform(xmin[0] + .01, xmax[0] - .01)
        y = np.random.uniform(xmin[1] + .01, xmax[1] - .01)
        return np.array([x,y])
    
    def accept_test(self, x_new=None, *args, **kwargs):
        if not hasattr(self.pot, "xmin"): return True
        if np.any(x_new < self.pot.xmin):
            return False
        if np.any(x_new > self.pot.xmax):
            return False
        return True
    
    def stop_criterion(self, coords, E, accepted):
        if accepted and E < self.pot.target_E + self.Etol:
            return True
        else:
            return False

    def do_benchmark_no_gradient(self, **kwargs):
#        kwargs = {}
        if hasattr(self.pot, "temperature"):
            kwargs["T"] = self.pot.temperature
        if hasattr(self.pot, "stepsize"):
            kwargs["stepsize"] = self.pot.stepsize

        
        minimizer_kwargs = {"method":"L-BFGS-B"}
#        minimizer_kwargs = {"method":"BFGS"}
        x0 = self.get_random_configuration()
        ret = basinhopping(self.pot.getEnergy, x0, accept_test=self.accept_test, 
                           callback=self.stop_criterion, niter=1000,
                           minimizer_kwargs=minimizer_kwargs,
                           **kwargs)
        print ret
        print "target energy", self.pot.target_E
        print "lowest energy found", ret.fun
        
    def do_benchmark(self, **kwargs):
#        kwargs = {}
        if hasattr(self.pot, "temperature"):
            kwargs["T"] = self.pot.temperature
        if hasattr(self.pot, "stepsize"):
            kwargs["stepsize"] = self.pot.stepsize

        
        minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
        x0 = self.get_random_configuration()
        ret = basinhopping(self.pot.getEnergyGradient, x0, accept_test=self.accept_test, 
                           callback=self.stop_criterion, niter=1000,
                           minimizer_kwargs=minimizer_kwargs,
                           **kwargs)        
        print ret
        print "target energy", self.pot.target_E
        print "lowest energy found", ret.fun
        
    def do_benchmark_anneal(self, **kwargs):
        x0 = self.get_random_configuration()
        ret = anneal(self.pot.getEnergy, x0, maxeval=3000)
        x = ret[0]
        print "target energy", self.pot.target_E
        print "lowest energy found", self.pot.getEnergy(x)
        print "coordinates", x
#        print ret

class Ackey(object):
    #note: this function is not smooth at the origin.  the gradient will never
    #converge in the minimizer
    target_E = 0.
    xmin = np.array([-5,-5])
    xmax = np.array([5,5])
    def getEnergy(self, coords):
        x, y = coords
        E = -20. * exp(-0.2 * sqrt(x**2 + y**2)) - \
            exp(0.5 * (cos(2. *pi * x) + cos(2. * pi * y))) + 20. + np.e
        return E
    
    def getEnergyGradient(self, coords):
        E = self.getEnergy(coords)
        x, y = coords
        R = sqrt(x**2 + y**2)
        term1 = -20. * exp(-0.2 * R)
        term2 = -exp(0.5 * (cos(2. *pi * x) + cos(2. * pi * y)))
        
        deriv1 = term1 * (-0.2 * 0.5 / R)
        
        dEdx = 2.* deriv1 * x  - term2 * pi * sin(2.*pi*x)
        dEdy = 2.* deriv1 * y  - term2 * pi * sin(2.*pi*y)
        
        return E, np.array([dEdx, dEdy])

class Levi(object):
    target_E = 0.
    xmin = np.array([-10,-10])
    xmax = np.array([10,10])

    def getEnergy(self, coords):
        x, y = coords
        E = sin(3.*pi*x)**2 + (x-1.)**2 * (1. + sin(3*pi*y)**2) \
            + (y-1.)**2 * (1. + sin(2*pi*y)**2)
        return E
    
    def getEnergyGradient(self, coords):
        x, y = coords
        E = self.getEnergy(coords)
        
        dEdx = 2.*3.*pi* cos(3.*pi*x) * sin(3.*pi*x) + 2.*(x-1.) * (1. + sin(3*pi*y)**2)
        
        dEdy = (x-1.)**2 * 2.*3.*pi* cos(3.*pi*y) * sin(3.*pi*y) + 2. *  (y-1.) * (1. + sin(2*pi*y)**2) \
            + (y-1.)**2 * 2.*2.*pi * cos(2.*pi*y) * sin(2.*pi*y)
        
        return E, np.array([dEdx, dEdy])

class HolderTable(object):
    target_E = -19.2085
    xmin = np.array([-10,-10])
    xmax = np.array([10,10])
    stepsize = 2.
    temperature = 2.
    def getEnergy(self, coords):
        x, y = coords
        E = - abs(sin(x)* cos(y) * exp(abs(1. - sqrt(x**2 + y**2)/ pi)))
        return E
    
    def dabs(self, x):
        """derivative of absolute value"""
        if x < 0: return -1.
        elif x > 0: return 1.
        else: return 0.

    def getEnergyGradient(self, coords):
        x, y = coords
        R = sqrt(x**2 + y**2)
        g = 1. - R / pi
        f = sin(x)* cos(y) * exp(abs(g))
        E = -abs(f)
        
        
        dRdx = x / R
        dgdx = - dRdx / pi
        dfdx = cos(x) * cos(y) * exp(abs(g)) + f * self.dabs(g) * dgdx
        dEdx = - self.dabs(f) * dfdx
        
        dRdy = y / R
        dgdy = - dRdy / pi
        dfdy = -sin(x) * sin(y) * exp(abs(g)) + f * self.dabs(g) * dgdy        
        dEdy = - self.dabs(f) * dfdy
        return E, np.array([dEdx, dEdy])

class LennardJones(object):
    """
    The Lennard Jones potential
    
    a mathematically simple model that approximates the interaction between a 
    pair of neutral atoms or molecules.    
    http://en.wikipedia.org/wiki/Lennard-Jones_potential
    
    E = sum_ij V(r_ij)
    
    where r_ij is the cartesian distance between atom i and atom j, and the
    pair potential has the form
    
    V(r) = 4 * eps * ( (sigma / r)**12 - (sigma / r)**6
    
    Notes
    -----
    the double loop over many atoms makes this *very* slow in Python.  If it
    were in a compiled language it would be much faster.
    """
    def __init__(self, eps=1.0, sig=1.0):
        self.sig = sig
        self.eps = eps

    def vij(self, r):
        return 4.*self.eps * ( (self.sig/r)**12 - (self.sig/r)**6 )

    def dvij(self, r):
        return 4.*self.eps * ( -12./self.sig*(self.sig/r)**13 + 6./self.sig*(self.sig/r)**7 )

    def getEnergy(self, coords):
        natoms = coords.size/3
        coords = np.reshape(coords, [natoms,3])
        energy=0.
        for i in xrange(natoms):
            for j in xrange(i+1,natoms):
                dr = coords[j,:]- coords[i,:]
                r = np.linalg.norm(dr)
                energy += self.vij(r)
        return energy

    def getEnergyGradient(self, coords):
        natoms = coords.size/3
        coords = np.reshape(coords, [natoms,3])
        energy=0.
        grad = np.zeros([natoms,3])
        for i in xrange(natoms):
            for j in xrange(i+1,natoms):
                dr = coords[j,:]- coords[i,:]
                r = np.linalg.norm(dr)
                energy += self.vij(r)
                g = self.dvij(r)
                grad[i,:] += -g * dr/r
                grad[j,:] += g * dr/r
        grad = grad.reshape([natoms*3])
        return energy, grad

    def get_random_configuration(self):
        return np.random.uniform(-1,1,[3*self.natoms]) * float(self.natoms)**(1./3) 

class LJ38(LennardJones):
    natoms = 38
    target_E = -173.928427

class LJ30(LennardJones):
    natoms = 30
    target_E = -128.286571

class LJ20(LennardJones):
    natoms = 20
    target_E = -77.177043

class LJ13(LennardJones):
    natoms = 13
    target_E = -44.326801


def benchmark_basinhopping():
    print ""
    print "doing benchmark for Ackey function"
    mysys = BenchmarkSystem(Ackey())
    mysys.do_benchmark()

    print ""
    print "doing benchmark for Levi function"
    mysys = BenchmarkSystem(Levi())
    mysys.do_benchmark()

    print ""
    print "doing benchmark for Holder Table function"
    mysys = BenchmarkSystem(HolderTable())
    mysys.do_benchmark()

    print ""
    print "doing benchmark for a cluster of 13 Lennard Jones atoms"
    mysys = BenchmarkSystem(LJ13())
    mysys.do_benchmark()

    print ""
    print "doing benchmark for a cluster of 20 Lennard Jones atoms"
    mysys = BenchmarkSystem(LJ20())
    mysys.do_benchmark()
    
    #because LJ is not compiled, it takes too long to benchmark larger LJ systems
    #with a compiled version, LJ38, which is a quite difficult problem for it's size,
    #can be found in about 500000 function evaluations.

def benchmark_anneal():
    print ""
    print "doing benchmark for Ackey function"
    mysys = BenchmarkSystem(Ackey())
    mysys.do_benchmark_anneal()

    print ""
    print "doing benchmark for Levi function"
    mysys = BenchmarkSystem(Levi())
    mysys.do_benchmark_anneal()

    print ""
    print "doing benchmark for Holder Table function"
    mysys = BenchmarkSystem(HolderTable())
    mysys.do_benchmark_anneal()

    print ""
    print "doing benchmark for a cluster of 13 Lennard Jones atoms"
    mysys = BenchmarkSystem(LJ13())
    mysys.do_benchmark_anneal()

if __name__ == "__main__":
    print ""
    print "doing benchmarks for basinhopping"
    print ""
    benchmark_basinhopping()

    #note: this is not a fair comparison because most of these are bounded 
    #and I don't know how to apply bounds with anneal.  also I don't know how
    #to optimize anneal
    print ""
    print "doing benchmarks for anneal"
    print "note: this is not a fair comparison because most of these are bounded" 
    print "and I don't know how to apply bounds with anneal.  Also I don't know how"
    print "to optimize anneal, so I'm just using the default settings"
    print ""
    benchmark_anneal()
