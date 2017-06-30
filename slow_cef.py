import numpy as np
from scipy.integrate import dblquad

def integrate_circle(func, R, args = None):
    if args is None:
        result, _ = dblquad(func, -R, R,
                            lambda x: -np.sqrt(R**2 - x**2),
                            lambda x: np.sqrt(R**2 - x**2))
    else:
        result, _ = dblquad(func, -R, R,
                            lambda x: -np.sqrt(R**2 - x**2),
                            lambda x: np.sqrt(R**2 - x**2), args = args)
    return result

def PSF_rad(z, R0, alpha):
    return np.sqrt(R0**2 + z**2 * np.tan(alpha))

def CEF(xprime, yprime, z, s0, R0, alpha):
    def integrand(y, x):
        R = PSF_rad(z, R0, alpha)
        rminusrp = np.sqrt((x - xprime)**2 + (y - yprime)**2)
        if rminusrp <= R:
            return 1 / (np.pi * R**2)
        else:
            return 0

    return integrate_circle(integrand, s0)

