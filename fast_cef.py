import numpy as np

'''
Find a faster way to calculate the collection efficiency function, accounting
for the confocal effects of a pinhole.

Key idea: minified image of pinhole lies at image plane. Treat each 
fluorophore's PSF as a uniform disk that spreads out (neglect the actual
Airy disk structure). 

CEF = area of overlap between pinhole image and PSF / area of PSF

Using scipy.integrate.dblquad to integrate over the pinhole image works badly 
because the integrand (PSF) is frequently inherently discontinuous over a 
weirdly-shaped boundary.

Try a geometric approach to calculating the overlap area instead. See the
following:

http://paulbourke.net/geometry/circlesphere/

http://mathworld.wolfram.com/Circle-CircleIntersection.html

'''

from slow_cef import PSF_rad


def CEF(xprime, yprime, zprime, s0, R0, alpha):
    # find radius of PSF
    Rz = PSF_rad(zprime, R0, alpha)

    # find distance between circle centers
    d = np.sqrt(xprime**2 + yprime**2)

    if d > (s0 + Rz):
        # no overlap 
        return 0
    elif d < np.abs(s0 - Rz):
        # one of the two is contained within the other
        if s0 > Rz:
            # pinhole image larger than PSF
            return 1.
        else:
            # PSF larger than pinhole, CEF = ratio of areas
            return s0**2 / Rz**2
    elif (d == 0) and (s0 == Rz):
        # circles perfectly overlap
        return 1.
    else:
        # two intersection points
        # we don't actually need to know what the points are
        overlap_area = circle_overlap(s0, Rz, d)
        return overlap_area / (np.pi * Rz**2)
        

def circle_overlap(r, R, d):
    '''
    Calculate overlap area between two circles of radii r and R separated
    by distance d that intersect at two points.

    See Eq. 14 at 
    http://mathworld.wolfram.com/Circle-CircleIntersection.html
    '''
    first_term = r**2 * np.arccos( (d**2 + r**2 - R**2)  / (2. * d * r) )
    second_term = R**2 * np.arccos( (d**2 + R**2 - r**2) / (2. * d * R) )
    expr_inside_sqrt = (- d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)
    third_term = -0.5 * np.sqrt(expr_inside_sqrt)

    return first_term + second_term + third_term
