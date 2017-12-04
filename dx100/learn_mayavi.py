#!/usr/bin/env mayavi2

import numpy as np

def V(x, y, z):
    """ A 3D sinusoidal lattice with a parabolic confinement. """
    return np.cos(10*x) + np.cos(10*y) + np.cos(10*z) + 2*(x**2 + y**2 + z**2)

X, Y, Z = np.mgrid[-2:2:100j, -2:2:100j, -2:2:100j]
V(X, Y, Z)

from mayavi import mlab
mlab.contour3d(X, Y, Z, V)

def gradient(f, x, y, z, d=0.01):
    """ Return the gradient of f in (x, y, z). """
    fx  = f(x+d, y, z)
    fx_ = f(x-d, y, z)
    fy  = f(x, y+d, z)
    fy_ = f(x, y-d, z)
    fz  = f(x, y, z+d)
    fz_ = f(x, y, z-d)
    return (fx-fx_)/(2*d), (fy-fy_)/(2*d), (fz-fz_)/(2*d)


Vx, Vy, Vz = gradient(V, X[50, ::3, ::3], Y[50, ::3, ::3], Z[50, ::3, ::3])
mlab.quiver3d(X[50, ::3, ::3], Y[50, ::3, ::3], Z[50, ::3, ::3],
                     Vx, Vy, Vz, scale_factor=-0.2, color=(1, 1, 1))
