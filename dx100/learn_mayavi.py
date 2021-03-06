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

# Create the data.
from numpy import pi, sin, cos, mgrid
dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)
y = r*cos(phi)
z = r*sin(phi)*sin(theta)

# View it.
from mayavi import mlab
s = mlab.mesh(x, y, z)
mlab.show()

# MRI example
### Download the data, if not already on disk #################################
import os
if not os.path.exists('MRbrain.tar.gz'):
    # Download the data
    try:
        from urllib import urlopen
    except ImportError:
        from urllib.request import urlopen
    print("Downloading data, Please Wait (7.8MB)")
    opener = urlopen(
                'http://graphics.stanford.edu/data/voldata/MRbrain.tar.gz')
    open('MRbrain.tar.gz', 'wb').write(opener.read())

# Extract the data
import tarfile
tar_file = tarfile.open('MRbrain.tar.gz')
try:
    os.mkdir('mri_data')
except:
    pass
tar_file.extractall('mri_data')
tar_file.close()


### Read the data in a numpy 3D array #########################################
import numpy as np
data = np.array([np.fromfile(os.path.join('mri_data', 'MRbrain.%i' % i),
                                        dtype='>u2') for i in range(1, 110)])
data.shape = (109, 256, 256)
data = data.T

# Display the data ############################################################
from mayavi import mlab

mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

src = mlab.pipeline.scalar_field(data)
# Our data is not equally spaced in all directions:
src.spacing = [1, 1, 1.5]
src.update_image_data = True


# Extract some inner structures: the ventricles and the inter-hemisphere
# fibers. We define a volume of interest (VOI) that restricts the
# iso-surfaces to the inner of the brain. We do this with the ExtractGrid
# filter.
blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
voi = mlab.pipeline.extract_grid(blur)
voi.set(x_min=125, x_max=193, y_min=92, y_max=125, z_min=34, z_max=75)

mlab.pipeline.iso_surface(voi, contours=[1610, 2480], colormap='Spectral')

# Add two cut planes to show the raw MRI data. We use a threshold filter
# to remove cut the planes outside the brain.
thr = mlab.pipeline.threshold(src, low=1120)
cut_plane = mlab.pipeline.scalar_cut_plane(thr,
                                plane_orientation='y_axes',
                                colormap='black-white',
                                vmin=1400,
                                vmax=2600)
cut_plane.implicit_plane.origin = (136, 111.5, 82)
cut_plane.implicit_plane.widget.enabled = False

cut_plane2 = mlab.pipeline.scalar_cut_plane(thr,
                                plane_orientation='z_axes',
                                colormap='black-white',
                                vmin=1400,
                                vmax=2600)
cut_plane2.implicit_plane.origin = (136, 111.5, 82)
cut_plane2.implicit_plane.widget.enabled = False

# Extract two views of the outside surface. We need to define VOIs in
# order to leave out a cut in the head.
voi2 = mlab.pipeline.extract_grid(src)
voi2.set(y_min=112)
outer = mlab.pipeline.iso_surface(voi2, contours=[1776, ],
                                        color=(0.8, 0.7, 0.6))

voi3 = mlab.pipeline.extract_grid(src)
voi3.set(y_max=112, z_max=53)
outer3 = mlab.pipeline.iso_surface(voi3, contours=[1776, ],
                                         color=(0.8, 0.7, 0.6))


mlab.view(-125, 54, 326, (145.5, 138, 66.5))
mlab.roll(-175)

mlab.show()

import shutil
shutil.rmtree('mri_data')
