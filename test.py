import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pptk

hyperbolic_poraboloid = lambda x, y: x ** 2 - y ** 2

t = np.linspace(-1.0, 1.0, 100)
x,y = np.meshgrid(t,t)
z = hyperbolic_poraboloid(x,y)
P = np.stack([x,y,z], axis = -1).reshape(-1,3)

v = pptk.viewer(P)
v.attributes(P[:, 2])
v.set(point_size=0.005)

dPdx = np.stack([np.gradient(x, axis=1), np.gradient(y, axis=1), np.gradient(z, axis=1)], axis=-1)
dPdy = np.stack([np.gradient(x, axis=0), np.gradient(y, axis=0), np.gradient(z, axis=0)], axis=-1)

# calculate gradient magnitudes
mag = np.sqrt(np.sum(dPdx ** 2 + dPdy ** 2, axis=-1)).flatten()

# calculate normal vectors
N = np.cross(dPdx, dPdy, axis=-1)
N /= np.sqrt(np.sum(N ** 2, axis=-1))[:, :, None]
N = N.reshape(-1, 3)

# set per-point attributes
v.attributes(P[:, 2], mag, 0.5 * (N + 1))