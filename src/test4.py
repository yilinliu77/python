import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]

# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])

mu = np.array([0.0, 0.0])

sigma = np.array([.5, .5])
covariance = np.diag(sigma ** 2)

z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

# Reshape back to a (30, 30) grid.
z = z.reshape(x.shape)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# Color according to z value.
ax.plot_surface(x, y, z, cmap='cividis')
# Disable axis
ax.axis('off')
# export
plt.savefig('gaussian.pdf', dpi=800)
plt.show()
