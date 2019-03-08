import numpy as np
import matplotlib.pyplot as plt
import math

origin = [0], [0] # origin point

for theta in range(0, 360, 20):

    theta_rads = math.radians(theta)
    x = math.sin(theta_rads)
    y = math.cos(theta_rads)
    V = 10 * np.array([[x, y]])

    V /= np.abs(V).max()

    plt.quiver(*origin, V[:,0], V[:,1], scale=5)


plt.show()
