"""
Distribute a set of n points on a sphere of radius r optimally

"""

import click
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


def distribute_sphere(n, r=1):
    ic = construct_initial_sample(n-1)
    bounds = [(0, (((k - 1) % 2) + 1)*np.pi) for k in range(2*(n - 1))]
    result = minimize(evaluate_objective, ic, bounds=bounds)
    solution = x_to_cartesian(result.x)
    return solution


def plot_x(fig=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='b')


def construct_solution(x):
    x = np.reshape(x, (-1, 2))


def construct_initial_sample(n):
    """
    Generate a uniform sample on the surface of the unit sphere in polar coordinates

    :param n: the number of points to generate
    :return: an n x 2 array that specifies [azimuth, inclination] in spherical coordinates
    """
    sample_normal = np.random.normal(size=(n, 3))
    sample_radius = np.linalg.norm(sample_normal, axis=1, keepdims=True)
    sample_cartesian = sample_normal / sample_radius
    sample_polar = cartesian_to_polar(sample_cartesian)
    return np.reshape(sample_polar[:, 1:3], (-1))


def cartesian_to_polar(p):
    radius = np.linalg.norm(p, axis=1, keepdims=True)
    azimuth = np.arctan2(p[:, 1:2], p[:, 0:1])
    inclination = np.arccos(p[:, 2:3] / radius)
    return np.concatenate((radius, azimuth, inclination), axis=1)


def polar_to_cartesian(p):
    r = p[:, 0:1]
    theta = p[:, 1:2]
    phi = p[:, 2:3]
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return np.concatenate((x, y, z), axis=1)


def evaluate_objective(x):
    """
    Compute the volume of the convex hull of the points
    :param x:
    :param r:
    :return:
    """

    x_points_cartesian = x_to_cartesian(x)
    hull = ConvexHull(x_points_cartesian)

    # Return the negative value because the optimization is a minimization
    return -hull.volume


def x_to_cartesian(x):
    x = np.concatenate((np.reshape(x, (-1, 2)), np.zeros((1, 2))), axis=0)
    x_polar = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    return polar_to_cartesian(x_polar)

