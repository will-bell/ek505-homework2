from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from geometry import EPS, LARGE
from common import CircleObstacle, distance, circle_distance
from navigation_function import NavigationFunction2D


class PotentialField2D:

    def __init__(self, goal: np.ndarray, workspace: CircleObstacle, obstacles: List[CircleObstacle], eta: float = .1,
                 zeta: float = .1, quadratic_radius: float = 10, default_epsilon: float = .1, k: int = None,):
        self._goal, self._eta, self._zeta, self._dstar, self._epsilon = \
            self._verify(goal, eta, zeta, quadratic_radius, default_epsilon)
        self._workspace = workspace
        self._obstacles = obstacles

        if k is not None:
            self._navigation_function = NavigationFunction2D(self._goal, self._workspace, self._obstacles, k)
        else:
            self._navigation_function = None

    @staticmethod
    def _verify(goal: np.ndarray, eta: float, zeta: float, quadratic_radius: float, default_epsilon: float) \
            -> Tuple[np.ndarray, float, float, float, float]:
        assert goal.size == 2, 'Goal must be a two-dimensional point'
        assert eta > 0, 'Eta must be a value greater than 0'
        assert zeta > 0, 'Eta must be a value greater than 0'
        assert quadratic_radius > 0, 'Eta must be a value greater than 0'
        assert default_epsilon > 0, 'Epsilon must be a value greater than 0'
        return goal.copy(), eta, zeta, quadratic_radius, default_epsilon

    def _evaluate_attractive_potential(self, q: np.ndarray) -> float:
        d2goal = distance(q, self._goal)
        if d2goal <= self._dstar:
            u_attr = .5 * self._zeta * d2goal**2
        else:
            u_attr = self._dstar * self._zeta * d2goal - .5 * self._zeta * self._dstar**2

        return u_attr

    def _evaluate_repulsive_potential(self, q: np.ndarray) -> float:
        u_rep = 0

        d2circle = self._workspace.radius - distance(q, self._workspace.center)
        if d2circle < EPS:
            u_rep += LARGE
        elif abs(d2circle) <= self._workspace.safe_distance:
            u_rep += .5 * self._eta * ((1 / d2circle) - (1 / self._workspace.safe_distance)) ** 2

        for circle in self._obstacles:
            d2circle = circle_distance(q, circle)
            if d2circle < EPS:
                u_rep += LARGE
            elif d2circle <= circle.safe_distance:
                u_rep += .5 * self._eta * ((1 / d2circle) - (1 / circle.safe_distance))**2

        return u_rep

    def _evaluate_grad_attractive_potential(self, q: np.ndarray) -> np.ndarray:
        d2goal = distance(q, self._goal)
        if d2goal <= self._dstar:
            grad_u_attr = self._zeta * (q - self._goal)
        else:
            grad_u_attr = self._dstar * self._zeta * (q - self._goal) / d2goal

        return grad_u_attr

    def _evaluate_grad_repulsive_potential(self, q: np.ndarray) -> np.ndarray:
        grad_u_rep = np.zeros(2)

        for circle in self._obstacles:
            d2circle = circle_distance(q, circle)
            grad_circle_distance = (q - circle.center) / d2circle
            if d2circle < EPS:
                grad_u_rep += LARGE * grad_circle_distance
            elif d2circle <= circle.safe_distance:
                grad_u_rep += self._eta * ((1 / circle.safe_distance) - (1 / d2circle)) * (1 / d2circle)**2 * grad_circle_distance

        return grad_u_rep

    def _evaluate_grad_potential(self, q: np.ndarray) -> np.ndarray:
        grad_attractive = self._evaluate_grad_attractive_potential(q)
        grad_repulsive = self._evaluate_grad_repulsive_potential(q)

        if self._navigation_function:
            grad_navigation = self._navigation_function.evaluate_gradient(q)
        else:
            grad_navigation = np.zeros(2)

        grad = grad_attractive + grad_repulsive + grad_navigation

        return grad

    def calculate_path(self, start: np.ndarray, max_steps: int = 1000000, epsilon: float = None) -> np.ndarray:
        epsilon = self._epsilon if epsilon is None else epsilon

        path = np.zeros((max_steps, 2))
        path[0] = start
        step = 1
        while distance(path[step - 1], self._goal) > EPS and step < max_steps:
            grad = self._evaluate_grad_potential(path[step - 1])
            path[step] = path[step - 1] - epsilon * grad
            step += 1

        return path[:step]

    def _evaluate_potential_on_grid(self, xx: np.ndarray, yy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        zz_attr, zz_rep, zz_nav = [], [], []
        for qx, qy in zip(xx.ravel(), yy.ravel()):
            q = np.array([qx, qy])
            zz_attr.append(self._evaluate_attractive_potential(q))
            zz_rep.append(self._evaluate_repulsive_potential(q))
            if self._navigation_function:
                zz_nav.append(self._navigation_function.evaluate(q))

        zz_attr = np.asarray(zz_attr).reshape(xx.shape)
        zz_rep = np.asarray(zz_rep).reshape(xx.shape)

        if self._navigation_function:
            zz_nav = np.asarray(zz_nav).reshape(xx.shape)
        else:
            zz_nav = np.zeros(zz_attr.shape)

        return zz_attr, zz_rep, zz_nav

    def plot_potential_surface(self, x: np.ndarray, y: np.ndarray):
        xx, yy = np.meshgrid(x, y)
        zz_attr, zz_rep, zz_nav = self._evaluate_potential_on_grid(xx, yy)

        ceiling = np.max(zz_attr)
        zz = zz_attr + zz_rep + zz_nav
        zz[zz > ceiling] = ceiling

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=45., azim=45.)

    def plot_potential_contour(self, x: np.ndarray, y: np.ndarray):
        xx, yy = np.meshgrid(x, y)
        zz_attr, zz_rep, zz_nav = self._evaluate_potential_on_grid(xx, yy)

        ceiling = np.max(zz_attr)
        zz = zz_attr + zz_rep + zz_nav
        zz[zz > ceiling] = ceiling

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contour(xx, yy, zz, 50)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        for obstacle in self._obstacles:
            obstacle.plot(ax)

    def _workspace_meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        x_min = self._workspace.center[0] - self._workspace.radius
        x_max = self._workspace.center[0] + self._workspace.radius
        y_min = self._workspace.center[1] - self._workspace.radius
        y_max = self._workspace.center[1] + self._workspace.radius
        x = np.arange(x_min - .1, x_max + .1, .05)
        y = np.arange(y_min - .1, y_max + .1, .05)

        xx, yy = np.meshgrid(x, y)

        return xx, yy

    def plot_path_on_surface(self, start: np.ndarray, **kwargs):
        xx, yy = self._workspace_meshgrid()
        zz_attr, zz_rep, zz_nav = self._evaluate_potential_on_grid(xx, yy)

        ceiling = np.max(zz_attr)
        zz = zz_attr + zz_rep + zz_nav
        zz[zz > ceiling] = ceiling

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(xx, yy, zz)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if len(start.shape) == 1:
            start = start[np.newaxis]

        if start.shape[0] > 1:
            colors = ['r', 'b', 'g', 'm']
            for i, start_ in enumerate(start):
                path = self.calculate_path(start_, **kwargs)
                path = np.hstack([path, np.zeros((path.shape[0], 1))])
                for i in range(path.shape[0]):
                    z_attr = self._evaluate_repulsive_potential(path[i, :2])
                    z_rep = self._evaluate_attractive_potential(path[i, :2])
                    z_nav = self._navigation_function.evaluate(path[i, :2]) if self._navigation_function else 0
                    path[i, 2] = z_attr + z_rep + z_nav

                c = i % len(colors)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], colors[c], lw=3)

        else:
            path = self.calculate_path(start, **kwargs)
            path = np.hstack([path, np.zeros((path.shape[0], 1))])
            for i in range(path.shape[0]):
                z_attr = self._evaluate_repulsive_potential(path[i, :2])
                z_rep = self._evaluate_attractive_potential(path[i, :2])
                z_nav = self._navigation_function.evaluate(path[i, :2]) if self._navigation_function else 0
                path[i, 2] = z_attr + z_rep + z_nav
            ax.plot(path[:, 0], path[:, 1], path[:, 2], '-r', lw=3)

    def plot_path_on_contour(self, start: np.ndarray, **kwargs):
        xx, yy = self._workspace_meshgrid()
        zz_attr, zz_rep, zz_nav = self._evaluate_potential_on_grid(xx, yy)

        ceiling = np.max(zz_attr)
        zz = zz_attr + zz_rep + zz_nav
        zz[zz > ceiling] = ceiling

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contour(xx, yy, zz, 50)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        for obstacle in self._obstacles:
            obstacle.plot(ax)

        if len(start.shape) == 1:
            start = start[np.newaxis]

        if start.shape[0] > 1:
            colors = ['r', 'b', 'g', 'm']
            for i, start_ in enumerate(start):
                path = self.calculate_path(start_, **kwargs)

                c = i % len(colors)
                ax.plot(path[:, 0], path[:, 1], colors[c])
        else:
            path = self.calculate_path(start, **kwargs)
            ax.plot(path[:, 0], path[:, 1], '-r')


if __name__ == '__main__':
    obstacles_ = [
        CircleObstacle(np.array([0, 5]), 2, safe_distance=5),
        CircleObstacle(np.array([0, -5]), 2, safe_distance=5),
        CircleObstacle(np.array([5, 0]), 2, safe_distance=5),
        CircleObstacle(np.array([-5, 0]), 2, safe_distance=5)
    ]
    workspace = CircleObstacle(np.array([0., 0.]), 12, safe_distance=2)
    goal = np.array([10., 0.])

    potential_field = PotentialField2D(np.array(goal), workspace, obstacles_, eta=1, zeta=.1, quadratic_radius=1)
    potential_field.plot_path_on_contour(np.array([[-9, .1], [-8, -5], [0, .1], [.1, -9]]), epsilon=1)
    # potential_field.plot_path_on_surface(np.array([0, 0.]), epsilon=1)
    # potential_field.plot_potential_surface(np.arange(-13, 13, .05), np.arange(-13, 13, .05))
    # potential_field.plot_potential_contour(np.arange(-13, 13, .05), np.arange(-13, 13, .05))
    plt.show()
