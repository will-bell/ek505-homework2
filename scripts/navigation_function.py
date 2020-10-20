from geometry import Circle, EPS, LARGE
from common import distance
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class NavigationFunction2D:

    def __init__(self, goal: np.ndarray, workspace: Circle, obstacles: List[Circle], k: int, default_epsilon: float = 0.1):
        self._goal, self._workspace, self._k, self._epsilon = self._verify(goal, workspace, k, default_epsilon)
        self._obstacles = obstacles

    @staticmethod
    def _verify(goal: np.ndarray, workspace: Circle, k: int, default_epsilon: float) -> Tuple[np.ndarray, Circle, int, float]:
        assert goal.size == 2, 'Goal must be a two-dimensional point'

        r_goal = distance(goal, workspace.center)
        assert r_goal <= workspace.radius, 'Goal must be inside the workspace'

        assert k > 0, 'Parameter k must be a value greater than zero'
        assert default_epsilon > 0, 'Epsilon must be a value greater than 0'

        return goal.copy(), workspace, k, default_epsilon

    def _evaluate_gamma(self, q: np.ndarray) -> float:
        gamma = distance(q, self._goal)**(2 * self._k)

        return gamma

    @staticmethod
    def _evaluate_beta_i(q: np.ndarray, obstacle: Circle):
        return distance(q, obstacle.center)**2 - obstacle.radius**2

    def _evaluate_beta(self, q: np.ndarray) -> float:
        beta = self._workspace.radius**2 - distance(q, self._workspace.center)**2
        for obstacle in self._obstacles:
            beta *= self._evaluate_beta_i(q, obstacle)

        return beta

    def _evaluate_alpha(self, q: np.ndarray) -> float:
        gamma = self._evaluate_gamma(q)
        beta = self._evaluate_beta(q)
        if abs(beta) < EPS:
            alpha = LARGE
        else:
            alpha = gamma / beta

        return alpha

    def _evaluate_phi(self, q: np.ndarray) -> float:
        alpha = self._evaluate_alpha(q)
        if alpha < 0:
            return 1
        else:
            phi = (alpha / (1 + alpha))**(1/self._k)

        return phi

    def evaluate(self, q: np.ndarray) -> float:
        return self._evaluate_phi(q)

    def _evaluate_grad_gamma(self, q: np.ndarray) -> np.ndarray:
        grad_gamma = 2 * self._k * distance(q, self._goal)**(2 * self._k - 1) * (q - self._goal) / distance(q, self._goal)

        return grad_gamma

    def _evaluate_grad_beta(self, q: np.ndarray) -> np.ndarray:
        beta_0 = self._workspace.radius**2 - distance(q, self._workspace.center)**2
        grad_beta_0 = -2 * (q - self._workspace.center)

        betas = [beta_0]
        grad_betas = [grad_beta_0]
        for obstacle in self._obstacles:
            beta_i = self._evaluate_beta_i(q, obstacle)
            grad_beta_i = 2 * (q - obstacle.center)

            betas.append(beta_i)
            grad_betas.append(grad_beta_i)

        grad_beta = np.zeros(2)
        n = len(betas)
        for i in range(n):
            product = grad_betas[i]
            for j in range(n):
                if j != i:
                    product *= betas[j]

            grad_beta += product

        return grad_beta

    def _evaluate_grad_alpha(self, q: np.ndarray) -> np.ndarray:
        gamma = self._evaluate_gamma(q)
        beta = self._evaluate_beta(q)
        grad_gamma = self._evaluate_grad_gamma(q)
        grad_beta = self._evaluate_grad_beta(q)
        grad_alpha = (grad_gamma * beta - gamma * grad_beta) / beta**2

        return grad_alpha

    def _evaluate_grad_phi(self, q: np.ndarray) -> np.ndarray:
        alpha = self._evaluate_alpha(q)
        grad_alpha = self._evaluate_grad_alpha(q)
        grad_phi = (1 / self._k) * (alpha / (1 + alpha))**((1 - self._k) / self._k) * (1 / (1 + alpha)**2) * grad_alpha

        return grad_phi

    def evaluate_gradient(self, q: np.ndarray) -> np.ndarray:
        return self._evaluate_grad_phi(q)

    def calculate_path(self, start: np.ndarray, max_steps: int = 10000, epsilon: float = None, scale_magnitude: float = None) -> np.ndarray:
        epsilon = self._epsilon if epsilon is None else epsilon

        starting_distance = distance(start, self._goal)

        path = np.zeros((max_steps, 2))
        path[0] = start
        step = 1
        while distance(path[step - 1], self._goal) > EPS and step < max_steps:
            grad = self._evaluate_grad_phi(path[step - 1])
            if scale_magnitude is not None:
                d2goal = min(distance(path[step - 1], self._goal), starting_distance)
                eta = (d2goal / starting_distance)**2
                grad = eta * scale_magnitude * grad / np.linalg.norm(grad)
            path[step] = path[step - 1] - epsilon * grad
            step += 1

        return path[:step]

    def _evaluate_potential_on_grid(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        zz_nav = []
        for qx, qy in zip(xx.ravel(), yy.ravel()):
            q = np.array([qx, qy])
            zz_nav.append(self._evaluate_phi(q))

        zz_nav = np.asarray(zz_nav).reshape(xx.shape)

        return zz_nav

    def _workspace_meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        x_min = self._workspace.center[0] - self._workspace.radius
        x_max = self._workspace.center[0] + self._workspace.radius
        y_min = self._workspace.center[1] - self._workspace.radius
        y_max = self._workspace.center[1] + self._workspace.radius
        x = np.arange(x_min - .1, x_max + .1, .05)
        y = np.arange(y_min - .1, y_max + .1, .05)

        xx, yy = np.meshgrid(x, y)

        return xx, yy

    def plot_path_on_contour(self, start: np.ndarray, **kwargs):
        xx, yy = self._workspace_meshgrid()
        zz_nav = self._evaluate_potential_on_grid(xx, yy)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contour(xx, yy, zz_nav, 50)
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

    def plot_navigation_contour(self, x: np.ndarray, y: np.ndarray):
        xx, yy = np.meshgrid(x, y)
        zz_nav = self._evaluate_potential_on_grid(xx, yy)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contour(xx, yy, zz_nav, 50)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        for obstacle in self._obstacles:
            obstacle.plot(ax)

    def plot_navigation_surface(self, x: np.ndarray, y: np.ndarray):
        xx, yy = np.meshgrid(x, y)
        zz_nav = self._evaluate_potential_on_grid(xx, yy)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, zz_nav)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=45., azim=45.)


if __name__ == '__main__':
    obstacles_ = [
        Circle(np.array([0, 5]), 2),
        Circle(np.array([0, -5]), 2),
        Circle(np.array([5, 0]), 2),
        Circle(np.array([-5, 0]), 2)
    ]
    workspace = Circle(np.array([0., 0.]), 12)
    goal = np.array([10., 0.])

    navigation_function = NavigationFunction2D(np.array(goal), workspace, obstacles_, 5)
    # navigation_function.plot_path_on_contour(np.array([[1, 2], [2, 1], [3, 3.1]]), epsilon=1)
    navigation_function.plot_path_on_contour(np.array([[-9, .1], [-8, -5], [0, .1], [.1, -9]]), epsilon=1, scale_magnitude=.1)
    # navigation_function.plot_path_on_contour(np.array([-9, 0.1]), epsilon=1, scale_magnitude=.1)
    # navigation_function.plot_navigation_surface(np.arange(-13, 13, .05), np.arange(-13, 13, .05))
    # navigation_function.plot_navigation_contour(np.arange(-13, 13, .05), np.arange(-13, 13, .05))
    plt.show()
