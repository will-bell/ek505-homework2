import os
import shutil
from datetime import datetime
from math import floor, ceil
from typing import List, Tuple, Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from geometry import Polygon, Square, convex_minkowski_sum, triangulate
from djikstra import DjikstraGraph, djikstra


class GridLocked(Exception):
    pass


class GridUnfilled(Exception):
    pass


manhattan_neighbors = [[0, 1], [1, 0], [0, -1], [-1, 0]]
diagonal_neighbors = [[1, 1], [1, -1], [-1, -1], [-1, 1]]


def datetime_name():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H-%M-%S")
    return dt_string


class GVD(DjikstraGraph):

    def __init__(self, grid: np.ndarray):
        self._grid = grid.copy()

    def _manhattan_neighbors(self, xp: int, yp: int):
        neighbors = []
        for dx, dy in manhattan_neighbors:
            xn = xp + dx
            yn = yp + dy
            if xn < 0 or yn < 0:
                continue
            if xn >= self._grid.shape[1] or yn >= self._grid.shape[0]:
                continue
            neighbors.append((xn, yn))

        return neighbors

    def evaluate_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return 1.

    def get_neighbors(self, a: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        xp, yp = a
        for xn, yn in self._manhattan_neighbors(xp, yp):
            if self._grid[yn][xn]:
                neighbors.append((xn, yn))

        return neighbors


class BrushfireGrid:

    def __init__(self, resolution: int, obstacles: List[Polygon], animate_calculation: bool = False, results_name: str = None):
        self.resolution = resolution
        self.side_length = 100
        self.pixel_width = self.side_length / self.resolution
        self.pixel_shape = Square(np.array([0, 0]), self.pixel_width)
        self._corners = [(0, 0), (self.resolution-1, self.resolution-1), (0, self.resolution-1), (self.resolution-1, 0)]

        self._obstacles = obstacles
        self._expanded_obstacles = []

        self._pixel_values = self._instantiate_pixel_values()
        self._voronoi_boundary = np.zeros((resolution, resolution), dtype=bool)
        self._pixel_ownership = self._instantiate_pixel_ownership()

        self._animate = animate_calculation

        results_dir = os.path.join(os.pardir, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_name = results_name if results_name is not None else datetime_name()
        self._results_dir = os.path.join(results_dir, results_name)
        if not os.path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        if animate_calculation:
            self._tmp_dir = os.path.join(self._results_dir, 'tmp')
            if not os.path.exists(self._tmp_dir):
                os.makedirs(self._tmp_dir)

        self._calculate()

    def __del__(self):
        if not len(os.listdir(self._results_dir)):
            shutil.rmtree(self._results_dir)

    @property
    def pixel_values(self) -> np.ndarray:
        return self._pixel_values.copy()

    @property
    def voronoi_boundary(self) -> np.ndarray:
        return self._voronoi_boundary.copy()

    @property
    def voronoi_region(self) -> np.ndarray:
        return self._pixel_ownership.copy()

    @property
    def gvd(self) -> GVD:
        return GVD(self._voronoi_boundary)

    def _instantiate_pixel_values(self) -> np.ndarray:
        pixel_values = np.zeros((self.resolution, self.resolution), dtype=int)
        pixel_values[:, 0] = 1
        pixel_values[0, :] = 1
        pixel_values[:, -1] = 1
        pixel_values[-1, :] = 1

        return pixel_values

    def _instantiate_pixel_ownership(self):
        pixel_owners = np.zeros((self.resolution, self.resolution), dtype=int)
        pixel_owners[0, :-1] = 1
        pixel_owners[:-1, -1] = 2
        pixel_owners[-1, 1:] = 3
        pixel_owners[1:, 0] = 4

        return pixel_owners

    def _real_to_pixel(self, real: float, direction: str = None) -> int:
        direction = 'neither' if direction is None else direction
        if direction == 'down':
            rounding = floor
        elif direction == 'up':
            rounding = ceil
        elif direction == 'neither':
            rounding = round
        else:
            raise ValueError("direction should be 'up', 'down', or 'neither'")
        return min(max(rounding((real - self.pixel_width / 2) / self.pixel_width), 0), self.resolution - 1)

    def _pixel_to_real(self, pixel: int) -> float:
        return pixel * self.pixel_width + self.pixel_width / 2

    def _block(self, obstacle: Polygon) -> np.ndarray:
        blocked_pixels = np.zeros((self.resolution, self.resolution), dtype=bool)

        xr_min = np.min(obstacle.points[:, 0])
        xr_max = np.max(obstacle.points[:, 0])
        yr_min = np.min(obstacle.points[:, 1])
        yr_max = np.max(obstacle.points[:, 1])

        xp_min = self._real_to_pixel(xr_min, 'down')
        xp_max = self._real_to_pixel(xr_max, 'up')
        yp_min = self._real_to_pixel(yr_min, 'down')
        yp_max = self._real_to_pixel(yr_max, 'up')

        for yp in range(yp_min, yp_max + 1):
            for xp in range(xp_min, xp_max + 1):
                if not blocked_pixels[yp][xp]:
                    real_position = np.array([xp, yp]) * self.pixel_width + self.pixel_width / 2
                    if obstacle.contains(real_position):
                        blocked_pixels[yp][xp] = True

        return blocked_pixels

    def _manhattan_neighbors(self, xp: int, yp: int):
        neighbors = []
        for dx, dy in manhattan_neighbors:
            xn = xp + dx
            yn = yp + dy
            if xn < 0 or yn < 0:
                continue
            if xn >= self.resolution or yn >= self.resolution:
                continue
            neighbors.append((xn, yn))

        return neighbors

    def _detect_boundaries(self, blocked_pixels: np.ndarray) -> List[Tuple[int, int]]:
        boundaries = []

        def detect_boundary():
            for xn, yn in self._manhattan_neighbors(xp, yp):
                if not blocked_pixels[yn][xn]:
                    boundaries.append((yp, xp))
                    return

        # Find the boundary pixels
        for yp, xp in zip(*blocked_pixels.nonzero()):
            detect_boundary()

        return boundaries

    def _expand_obstacle_pixels(self, obstacle_pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        next_obstacle_pixels = []
        this_obstacle = self._pixel_ownership[obstacle_pixels[0]]

        for yp, xp in obstacle_pixels:
            distance = self._pixel_values[yp][xp] + 1
            for xn, yn in self._manhattan_neighbors(xp, yp):
                pixel_ownership = self._pixel_ownership[yn][xn]
                if pixel_ownership == 0:
                    next_obstacle_pixels.append((yn, xn))
                    self._pixel_values[yn][xn] = distance
                    self._pixel_ownership[yn][xn] = this_obstacle
                elif pixel_ownership != this_obstacle and self._pixel_values[yp, xp] != 1:
                    if (yp, xp) not in self._corners:
                        self._voronoi_boundary[yn][xn] = True

        return next_obstacle_pixels

    def _calculate(self):
        obstacle_queue = [
            [(0, xp) for xp in range(0, self.resolution - 1)],
            [(yp, self.resolution - 1) for yp in range(0, self.resolution - 1)],
            [(self.resolution - 1, xp) for xp in range(1, self.resolution)],
            [(yp, 0) for yp in range(1, self.resolution)],
        ]

        blocked_pixels_layers = [self._instantiate_pixel_values().astype(bool)]
        for i, obstacle in enumerate(self._obstacles):
            for triangle in triangulate(obstacle):
                expanded_obstacle = convex_minkowski_sum(self.pixel_shape, triangle)

                blocked_pixels = self._block(expanded_obstacle)
                self._pixel_ownership[blocked_pixels] = i + 5

                obstacle_boundaries = self._detect_boundaries(blocked_pixels)

                blocked_pixels_layers.append(blocked_pixels)
                obstacle_queue.append(obstacle_boundaries)

        blocked_pixels = np.dstack(blocked_pixels_layers)
        self._pixel_values = np.any(blocked_pixels, axis=2).astype(int)

        images = []
        i = 0
        n_obstacles = len(obstacle_queue)
        while 1:
            obstacle_pixels = obstacle_queue[i]
            if len(obstacle_pixels):
                next_obstacle_pixels = self._expand_obstacle_pixels(obstacle_pixels)
                obstacle_queue[i] = next_obstacle_pixels
            if all([len(pixels) == 0 for pixels in obstacle_queue]):
                break

            if self._animate:
                fig = Figure()
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                self.plot_voronoi_diagram(ax=ax)
                canvas.draw()
                w, h = fig.canvas.get_width_height()
                image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((h, w, 3))
                images.append(image)

            i = (i + 1) % n_obstacles

        if self._animate:
            imageio.mimsave(os.path.join(self._results_dir, 'VoronoiRegionGeneration.gif'), images, format='GIF', fps=60)
            shutil.rmtree(self._tmp_dir)
            plt.close('all')

    def plot_distances(self, save: bool = False, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(self._pixel_values, origin='lower')
        plt.colorbar(im)
        if save:
            plt.savefig(os.path.join(self._results_dir, 'BrushfireDistances.png'))

    def plot_voronoi_boundary(self, save: bool = False, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self._voronoi_boundary, origin='lower')
        if save:
            plt.savefig(os.path.join(self._results_dir, 'VoronoiBoundary.png'))

    def plot_voronoi_diagram(self, save: bool = False, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        array = self._pixel_ownership.copy()
        array[self._pixel_values == 1] = 0
        ax.imshow(array, origin='lower')
        if save:
            plt.savefig(os.path.join(self._results_dir, 'VoronoiDiagram.png'))

    def _get_greatest_neighbor(self, xp: int, yp: int) -> Optional[Tuple[int, int]]:
        greatest_value = -1
        greatest_neighbor = None
        for xn, yn in self._manhattan_neighbors(xp, yp):
            if self._pixel_values[yn][xn] > greatest_value:
                greatest_value = self._pixel_values[yn][xn]
                greatest_neighbor = (xn, yn)

        if greatest_value <= self._pixel_values[yp][xp]:
            return None

        return greatest_neighbor

    def move_up(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [start]
        while 1:
            xp, yp = path[-1]
            if self._voronoi_boundary[yp][xp]:
                return path

            greatest_neighbor = self._get_greatest_neighbor(xp, yp)
            if greatest_neighbor is None:
                return path
            else:
                path.append(greatest_neighbor)

    def calculate_paths(self, start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        startp = (self._real_to_pixel(start[0]), self._real_to_pixel(start[1]))
        goalp = (self._real_to_pixel(goal[0]), self._real_to_pixel(goal[1]))

        up_path = self.move_up(startp)
        down_path = list(reversed(self.move_up(goalp)))
        gvd_path = djikstra(self.gvd, up_path[-1], down_path[0])

        pathp = np.vstack([np.asarray(up_path[:-1]), np.asarray(gvd_path), np.asarray(down_path[1:])])
        path = pathp * self.pixel_width + self.pixel_width / 2

        return pathp, path

    def calculate_pixel_path(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        path, _ = self.calculate_paths(start, goal)
        return path

    def calculate_real_path(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        _, path = self.calculate_paths(start, goal)
        return path

    def _plot_pixel_paths(self, start: np.ndarray, goal: np.ndarray, ax):
        if start.shape[0] > 1:
            colors = ['r', 'b', 'g', 'm']
            for i, start_ in enumerate(start):
                path = self.calculate_pixel_path(start_, goal)
                c = i % len(colors)
                ax.plot(path[:, 0], path[:, 1], color=colors[c], lw=3)
        else:
            path = self.calculate_pixel_path(start, goal)
            ax.plot(path[:, 0], path[:, 1], '-r', lw=3)

    def _plot_real_paths(self, start: np.ndarray, goal: np.ndarray, ax):
        if start.shape[0] > 1:
            colors = ['r', 'b', 'g', 'm']
            for i, start_ in enumerate(start):
                path = self.calculate_real_path(start_, goal)
                c = i % len(colors)
                ax.plot(path[:, 0], path[:, 1], color=colors[c], lw=3)
        else:
            path = self.calculate_pixel_path(start, goal)
            ax.plot(path[:, 0], path[:, 1], '-r', lw=3)

    def plot_path_on_distances(self, start: np.ndarray, goal: np.ndarray, ax=None, save: bool = False):
        if len(start.shape) < 2:
            start = start[np.newaxis]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        self.plot_distances(ax=ax)
        self._plot_pixel_paths(start, goal, ax)
        if save:
            plt.savefig(os.path.join(self._results_dir, 'PathOnBrushfire.png'))

    def plot_path_on_gvd(self, start: np.ndarray, goal: np.ndarray, ax=None, save: bool = False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        self.plot_voronoi_boundary(ax=ax)
        self._plot_pixel_paths(start, goal, ax)
        if save:
            plt.savefig(os.path.join(self._results_dir, 'PathOnGVD.png'))

    def plot_path_on_real(self, start: np.ndarray, goal: np.ndarray, ax=None, save: bool = False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        for obstacle in self._obstacles:
            obstacle.plot(ax, color='r', lw=3)
        self._plot_real_paths(start, goal, ax)
        ax.set_aspect('equal', 'box')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        if save:
            plt.savefig(os.path.join(self._results_dir, 'PathOnRealSpace.png'))


def example_1(resolution: int = 100, save: bool = False):
    obstacle = Square(np.array([50, 50]), 50)

    grid = BrushfireGrid(resolution, [obstacle], animate_calculation=save, results_name='Example1')

    grid.plot_distances(save=save)
    grid.plot_voronoi_boundary(save=save)
    grid.plot_voronoi_diagram(save=save)
    plt.show()


def example_2(resolution: int = 100, save: bool = False):
    obstacle1 = Polygon(np.array([[20, 20], [60, 20], [20, 60]]))
    obstacle2 = Polygon(np.array([[80, 80], [60, 80], [80, 60]]))

    grid = BrushfireGrid(resolution, [obstacle1, obstacle2], animate_calculation=save, results_name='Example2')

    grid.plot_distances(save=save)
    grid.plot_voronoi_boundary(save=save)
    grid.plot_voronoi_diagram(save=save)
    plt.show()


def example_3(resolution: int = 100, save: bool = False):
    obstacle1 = Square(np.array([40, 40]), 30)
    obstacle2 = Square(np.array([60, 60]), 25)

    grid_ = BrushfireGrid(resolution, [obstacle1, obstacle2], animate_calculation=save, results_name='Example3')

    grid_.plot_distances(save=save)
    grid_.plot_voronoi_boundary(save=save)
    grid_.plot_voronoi_diagram(save=save)
    plt.show()


def example_4(resolution: int = 100, save: bool = False):
    obstacle1 = Square(np.array([40, 40]), 30)
    obstacle2 = Square(np.array([60, 60]), 25)

    grid_ = BrushfireGrid(resolution, [obstacle1, obstacle2], results_name='Example4')

    start = np.array([[10, 40], [60, 20], [40, 60]])
    goal = np.array([70, 80])

    grid_.plot_path_on_distances(start, goal, save=save)
    grid_.plot_path_on_gvd(start, goal, save=save)
    grid_.plot_path_on_real(start, goal, save=save)

    plt.show()


def run_examples(resolution: int = 100, save: bool = False):
    example_1(resolution, save=save)
    example_2(resolution, save=save)
    example_3(resolution, save=save)
    example_4(resolution, save=save)


if __name__ == '__main__':
    # run_examples()
    example_4(resolution=200, save=True)