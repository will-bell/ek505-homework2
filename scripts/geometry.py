import sys
from math import atan2, pi
from typing import Tuple, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

EPS = sys.float_info.epsilon
LARGE = 1e15


class WrapIndex:

    def __init__(self, list_length: int, start_index: int = 0, stop_index: int = None, reverse: bool = False):
        self._list_length = list_length
        self._start_index = start_index % list_length
        self._stop_index = self._start_index if stop_index is None else stop_index
        self._index = self._start_index
        self._reverse = reverse
        self._increment = 1 if not reverse else -1

        self._first_call = True
        self._loop_count = 0

    def __iter__(self) -> 'WrapIndex':
        return self

    def __next__(self) -> int:
        if self._first_call:
            self._first_call = False
            return self._start_index
        return self._next()

    def _next(self):
        next_index = (self._index + self._increment) % self._list_length
        if next_index == self._stop_index:
            raise StopIteration()
        self._index = next_index
        return next_index


class Segment:

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self._a, self._b = self._verify(a, b)

    @staticmethod
    def _verify(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert not np.all(np.isclose(a, b)), 'Endpoints a and b cannot be coincident'
        return a.copy(), b.copy()

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def b(self) -> np.ndarray:
        return self._b


class Polygon:

    def __init__(self, points: np.ndarray, verify: bool = True):
        self._points = self._verify(points) if verify else points
        self._n_points = self._points.shape[0]

    @staticmethod
    def _verify(points: np.ndarray) -> np.ndarray:
        assert len(points.shape) == 2, 'Points array must have shape Nx2'
        assert points.shape[0] >= 3, 'Polygons must be made from at least 3 vertices'
        assert points.shape[1] == 2, 'Points must be two-dimensional'

        points = points.copy()
        indices = np.arange(points.shape[0])

        # Get the bottom-most point
        min_y = np.min(points[:, 1])
        min_y_args = points[:, 1] == min_y

        # Break ties with minimum x-value
        if np.sum(min_y_args) > 1:
            min_x = np.min(points[min_y_args, 0])
            min_y_args = np.logical_and(points[:, 0] == min_x, points[:, 1] == min_y)

        bottom_left = indices[min_y_args].item()

        # Reorder the points according the bottom left points
        points = np.vstack([points[bottom_left:, :], points[:bottom_left, :]])

        return points

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def points(self) -> np.ndarray:
        return self._points

    def contains(self, point: np.ndarray) -> bool:
        return inside_polygon(point, self)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)

        patch = patches.Polygon(self._points, True)
        patch_collection = PatchCollection([patch], **kwargs)
        ax.add_collection(patch_collection)


class Square(Polygon):

    def __init__(self, center: np.ndarray, side_length: float):
        if len(center.shape) < 2:
            center = center[np.newaxis]
        points = side_length / 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) + np.repeat(center, 4, axis=0)
        super().__init__(points, verify=False)


def points_orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < EPS:
        return 0
    if val > 0:
        return 1
    return 2


def on_segment(q: np.ndarray, segment: Segment) -> bool:
    p, r = segment.a, segment.b
    colinear = points_orientation(p, q, r) == 0
    in_x_range = (q[0] - max(p[0], r[0]) < EPS) and (q[0] - min(p[0], r[0]) > -EPS)
    in_y_range = (q[1] - max(p[1], r[1]) < EPS) and (q[1] - min(p[1], r[1]) > -EPS)
    if colinear and in_x_range and in_y_range:
        return True
    return False


def lines_intersect(line1: Segment, line2: Segment) -> bool:
    a = line1.a
    b = line1.b
    c = line2.a
    d = line2.b

    o1 = points_orientation(a, b, c)
    o2 = points_orientation(a, b, d)
    o3 = points_orientation(c, d, a)
    o4 = points_orientation(c, d, b)

    if o1 != o2 and o3 != o4:
        return True

    if on_segment(a, line2):
        return True
    if on_segment(b, line2):
        return True
    if on_segment(c, line1):
        return True
    if on_segment(d, line1):
        return True

    return False


class Circle:

    def __init__(self, center: np.ndarray, radius: float):
        self._center, self._radius = self._verify(center, radius)

    @staticmethod
    def _verify(center: np.ndarray, radius: float) -> Tuple[np.ndarray, float]:
        assert center.size == 2, 'Center must be a 2D point'
        assert radius > 0, 'Radius must be a value greater than zero'
        return center.copy(), radius

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def plot(self, ax=None, **kwargs):
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(1, 1, 1)
        patch = patches.Circle((self._center[0], self._center[1]), self._radius, fill=False)
        patch_collection = PatchCollection([patch], **kwargs)
        ax.add_collection(patch_collection)


def inside_polygon(point: np.ndarray, polygon: Polygon):
    def on_vertex(q: np.ndarray, segment: Segment) -> bool:
        p, r = segment.a, segment.b
        if abs(p[1] - q[1]) < EPS or abs(r[1] - q[1]) < EPS:
            return True
        return False

    def count_segment(q: np.ndarray, segment: Segment) -> bool:
        p = segment.a
        if (q[1] - p[1]) > 0:
            return True
        return False

    extreme = np.array([LARGE, point[1]])
    ray = Segment(point, extreme)

    count = 0
    for a, b in zip(WrapIndex(polygon.n_points), WrapIndex(polygon.n_points, start_index=1)):
        poly_segment = Segment(polygon.points[a, :], polygon.points[b, :])
        if lines_intersect(ray, poly_segment):
            if on_segment(point, poly_segment):
                return True
            if on_vertex(point, poly_segment):
                if count_segment(point, poly_segment):
                    count += 1
            else:
                count += 1

    if count % 2 == 1:
        return True
    return False


class TriangulationError(Exception):
    pass


def cut_ear(polygon: Polygon) -> Tuple[Optional[Polygon], Optional[Polygon]]:
    n = polygon.n_points
    if n == 3:
        return polygon, None

    def valid_ear():
        for u, v, w in zip(WrapIndex(n, r), WrapIndex(n, r + 1, p), WrapIndex(n, r + 2)):
            if points_orientation(polygon.points[u], polygon.points[v], polygon.points[w]) == 1:
                if inside_polygon(polygon.points[v], ear_candidate):
                    return False
        return True

    for p, q, r in zip(WrapIndex(n, start_index=-1), WrapIndex(n), WrapIndex(n, start_index=1)):
        if points_orientation(polygon.points[p], polygon.points[q], polygon.points[r]) == 2:
            ear_candidate = Polygon(np.vstack([polygon.points[p], polygon.points[q], polygon.points[r]]))
            if valid_ear():
                base_points = []
                for b in WrapIndex(n, start_index=r, stop_index=q):
                    base_points.append(polygon.points[b])
                base_polygon = Polygon(np.vstack(base_points))
                return base_polygon, ear_candidate

    raise TriangulationError(f'Polygon has {n} points but no ears were found')


def triangulate(polygon: Polygon) -> List[Polygon]:
    triangles = []
    base, ear = cut_ear(polygon)
    if ear is not None:
        triangles.append(ear)
        for triangle in triangulate(base):
            triangles.append(triangle)
    else:
        triangles.append(base)

    return triangles


def segment_polar_angles(polygon: Polygon) -> List[float]:
    angles = []
    for a, b in zip(WrapIndex(polygon.n_points), WrapIndex(polygon.n_points, 1)):
        point_a = polygon.points[a]
        point_b = polygon.points[b]
        angle = atan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
        if angle < -EPS:
            angle = angle + 2 * pi
        elif abs(angle) < EPS:
            angle = 0.
        angles.append(angle)

    return angles


def convex_minkowski_sum(polygon_a: Polygon, polygon_b: Polygon):
    a_points = np.vstack([polygon_a.points, polygon_a.points[0]])
    a_angles = segment_polar_angles(polygon_a)
    a_angles.append(2 * pi)
    n = polygon_a.n_points

    b_points = np.vstack([polygon_b.points, polygon_b.points[0]])
    b_angles = segment_polar_angles(polygon_b)
    b_angles.append(2 * pi)
    m = polygon_b.n_points

    i, j = 0, 0
    sum_points = []
    while i < n or j < m:
        sum_points.append(a_points[i] + b_points[j])

        if a_angles[i] - b_angles[j] < -EPS:
            i += 1
        elif a_angles[i] - b_angles[j] > EPS:
            j += 1
        else:
            i += 1
            j += 1

    return Polygon(np.vstack(sum_points))


if __name__ == '__main__':
    points = np.array([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
    square = Polygon(points)

    points = np.array([[1, 1], [0, 1], [1, 0]]) + np.repeat(np.array([[2, 2]]), 3, axis=0)
    triangle = Polygon(points)

    poly_sum = convex_minkowski_sum(square, triangle)

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.set_aspect('equal', 'box')
    square.plot(ax, color='g')
    triangle.plot(ax, color='b')
    poly_sum.plot(ax, alpha=.5, color='r')
    plt.show()
