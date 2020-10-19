import numpy as np

from geometry import Circle


def distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.linalg.norm(point1 - point2)


def circle_distance(point: np.ndarray, circle: Circle) -> float:
    return distance(point, circle.center) - circle.radius


class CircleObstacle(Circle):

    def __init__(self, center: np.ndarray, radius: float, safe_distance: float = None):
        super().__init__(center, radius)
        self._qstar = safe_distance if safe_distance is not None else .1 * radius

    @property
    def safe_distance(self) -> float:
        return self._qstar
