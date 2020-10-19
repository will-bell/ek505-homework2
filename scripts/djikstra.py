from heapq import heappop, heappush
from typing import Any, List, Dict
import itertools
from abc import abstractmethod
import matplotlib as plt


class DjikstraSearchError(Exception):
    pass


class PriorityQueue:

    def __init__(self):
        self._queue = []
        self._item_finder = {}
        self._counter = itertools.count()

    def push(self, item: Any, value: float):
        count = next(self._counter)
        entry = (value, count, item)
        self._item_finder[item] = entry
        heappush(self._queue, entry)

    def pop(self):
        if self._queue:
            value, count, item = heappop(self._queue)
            del self._item_finder[item]
            return item
        raise KeyError('Trying to pop from an empty queue')

    def empty(self) -> bool:
        return not self._queue


class DjikstraGraph:

    @abstractmethod
    def get_neighbors(self, a: Any) -> List[Any]:
        pass

    @abstractmethod
    def evaluate_cost(self, a: Any, b: Any) -> float:
        pass


def reconstruct_path(came_from: Dict[Any, Any], goal: Any) -> List[Any]:
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)

    path.append(current)
    return list(reversed(path))


def djikstra(graph: DjikstraGraph, start: Any, goal: Any) -> List[Any]:
    priority_queue = PriorityQueue()
    priority_queue.push(start, 0)

    came_from = {start: None}
    cost_so_far = {start: 0}

    while not priority_queue.empty():
        current = priority_queue.pop()
        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in graph.get_neighbors(current):
            traversal_cost = graph.evaluate_cost(current, neighbor)
            candidate_cost = cost_so_far[current] + traversal_cost

            expand = False
            if neighbor in cost_so_far.keys():
                if candidate_cost < cost_so_far[neighbor]:
                    expand = True
            else:
                expand = True

            if expand:
                came_from[neighbor] = current
                cost_so_far[neighbor] = candidate_cost
                priority_queue.push(neighbor, candidate_cost)

    raise DjikstraSearchError('Open set empty without finding a path')