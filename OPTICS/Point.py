from dataclasses import dataclass

from pyspark.mllib.linalg import Vector


@dataclass(frozen=False)
class Point:
    vector: Vector

    @property
    def x(self):
        return self.vector[0]

    @property
    def y(self):
        return self.vector[1]

    def distanceSquared(self, other) -> float:
        dx = other.x - self.x
        dy = other.y - self.y
        return (dx * dx) + (dy * dy)

    def __hash__(self):
        return self.vector.__hash__()
