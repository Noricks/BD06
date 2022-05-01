from pyspark.mllib.linalg import Vector
from dataclasses import dataclass, field


# from OPTICS.DBSCANLabeledPoint import DBSCANLabeledPoint

@dataclass(frozen=False)
class DBSCANPoint:
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
