from pyspark.mllib.linalg import Vector
from dataclasses import dataclass, field


# from DBSCAN.DBSCANLabeledPoint import DBSCANLabeledPoint

@dataclass(frozen=False)
class DBSCANPoint:
    vector: Vector
    x: float = field(init=False)
    y: float = field(init=False)

    def __post_init__(self):
        self.x = self.vector[0]
        self.y = self.vector[1]

    # def __init__(self, vector: Vector):
    #     self.vector = vector  # TODO: meaning unclear
    #     self.x = vector[0]
    #     self.y = vector[1]

    def distanceSquared(self, other) -> float:
        dx = other.x - self.x
        dy = other.y - self.y
        return (dx * dx) + (dy * dy)
