from pyspark.mllib.linalg import Vector
from enum import Enum, auto
from common.Point import Point

"""
 Companion constants for labeled points
"""


class Flag(Enum):
    Border = auto()
    Core = auto()
    Noise = auto()
    NotFlagged = auto()


class LabeledPoint(Point):
    Unknown = 0
    Undefined = -1.0

    def __init__(self, vector: Vector):
        super().__init__(vector)
        self.flag = Flag.NotFlagged
        self.cluster = LabeledPoint.Unknown
        self.visited = False
        self.reachDist = LabeledPoint.Undefined

    def toString(self) -> str:
        return "$vector,$cluster,$flag"
