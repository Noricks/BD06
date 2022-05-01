from pyspark.mllib.linalg import Vector
from enum import Enum, auto
from OPTICS.Point import DBSCANPoint

"""
 Companion constants for labeled points
"""

class Flag(Enum):
    Border = auto()
    Core = auto()
    Noise = auto()
    NotFlagged = auto()

class DBSCANLabeledPoint(DBSCANPoint):
    Unknown = 0
    Undefined = -1.0
    def __init__(self, vector: Vector):
        super().__init__(vector)
        self.flag = Flag.NotFlagged
        self.cluster = DBSCANLabeledPoint.Unknown
        self.visited = False
        self.reachDist = DBSCANLabeledPoint.Undefined

    def toString(self) -> str:
        return "$vector,$cluster,$flag"

