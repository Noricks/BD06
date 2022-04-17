# postponed evaluation (for typing), which will become the default behavior in Python 3.10
from __future__ import annotations

from dataclasses import dataclass

from DBSCAN.DBSCANPoint import DBSCANPoint


@dataclass(frozen=True)
class DBSCANRectangle:
    """
    A rectangle with a left corner of (x, y) and a right upper corner of (x2, y2)

    dataclass: similar to case class in scala:
            url: https://stackoverflow.com/questions/51342228/python-equivalent-of-scala-case-class
    """
    x: float
    y: float
    x2: float
    y2: float    
                

    def contains(self, i: DBSCANPoint | DBSCANRectangle) -> bool:
        if isinstance(i, DBSCANPoint):
            # Returns whether point is contained by this box
            point = i
            return self.x <= point.x <= self.x2 and self.y <= point.y <= self.y2
        else:
            # Returns whether other is contained by this box
            other = i
            return self.x <= other.x and other.x2 <= self.x2 and self.y <= other.y and other.y2 <= self.y2

    # Returns whether point is contained by this box
    # def contains(self, point: ) -> bool:
    #     return self.x <= point.x <= self.x2 and self.y <= point.y <= self.y2


    # Returns a new boself.x from shrinking this box by the given amount
    def shrink(self, amount: float)  -> DBSCANRectangle:
        return DBSCANRectangle(self.x + amount, self.y + amount, self.x2 - amount, self.y2 - amount)


    """
        Returns a whether the rectangle contains the point, and the point
        is not in the rectangle's border
    """
    def almostContains(self, point: DBSCANPoint) -> bool:
        return self.x < point.x < self.x2 and self.y < point.y < self.y2
