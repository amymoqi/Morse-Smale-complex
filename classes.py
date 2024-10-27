class criticalPoint:
    def __init__(self, key: list, value: int, wedge:int, index: int):
        self.value = value
        self.wedge = wedge
        self.key = key
        self.index = index
        if self.wedge == 0:
            self.PoincareIndex = 1
        elif self.wedge > 1:
            self.PoincareIndex = -1

    def __repr__(self):
        return f"criticalPoint(position={self.key}, value={self.value}, type={self.type()})"

    def type(self):
        if self.wedge == 0:
            if self.index == 0:
                return 'local minima'
            else:
                return 'local maxima'
        elif self.wedge == 1:
            return "regular point"
        elif self.wedge == 2:
            return "simple saddle"
        elif self.wedge >= 3:
            return  str(self.wedge-1) + "-fold saddle"
