class Line(object):
    """
    Convinience Class
    """
    def __init__(self, start, direction):
        self.point0 = start
        self.point1 = self.add(start, direction)
        self.point2 = self.add(start, self.scale(direction, 2))
      

    def add(self, a, b):
        return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
    def scale(self, a, c):
        return [c*a[0], c*a[1], c*a[2]]

    def onLine(self, point):
        if point == point0 or point == point1 or point == point2:
            return True
        else:
            return False

    def __str__(self):
        return f"Line: Start - {self.point0}, mid {self.point1}, end {self.point2}"



    
def get_win_lines():
    """
    Returns a list of all winning lines in the board
    """
    win_lines = []
    win_lines.append(Line([0,0,0], [0,0,1]))
    win_lines.append(Line([0,1,0], [0,0,1]))
    win_lines.append(Line([0,2,0], [0,0,1]))
    win_lines.append(Line([0,0,0], [0,1,0]))
    win_lines.append(Line([0,0,1], [0,1,0]))
    win_lines.append(Line([0,0,2], [0,1,0]))
    win_lines.append(Line([0,0,0], [0,1,1]))
    win_lines.append(Line([0,0,2], [0,1,-1]))

    win_lines.append(Line([1,0,0], [0,0,1]))
    win_lines.append(Line([1,1,0], [0,0,1]))
    win_lines.append(Line([1,2,0], [0,0,1]))
    win_lines.append(Line([1,0,0], [0,1,0]))
    win_lines.append(Line([1,0,1], [0,1,0]))
    win_lines.append(Line([1,0,2], [0,1,0]))
    win_lines.append(Line([1,0,0], [0,1,1]))
    win_lines.append(Line([1,0,2], [0,1,-1]))

    win_lines.append(Line([2,0,0], [0,0,1]))
    win_lines.append(Line([2,1,0], [0,0,1]))
    win_lines.append(Line([2,2,0], [0,0,1]))
    win_lines.append(Line([2,0,0], [0,1,0]))
    win_lines.append(Line([2,0,1], [0,1,0]))
    win_lines.append(Line([2,0,2], [0,1,0]))
    win_lines.append(Line([2,0,0], [0,1,1]))
    win_lines.append(Line([2,0,2], [0,1,-1]))

    win_lines.append(Line([0,0,0], [1,0,0]))
    win_lines.append(Line([0,1,0], [1,0,0]))
    win_lines.append(Line([0,2,0], [1,0,0]))
    win_lines.append(Line([0,0,1], [1,0,0]))
    win_lines.append(Line([0,0,2], [1,0,0]))
    win_lines.append(Line([0,1,1], [1,0,0]))
    win_lines.append(Line([0,1,2], [1,0,0]))
    win_lines.append(Line([0,2,1], [1,0,0]))
    win_lines.append(Line([0,2,2], [1,0,0]))

    win_lines.append(Line([0,0,0], [1,1,0]))
    win_lines.append(Line([0,2,0], [1,-1,0]))
    win_lines.append(Line([0,0,0], [1,0,1]))
    win_lines.append(Line([0,0,2], [1,0,-1]))
    win_lines.append(Line([0,0,2], [1,1,0]))
    win_lines.append(Line([0,2,2], [1,-1,0]))
    win_lines.append(Line([0,2,0], [1,0,1]))
    win_lines.append(Line([0,2,2], [1,0,-1]))

    win_lines.append(Line([0,0,0], [1,1,1]))
    win_lines.append(Line([0,0,1], [1,1,0]))
    win_lines.append(Line([0,0,2], [1,1,-1]))
    win_lines.append(Line([0,1,0], [1,0,1]))
    win_lines.append(Line([0,1,2], [1,0,-1]))
    win_lines.append(Line([0,2,0], [1,-1,1]))
    win_lines.append(Line([0,2,1], [1,-1,0]))
    win_lines.append(Line([0,2,2], [1,-1,-1]))
    return win_lines
