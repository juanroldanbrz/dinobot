class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_tuple(self):
        return self.x1, self.y1, self.x2, self.y2

    def to_points(self):
        return [(self.x1, self.y1), (self.x2, self.y2)]

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def shape(self):
        return self.height(), self.width()

    def area(self):
        return self.width() * self.height()

    def relativize_from(self, rectangle):
        return Rectangle(self.x1 + rectangle.x1,
                         self.y1 + rectangle.y1,
                         self.x2 + rectangle.x1,
                         self.y2 + rectangle.y1)
