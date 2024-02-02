from enum import Enum, auto
from drawing import Container

class SHAPE(Enum):
    CIRCLE = auto()
    RECTANGLE = auto()

class RectCirc(Container):
    """
    A simple example canvas to draw circles and rectangles.

    Controls
    --------
    - Left click: Draw a circle.
    - Right click: Draw a rectangle.
    - Middle click: Clear the canvas.
    """

    def setup(self):
        self.elements = []

        self.bind("<Button-1>", self.draw_circle)
        self.bind("<Button-2>", lambda _: self.elements.clear())
        self.bind("<Button-3>", self.draw_rectangle)

    def draw(self):
        self.draw_grid(50, 50, 10)
        for element in self.elements:
            if element['type'] == SHAPE.CIRCLE:
                self.canvas.create_oval(element['x'] - 10, element['y'] - 10,
                                        element['x'] + 10, element['y'] + 10,
                                        fill="red")
            elif element['type'] == SHAPE.RECTANGLE:
                self.canvas.create_rectangle(element['x'] - 10, element['y'] - 10,
                                            element['x'] + 10, element['y'] + 10,
                                            fill="blue")
                
    def draw_circle(self, event):
        self.elements.append({
            'x': event.x,
            'y': event.y,
            'type': SHAPE.CIRCLE
        })

    def draw_rectangle(self, event):
        self.elements.append({
            'x': event.x,
            'y': event.y,
            'type': SHAPE.RECTANGLE
        })

RectCirc(500, 500)
