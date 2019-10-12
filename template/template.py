from model.rectangle import Rectangle

screen_template = Rectangle(0, 180, 500, 400)

# Relative to screen_template

w = 70
h = 70
x1 = 400
y1 = 0
game_over_template = Rectangle(x1, y1, x1 + w, y1 + h)
