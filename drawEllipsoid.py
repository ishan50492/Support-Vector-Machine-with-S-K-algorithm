from PIL import Image,ImageDraw
import random



def drawEllipsoid(draw):

    # Random position
    x1 = random.randint(1, 24)
    y1 = random.randint(1, 24)

    # Random size
    s = random.randint(1, min(min(x1, abs(25 - x1)), min(y1, abs(25 - y1))))

    # Random thickness
    outline_width = random.randint(1, s)

    outline_color = "black"

    top_left = (x1 - s, y1 - s)
    bottom_right = (x1 + s, y1 + s)

    draw.ellipse((top_left, bottom_right), outline=(0,0,0))