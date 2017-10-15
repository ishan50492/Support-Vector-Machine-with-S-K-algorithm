from PIL import Image,ImageDraw
import random
import drawEllipsoid

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)


def drawRectangle(num_examples):

    for i in range(0,num_examples):

        im = Image.new('RGBA', (25, 25), "white")
        draw = ImageDraw.Draw(im)

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

        draw_rectangle(draw, (top_left, bottom_right), color=outline_color, width=outline_width)

        drawEllipsoid.drawEllipsoid(draw)

        imb = Image.new('RGBA', (25, 25), "white")
        imf = Image.composite(im, imb, im)

        # Saving the image
        imf.save(str(i + 1) + "_Q.png", "PNG")

def drawRectangleWV(num_examples,k):

    for i in range(0,num_examples):

        im = Image.new('RGBA', (25, 25), "white")
        draw = ImageDraw.Draw(im)

        # Random position
        x1 = 12.5
        y1 = 12.5

        s = 7

        # Random thickness
        outline_width = 1


        outline_color = "black"

        top_left = (x1 - s, y1 - s)
        bottom_right = (x1 + s, y1 + s)

        draw_rectangle(draw, (top_left, bottom_right), color=outline_color, width=outline_width)



        imb = Image.new('RGBA', (25, 25), "white")
        imf = Image.composite(im, imb, im)

        # Saving the image
        imf.save(str(k + 1) + "_Q.png", "PNG")
