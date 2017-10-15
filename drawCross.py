from PIL import Image,ImageDraw,ImageFont
import random
import drawEllipsoid


def drawCross(num_examples):

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

        draw.line((x1,y1-s,x1,y1+s), fill=(0, 0, 0), width=outline_width)  # draw a black line
        draw.line((x1-s, y1, x1+s,y1), fill=(0, 0, 0), width=outline_width)  # draw a black line

        drawEllipsoid.drawEllipsoid(draw)

        imb = Image.new('RGBA', (25, 25), "white")
        imf = Image.composite(im, imb, im)

        # Saving the image
        imf.save(str(i + 1) + "_P.png", "PNG")

def drawCrossWV(num_examples,k):

    for i in range(0,num_examples):
        im = Image.new('RGBA', (25, 25), "white")
        draw = ImageDraw.Draw(im)

        # Random position
        x1 = 12.5
        y1 = 12.5

        # Random size
        s = 5

        # Random thickness
        outline_width = 1

        outline_color = "black"

        draw.line((x1,y1-s,x1,y1+s), fill=(0, 0, 0), width=outline_width)  # draw a black line
        draw.line((x1-s, y1, x1+s,y1), fill=(0, 0, 0), width=outline_width)  # draw a black line


        imb = Image.new('RGBA', (25, 25), "white")
        imf = Image.composite(im, imb, im)

        # Saving the image
        imf.save(str(k + 1) + "_P.png", "PNG")
