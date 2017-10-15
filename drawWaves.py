import numpy as np
from PIL import Image,ImageDraw
import random
import drawEllipsoid

def drawWaves(num_examples):

    for j in range(0,num_examples):

        im = Image.new('RGBA', (25, 25), "white")
        draw = ImageDraw.Draw(im)

        x = np.arange(0,25, 0.01);

        #Random thickness
        thickness = random.uniform(0.001,0.1)
        y = 5 * np.sin(x/4) + 7.5

        draw.point(list(zip(x, y)), fill=(255, 0, 0))

        for i in range(0, x.size):
            draw.ellipse((x[i] - thickness, y[i] - thickness, x[i] + thickness, y[i] + thickness), fill=(0, 0, 0))

        y = y + 5
        for i in range(0, x.size):
            draw.ellipse((x[i] - thickness, y[i] - thickness, x[i] + thickness, y[i] + thickness), fill=(0, 0, 0))

        y = y + 5
        for i in range(0, x.size):
            draw.ellipse((x[i] - thickness, y[i] - thickness, x[i] + thickness, y[i] + thickness), fill=(0, 0, 0))

        drawEllipsoid.drawEllipsoid(draw)

        #random rotation
        r=random.randint(0,360)
        im=im.rotate(r)

        imb = Image.new('RGBA', (25, 25), "white")
        imf = Image.composite(im, imb, im)


        imf.save(str(j + 1) + "_W.png", "PNG")

def drawWavesWV(num_examples,k):

    for j in range(0,num_examples):

        im = Image.new('RGBA', (25, 25), "white")
        draw = ImageDraw.Draw(im)

        x = np.arange(0,25, 0.01);

        #Random thickness
        thickness = 0.01
        y = 5 * np.sin(x/4) + 7.5

        draw.point(list(zip(x, y)), fill=(255, 0, 0))

        for i in range(0, x.size):
            draw.ellipse((x[i] - thickness, y[i] - thickness, x[i] + thickness, y[i] + thickness), fill=(0, 0, 0))

        y = y + 5
        for i in range(0, x.size):
            draw.ellipse((x[i] - thickness, y[i] - thickness, x[i] + thickness, y[i] + thickness), fill=(0, 0, 0))

        y = y + 5
        for i in range(0, x.size):
            draw.ellipse((x[i] - thickness, y[i] - thickness, x[i] + thickness, y[i] + thickness), fill=(0, 0, 0))

        im=im.rotate(90)
        imb = Image.new('RGBA', (25, 25), "white")
        imf = Image.composite(im, imb, im)


        imf.save(str(k + 1) + "_W.png", "PNG")

