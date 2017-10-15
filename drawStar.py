from PIL import Image,ImageDraw
import random
import drawEllipsoid

def draw_star(draw, coordinates, color, width=1):
    for j in range(0,9):
        draw.line((coordinates[j][0], coordinates[j][1], coordinates[j+1][0], coordinates[j+1][1]), fill=color,width=width)
    draw.line((coordinates[9][0], coordinates[9][1], coordinates[0][0], coordinates[0][1]), fill=color,width=width)




def drawStar(num_examples):

    for i in range(0,num_examples):

        # Create a white image with length=1000,breadth=1000 and color: 255,255,255 White
        im = Image.new('RGBA', (25, 25),"white")

        #Random scale value
        a =random.uniform(20,25)

        # Now lets draw on the white image
        drawing = ImageDraw.Draw(im)
        p1 = (a / 2, 0)
        p2 = (3 * a / 8, 3 * a / 8)
        p3 = (0, a / 2)
        p4 = (3 * a / 8, 5 * a / 8)
        p5 = (a / 4, a)
        p6 = (a / 2, 3 * a / 4)
        p7 = (3 * a / 4, a)
        p8 = (5 * a / 8, 5 * a / 8)
        p9 = (a, a / 2)
        p10 = (5 * a / 8, 3 * a / 8)

        outline_color = 'black'
        outline_width = 1

        draw_star(drawing, (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10), color=outline_color, width=outline_width)

        #Generate random rotation value
        r=random.randint(0,360)
        im=im.rotate(r)

        drawEllipsoid.drawEllipsoid(drawing)

        imb=Image.new('RGBA', (25, 25),"white")
        imf=Image.composite(im,imb,im)

        #Saving the image
        imf.save(str(i + 1) + "_S.png", "PNG")

def drawStarWV(num_examples,k):

    for i in range(0,num_examples):

        # Create a white image with length=1000,breadth=1000 and color: 255,255,255 White
        im = Image.new('RGBA', (25, 25),"white")

        #Random scale value
        a =25

        # Now lets draw on the white image
        drawing = ImageDraw.Draw(im)
        p1 = (a / 2, 0)
        p2 = (3 * a / 8, 3 * a / 8)
        p3 = (0, a / 2)
        p4 = (3 * a / 8, 5 * a / 8)
        p5 = (a / 4, a)
        p6 = (a / 2, 3 * a / 4)
        p7 = (3 * a / 4, a)
        p8 = (5 * a / 8, 5 * a / 8)
        p9 = (a, a / 2)
        p10 = (5 * a / 8, 3 * a / 8)

        outline_color = 'black'
        outline_width = 1

        draw_star(drawing, (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10), color=outline_color, width=outline_width)

        imb=Image.new('RGBA', (25, 25),"white")
        imf=Image.composite(im,imb,im)

        #Saving the image
        imf.save(str(k + 1) + "_S.png", "PNG")

