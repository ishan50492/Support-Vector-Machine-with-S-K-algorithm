import os.path
import sys
import numpy as np
import drawCircle
import drawRectangle
import drawCross
import drawWaves
import drawStar
import random

#Checking number of arguments
if len(sys.argv)!=3:
   print("Invalid number of arguments")
   exit()

folder_name=sys.argv[1]
num_examples=int(sys.argv[2])

folder_name="20000WV"
num_examples=20000

#Creating a folder which will contain the dataset
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, folder_name)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

#Changing the current working directory to inside the folder
os.chdir(current_directory+'/'+folder_name)

#Start drawing
#0 for Circle
#1 for Rectangle
#2 for Cross
#3 for Waves
#4 for Star

draw=[0,1,2,3,4]

def drawV(num_examples):
    for i in range(0,num_examples):

        r=random.randint(0,4)

        if r==0:
            drawCircle.drawCircle(1)
        elif r==1:
            drawRectangle.drawRectangle(1)
        elif r==2:
            drawCross.drawCross(1)
        elif r==3:
            drawWaves.drawWaves(1)
        elif r==4:
            drawStar.drawStar(1)

def drawWV(num_examples):
    for i in range(0,num_examples):

        r=random.randint(0,4)

        if r==0:
            drawCircle.drawCircleWV(1,i)
        elif r==1:
            drawRectangle.drawRectangleWV(1,i)
        elif r==2:
            drawCross.drawCrossWV(1,i)
        elif r==3:
            drawWaves.drawWavesWV(1,i)
        elif r==4:
            drawStar.drawStarWV(1,i)



drawWV(num_examples)
