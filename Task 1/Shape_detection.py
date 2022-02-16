

import numpy as np
import cv2 as cv

img = cv.imread('img1.jpeg') #replace the name/path of image to try different images
imgOriginal = img;
imgGrey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(imgGrey,100,200)
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

class Shape:
  def __init__(shape, shapeName, shapeColor, shapeCentroidX, shapeCentroidY, shapeArea):
    shape.name = shapeName
    shape.color = shapeColor
    shape.center = [shapeCentroidX, shapeCentroidY]
    shape.area = shapeArea

  def print(shape):
    print("ShapeType:",shape.name)
    print("ShapeColor[BGR]:",shape.color)
    print("ShapeCentroid:",shape.center)
    print("ShapeArea:",shape.area)

for i in range(len(contours)):
    print(i)
    cnt = contours[i]
    M = cv.moments(cnt)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    area = cv.contourArea(cnt)
    
    epsilon = 0.01*cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon,True)

    x,y,w,h = cv.boundingRect(cnt)
    
    name=''
    if(len(approx) == 3):
        name = 'Triangle'
    elif len(approx) == 4: 
        aspectRatio = float(w)/h
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
           name='Square'
        else: 
            name='Rectangle'
    elif(len(approx)==5):
        name='pentagon'
    else:
        name='Circle'
    
    img = cv.circle(img, (cx, cy), 3, (0, 0, 0), -1)
    
    cv.putText(img, str(i), (cx+20,cy+20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
    
    s = Shape(name, np.array(cv.mean(imgOriginal[y:y+h,x:x+w])).astype(np.uint8), cx, cy, area)
    s.print()
    
    
    
cv.drawContours(img, contours, -1, (0, 0, 0), 3)

cv.imshow("Img", img)

cv.waitKey(0)
cv.destroyAllWindows()
    
