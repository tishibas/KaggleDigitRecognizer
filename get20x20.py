#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import cv
import numpy as np
import sys

data = np.genfromtxt('train.csv', delimiter=',')

train_data = data[1:,1:]
label = data[1:,0]

#exclude margin
def trimArea(m):
    for j in range(28):
        for i in range(28):
            if 0 != m[j,i]:
                top_y = j
                break
        else:
            continue
        break

    for i in range(28):
        for j in range(28):
            if 0 != m[j,i]:
                top_x = i
                break
        else:
            continue
        break

    for j in range(28)[::-1]:
        for i in range(28):
            if 0 != m[j,i]:
                bottom_y = j
                break
        else:
            continue
        break

    for i in range(28)[::-1]:
        for j in range(28):
            if 0 != m[j,i]:
                bottom_x = i
                break
        else:
            continue
        break

    sub = cv.GetSubRect(m, (top_x, top_y, bottom_x - top_x + 1, bottom_y - top_y + 1))
    #cv.SaveImage("sub.png",sub)
    sub2 = cv.CloneMat(sub)
    square = cv.CreateMat(20, 20, cv.CV_8UC1)
    cv.Resize(sub2, square)
    return square

squares = []
count = 0
for d in train_data:
    count = count + 1
    tmp = cv.fromarray(d.reshape(28,28))
    dmat = cv.CreateMat(28,28,cv.CV_8UC1)
    cv.Convert(tmp,dmat)
    #cv.SaveImage("original.png", dmat)
    square = trimArea(dmat)
    squareArray = np.asarray(square)
    squares.append(squareArray.flatten())
    #cv.SaveImage('test.png', nonZeroArea)
    #sys.exit()

#convert list to ndarray
squares = np.array(squares)

#normalize
avg = np.mean(squares, axis=1)
sd = np.std(squares, axis=1)
squares = (squares - avg.reshape(len(squares),1)) / sd.reshape(len(squares),1)

np.savetxt("train_20x20_normalized.txt", squares)
