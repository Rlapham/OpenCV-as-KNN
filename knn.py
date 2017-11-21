import os
import csv
from collections import Counter
import sys
import math
from math import *
from vidpy import Composition, Clip
from numpy import interp
import numpy as np
import cv2

count = 0

def neighbors(k, trained_points, new_point):
    neighbor_distances = {}

    for point in trained_points:
        if point not in neighbor_distances:
            neighbor_distances[point] = euclidean_distance(point, new_point)

    least_common = sorted(neighbor_distances.items(), key = lambda x: x[1])

    k_nearest_neighbors = zip(*least_common[:k])

    return list(k_nearest_neighbors[0])


def knn_classifier(neighbors, input_data):
    knn = [input_data[i] for i in neighbors]
    # print knn
    knn = Counter(knn)
    # print knn
    classifier, _ = knn.most_common(1)[0]
    # print knn.most_common(1)
    return classifier

def euclidean_distance(x, y):
    if len(x) != len(y):
        return "Error: try equal length vectors"
    else:
        return sqrt(sum([(x[i]-y[i])**2 for i in xrange(len(y))]))

euclidean_distance([1,2,3],[4,5])

###opencv
face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
img = cv2.imread('beatles2.jpg')

clips = []
totvals1 = []
totvals1.append(0)
temppoints = []
points = []
col = (0, 0, 0)

x_ = []
y_ = []
z_ = []

test_x = []
test_y = []

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    if (count == 0):
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(img,(x+1,y+1),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(img,(x+2,y+2),(x+w,y+h),(255,0,0),2)

        z_.append("A")
    if (count == 1):
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x+1,y+1),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(img,(x+2,y+2),(x+w,y+h),(0,255,0),2)


        z_.append("B")

    if (count == 2):
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(img,(x+1,y+1),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(img,(x+2,y+2),(x+w,y+h),(0,0,255),2)

        z_.append("C")


    x_.append(float(x))
    y_.append(float(y))
    count += 1

for i in range (0, 1600, 50):
    for a in range (0, 1600, 50):
        # img[i][a] = (255, 0, 0)
        test_x.append(float(i))
        test_y.append(float(a))

coordinates = zip(x_,y_)
input_data = {coordinates[i]:z_[i] for i in xrange(len(coordinates))}


test_coordinates = zip(test_x,test_y)
# print len(test_coordinates) 336
a = neighbors(1, input_data.keys(), test_coordinates[120])
knn_classifier(a, input_data)
# print input_data
print a
print "class: "
print test_coordinates[20]
print knn_classifier(a, input_data)





results = {}
# print test_coordinates
for item in test_coordinates:
    # print item
    results[item] = knn_classifier(neighbors(1,input_data.keys(), item), input_data)
    tempx = int(item[0])
    tempy = int(item[1])
    print results[item]


    if (results[item] == "A"):
        for i in range (0, 10):
            for a in range (0, 10):
                newx = tempx + i
                newy = tempy + i
                img[newx][newy] = (255, 0, 0)
                img[newx+1][newy] = (255, 0, 0)
                newx = tempx + a
                newy = tempy - a
                img[newx][newy] = (255, 0, 0)
                img[newx+1][newy] = (255, 0, 0)
    if (results[item] == "B"):
        for i in range (0, 10):
            for a in range (0, 10):
                newx = tempx + i
                newy = tempy + i
                img[newx][newy] = (0, 255, 0)
                img[newx+1][newy] = (0, 255, 0)
                newx = tempx + a
                newy = tempy - a
                img[newx][newy] = (0, 255, 0)
                img[newx+1][newy] = (0, 255, 0)

    if (results[item] == "C"):
        for i in range (0, 10):
            for a in range (0, 10):
                newx = tempx + i
                newy = tempy + i
                img[newx][newy] = (0, 0, 255)
                img[newx+1][newy] = (0, 0, 255)
                newx = tempx + a
                newy = tempy - a
                img[newx][newy] = (0, 0, 255)
                img[newx+1][newy] = (0, 0, 255)
    # if (results[item] == 3):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (0, 255, 255)
    # if (results[item] == 4):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (255, 0, 0)
    # if (results[item] == 5):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (0, 255, 0)
    # if (results[item] == 6):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (0, 0, 255)
    #
    # if (results[item] == 7):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (100, 100, 100)
    # if (results[item] == 8):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (100, 100, 0)
    # if (results[item] == 9):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (100, 0, 100)
    # if (results[item] == 10):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (0, 100, 100)
    # if (results[item] == 11):
    #     for i in range (0, 5):
    #         newx = tempx + i
    #         newy = tempy + i
    #         img[newx][newy] = (100, 0, 0)

# print results


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("beatlesknn3.jpg", img)     # save frame as JPEG file

#######
