# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import math
from hw2_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib import pyplot as plt


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.comboBox.addItems([str(x) for x in range(1, 16)])

    # button for problem 1.1
    def on_btn1_1_click(self):
        img = cv2.imread(os.path.join('images', 'plant.jpg'), cv2.IMREAD_GRAYSCALE)
        plt.hist(img.ravel(), 256, [0,256], color='r')
        cv2.imshow('1.1', img)
        plt.xlim(0, 256)
        plt.show()

    def on_btn1_2_click(self):
        img = cv2.imread(os.path.join('images', 'plant.jpg'), cv2.IMREAD_GRAYSCALE)
        equ = cv2.equalizeHist(img)
        plt.hist(equ.ravel(), 256, [0,256], color='r')
        cv2.imshow('1.2', equ)
        plt.xlim(0, 256)
        plt.show()

    def on_btn2_1_click(self):
        img = cv2.imread(os.path.join('images', 'q2_train.jpg'))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 10, param1=100, \
                                   param2=15, minRadius=15, maxRadius=20)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(img, (i[0],i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0],i[1]), 1, (0, 0, 255), 2)
        cv2.imshow('2.1', img)

    def on_btn2_2_click(self, ret=False):
        img = cv2.imread(os.path.join('images', 'q2_train.jpg'))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 10, param1=100, \
                                   param2=15, minRadius=15, maxRadius=20)
        circles = np.uint16(np.around(circles))
        
        mask = np.zeros(img_gray.shape, dtype=bool)
        for circle in circles[0,:]:
            for x in range(circle[0]-circle[2]+1, circle[0]+circle[2]):
                for y in range(circle[1]-circle[2]+1, circle[1]+circle[2]):
                    if math.hypot(x-circle[0], y-circle[1]) < circle[2]: mask[y,x] = True
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        roi = img_hsv[mask]

        if ret == True:
            M = cv2.calcHist([img_hsv], [0,1], mask.astype(np.uint8), \
                             [180,256], [0,180,0,256])
            return M
        else:
            plt.hist(roi[...,0].ravel(), 256, [0,256], color='r', density=True)
            plt.show()

    def on_btn2_3_click(self):
        roihist = self.on_btn2_2_click(ret=True)
        cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
        
        target = cv2.imread(os.path.join('images', 'q2_test.jpg'))
        target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([target_hsv], [0,1], roihist, [0,180,0,256], 1)
        cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)
        
        _, backproj = cv2.threshold(backproj, 6, 255, cv2.THRESH_BINARY)
        cv2.imshow('2.3', backproj)
        # plt.imshow(roihist, interpolation='nearest')
        # plt.show()

    def on_btn3_1_click(self):
        pass
    
    def on_btn3_2_click(self):
        pass
    
    def on_btn3_3_click(self):
        pass
   
    def on_btn3_4_click(self):
        pass

    def on_btn4_1_click(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
