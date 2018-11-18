# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import math
import os
from hw2_ui import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib import pyplot as plt


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()
        self.rvecs = None
        self.tvecs = None
        self.distCoeffs = None
        self.cameraMatrix = None
        self.objpts, self.imgpts = None, None
        np.set_printoptions(suppress=True, precision=6)

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

    def on_btn3_1_click(self, insideCall=False):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1,2)

        objpts = [] # 3d point in real world space
        imgpts = [] # 2d points in image plane.
    
        for i in range(1, 16):
            img = cv2.imread(os.path.join('images', 'CameraCalibration', str(i)+'.bmp'))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(img, (11,8))
            if ret == True:
                objpts.append(objp)
                corners2 = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
                imgpts.append(corners2)
                if not insideCall:
                    result = cv2.drawChessboardCorners(img, (11,8), corners2, ret)
                    result = cv2.pyrDown(result)
                    cv2.imshow('3.1' + ' ' + str(i) + '.bmp', result)
        if insideCall == True:
            return objpts, imgpts

    def on_btn3_2_click(self, insideCall=False):
        objpts, imgpts = self.on_btn3_1_click(insideCall=True)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
            cv2.calibrateCamera(objpts, imgpts, (2048,2048), None, None)

        self.objpts, self.imgpts = np.array(objpts), np.array(imgpts)
        self.cameraMatrix = np.array(cameraMatrix)
        self.distCoeffs = np.array(distCoeffs)
        self.rvecs, self.tvecs = np.array(rvecs), np.array(tvecs)
        if not insideCall:
            print(np.array(cameraMatrix), '\n')
    
    def on_btn3_3_click(self, insideCall=False):
        if self.rvecs is None:
            self.on_btn3_2_click(insideCall=True)

        index = int(self.comboBox.currentText())-1
        extrinsic, _ = cv2.Rodrigues(self.rvecs[index])
        extrinsic = np.append(extrinsic, self.tvecs[index], axis=1)
        if insideCall: return extrinsic
        print(extrinsic, '\n')
   
    def on_btn3_4_click(self):
        if self.distCoeffs is None:
            self.on_btn3_2_click(insideCall=True)
        print(self.distCoeffs)

    def on_btn4_1_click(self):
        if self.cameraMatrix is None:
            self.on_btn3_2_click(insideCall=True)

        img = cv2.imread(os.path.join('images', 'CameraCalibration', '2.bmp'))
        axis = np.array([[0,0,0], [0,-2,0], [-2,-2,0], [-2,0,0],
                         [0,0,-2],[0,-2,-2],[-2,-2,-2],[-2,0,-2]], dtype=np.float32)
        axis[...,0] += 10; axis[...,1] += 7
        imgpts, jacb = cv2.projectPoints(
            axis, self.rvecs[1], self.tvecs[1], self.cameraMatrix, self.distCoeffs
        )
        imgpts = np.int32(imgpts).reshape(-1,2)
        img = cv2.drawContours(img, [imgpts[:4]], -1, (0,0,255), 5)
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 5)
        for i, j in zip(range(4), range(4,8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0,0,255), 5)
        cv2.imshow('4.1', cv2.pyrDown(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
