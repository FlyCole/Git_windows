#! /usr/bin/env python
# -*- encoding: UTF-8 -*-
"""
    Author: Yifei Ren
    Name: WaveDetection
    Version: 1.0
    Date: 09/03/2019
    Description: Judge if the person is waving and give the waving hand.
    Note: Use for one person.
          If the light is not usual, the parameters of HSV should be replaced.
"""
import cv2
import dlib
import math
import time
import numpy as np


class wave_detection:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        print("Detector Succeed")
        # Initial parameters
        cap = cv2.VideoCapture(0)
        self.is_detect = False
        self.is_wave = False  # Judge if the waving is detected
        self.rects = None  # Face detection results
        self.rect_width = None
        self.frame_copy = None
        self.frame_left = None  # Left waving detection
        self.frame_right = None  # Right waving detection
        self.hsv_range = None
        self.sec_last = None  # Record the start time
        self.sec_cur = None  # Record the current time
        self.first_record = True  # Judge if first record
        self.max_threshold = 0.4
        self.last_flag = None  # Record the last flag to judge left or right
        success, frame = cap.read()
        while success:
            self.is_detect = True
            if self.is_detect:
                self.frame_copy = frame.copy()
                self.rects = self.detector(self.frame_copy, 1)
                if len(self.rects) != 0:
                    for rect in self.rects:
                        self.rect_width = rect.right() - rect.left()
                        cv2.rectangle(self.frame_copy, (rect.left(), rect.top()),
                                      (rect.right(), rect.bottom()), (0, 0, 255), 2, 8)
                        # Calculate possible hsv range
                        face_copy = self.frame_copy[rect.top(): rect.bottom(), rect.left(): rect.right()]
                        cv2.imshow("face_copy", face_copy)
                        self.hsv_range = self.extract_face_hsv(face_copy)
                        # Extract left waving information
                        right_copy = self.assign_broad_right(rect)
                        self.frame_left = self.frame_copy[int(rect.top() / 1.2): int(rect.bottom() * 1.05),
                                          right_copy[0]: right_copy[1]]
                        # Extract right waving information
                        left_copy = self.assign_broad_left(rect)
                        self.frame_right = self.frame_copy[int(rect.top() / 1.2): int(rect.bottom() * 1.05),
                                           left_copy[0]: left_copy[1]]
                        # Draw the extracted area
                        cv2.rectangle(self.frame_copy, (right_copy[0], int(rect.top() / 1.2)),
                                      (right_copy[1], int(rect.bottom() * 1.05)), (0, 255, 0), 2, 8)
                        cv2.rectangle(self.frame_copy, (left_copy[0], int(rect.top() / 1.2)),
                                      (left_copy[1], int(rect.bottom() * 1.05)), (255, 0, 0), 2, 8)
                        # Judge if waving
                        if not self.is_wave:
                            self.judge(self.frame_left, flag=0)
                        if not self.is_wave:
                            self.judge(self.frame_right, flag=1)
                        if not self.is_wave:
                            if self.first_record:
                                time_last = time.localtime(time.time())
                                self.sec_last = time_last[3] * 3600 + time_last[4] * 60 + time_last[5]
                                self.first_record = False
                            time_cur = time.localtime(time.time())
                            self.sec_cur = time_cur[3] * 3600 + time_cur[4] * 60 + time_cur[5]
                            if self.sec_cur - self.sec_last <= 1:
                                if self.last_flag == 1:
                                    print("========== Waving Left========== " + str(time_last[3]) + ":"
                                          + str(time_last[4]) + ":" + str(time_last[5]))
                                elif self.last_flag == 0:
                                    print("========== Waving Right========== " + str(time_last[3]) + ":"
                                          + str(time_last[4]) + ":" + str(time_last[5]))
                            else:
                                print("========== Not Waving ========== " + str(time_last[3]) + ":"
                                      + str(time_last[4]) + ":" + str(time_last[5]))
                                self.last_flag = 2
                        self.is_wave = False
                if self.last_flag == 1:
                    cv2.putText(self.frame_copy, 'Waving Left Hand', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif self.last_flag == 0:
                    cv2.putText(self.frame_copy, 'Waving Right Hand', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif self.last_flag == 2:
                    cv2.putText(self.frame_copy, 'Not Waving', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
                cv2.imshow("capture", self.frame_copy)
            self.is_detect = False
            if cv2.waitKey(1) >= 0:
                break
            success, frame = cap.read()
        cv2.destroyAllWindows()
        cap.release()

    def assign_broad_right(self, rect):
        if rect.left() - 2 * self.rect_width < 0:
            copy_left = 0
        else:
            copy_left = int(rect.left() - 2 * self.rect_width)
        if rect.right() - 2 * self.rect_width < 0:
            copy_right = rect.left()
        else:
            copy_right = int(rect.right() - 2 * self.rect_width)
        return [copy_left, copy_right]

    def assign_broad_left(self, rect):
        if rect.right() + 2 * self.rect_width > self.frame_copy.shape[1]:
            copy_right = self.frame_copy.shape[1]
        else:
            copy_right = int(rect.right() + 2 * self.rect_width)
        if rect.left() + 2 * self.rect_width > self.frame_copy.shape[1]:
            copy_left = rect.right()
        else:
            copy_left = int(rect.left() + 2 * self.rect_width)
        return [copy_left, copy_right]

    def extract_face_hsv(self, img):
        print("-------------------------Extracting face HSV-------------------------")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lower = np.array([int(np.mean(h) - 0.4 * math.sqrt(np.var(h))),
                          int(np.mean(s) - 0.5 * math.sqrt(np.var(s))),
                          int(np.mean(v) - 0.1 * math.sqrt(np.var(v)))])
        upper = np.array([int(np.mean(h) + math.sqrt(np.var(h))), 255, 255])
        return lower, upper

    def judge(self, img, flag):
        max = np.shape(img)[0] * np.shape(img)[1]
        cv2.namedWindow("judge", cv2.WINDOW_NORMAL)
        cv2.imshow("judge", img)
        cv2.waitKey(1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # print(self.hsv_range)
        mask = cv2.inRange(hsv, self.hsv_range[0], self.hsv_range[1])
        binary = cv2.dilate(mask, None, iterations=3)
        pic, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binary, cnts, 0, (0, 255, 255), 8)
        if flag:
            cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
            cv2.imshow("Left", binary)
        elif flag == 0:
            cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
            cv2.imshow("Right", binary)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        print("The ratio is: " + str(sum / max))
        if sum / max > self.max_threshold:
            time_last = time.localtime(time.time())
            self.sec_last = time_last[3] * 3600 + time_last[4] * 60 + time_last[5]
            if flag:
                print("========== Waving Left========== " + str(time_last[3]) + ":"
                      + str(time_last[4]) + ":" + str(time_last[5]))
                self.last_flag = 1
            elif flag == 0:
                print("========== Waving Right========== " + str(time_last[3]) + ":"
                      + str(time_last[4]) + ":" + str(time_last[5]))
                self.last_flag = 0
            self.is_wave = True

def main():
    wave_detection()


if __name__ == '__main__':
    main()


