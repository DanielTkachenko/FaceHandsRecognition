import cv2 as cv
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, staticMode=False, maxHands=2, minDetectionConf=0.5, minTrackConf=0.5):
        self.staticMode= staticMode
        self.maxHands= maxHands
        self.minDetectionConf= minDetectionConf
        self.minTrackConf= minTrackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.staticMode, self.maxHands, 1, self.minDetectionConf, self.minTrackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipsIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(self.imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xlist = []
        ylist = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 7, (255, 0, 0), cv.FILLED)
            xmin, xmax = min(xlist), max(xlist)
            ymin, ymax = min(ylist), max(ylist)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return self.lmList, bbox

    '''def fingersUp(self):
        fingers = []
        if self.lmList[self.tipsIds[0]][1] > self.lmList[self.tipsIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipsIds[id]][2] < self.lmList[self.tipsIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        print(len(self.lmList))
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), (255, 0, 255), cv.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2, cx, cy]'''



def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img, draw=True)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()