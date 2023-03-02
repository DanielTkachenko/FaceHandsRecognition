import cv2 as cv
import HandTrackingModule as htm
import FaceMeshModule as fmm
import time
import pandas


def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    cTime = 0
    hDetector = htm.HandDetector()
    fDetector = fmm.FaceMeshDetector()
    while True:
        success, img = cap.read()
        img = hDetector.findHands(img)
        img, faces = fDetector.findFaceMesh(img)
        lmList, bbox = hDetector.findPosition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) == 27:
            break



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
