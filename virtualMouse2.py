import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
wScr, hScr = pyautogui.size()
frameR = 100
smoothening = 2
plocX,plocY = 0,0
clocX,clocY = 0,0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hCam, wCam, _ = img.shape
    hands, img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        # print(hand1["type"])
        x1,y1 = lmList1[8][0:2]
        x2,y2 = lmList1[12][0:2]
        fingers = detector.fingersUp(hand1)
        # print(x1, y1, x2, y2)
        cv2.rectangle(img, (frameR, frameR),(wCam - frameR, hCam-frameR),(0,255,0),2)
        if fingers[1] == 1 and fingers[2] == 0 and hand1["type"] == "Left":
            # print(wScr,hScr)
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR,hCam - frameR), (0,hScr))
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(clocX,clocY)
            plocX,plocY = clocX,clocY
        # print(fingers)
        if fingers[1] == 1 and fingers[2] == 1 and hand1["type"] == "Left":
            length, info, img= detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img)
            # print(length)
            if length <= 40:
                cv2.circle(img,(info[4],info[5]),10,(0,255,0),cv2.FILLED)
                pyautogui.click()
                pyautogui.sleep(1)
    # if hands:
    #     hand1 = hands[0]
    #     lmList1 = hand1["lmList"]
    #     bbox1 = hand1["bbox"]
    #     handType1 = hand1["type"]
    #     fingers1 = detector.fingersUp(hand1)
    #     print(fingers1)
        # print(len(lmList1))
    cv2.imshow("Image", img)
    cv2.waitKey(1)