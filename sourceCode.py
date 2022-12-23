import cv2
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np
import cvzone
import os
import face_recognition

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
wScr, hScr = pyautogui.size()
frameR = 100
smoothening = 2
plocX,plocY = 0,0
clocX,clocY = 0,0

path = 'authorizedPic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(cl.split('.')[0])
    print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = findEncodings(images)
print("Encoding Complete")

i = 10
checkFace = True
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hCam, wCam, _ = img.shape
    hands, img = detector.findHands(img)


    imgS = cv2.resize(img,(0,0),None,0.20,0.20)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    if i == 10:
        i = 0
        encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
        # print(encodesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame,faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
            faceDistance = face_recognition.face_distance(encodeListKnow,encodeFace)
            # print(matches)
        
            if(matches[0] == True):
                checkFace = True
            else:
                checkFace = False

    i = i + 1
    
    for faceLoc in faceCurFrame:
        if checkFace == True:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*5,x2*5,y2*5,x1*5
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),2,cv2.FILLED)
            cv2.putText(img, "Avik", (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        else:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,0,255),2,cv2.FILLED)
            cv2.putText(img, "Unknown", (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)


    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        # print(hand1["type"])
        x1,y1 = lmList1[8][0:2]
        x2,y2 = lmList1[12][0:2]
        fingers = detector.fingersUp(hand1)
        # print(x1, y1, x2, y2)
        cv2.rectangle(img, (frameR, frameR),(wCam - frameR, hCam-frameR),(0,255,0),2)
        if fingers[1] == 1 and fingers[2] == 0 and hand1["type"] == "Left" and checkFace:
            # print(wScr,hScr)
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR,hCam - frameR), (0,hScr))
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(clocX,clocY)
            plocX,plocY = clocX,clocY
        # print(fingers)
        if fingers[1] == 1 and fingers[2] == 1 and hand1["type"] == "Left" and checkFace:
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