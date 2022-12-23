import cv2
import cvzone
import face_recognition
import numpy as np

imgAvik = face_recognition.load_image_file("avikPic.jpg")
imgAvik = cv2.cvtColor(imgAvik, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("avikPic2.jpeg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgAvik)[0]
encodeAvik = face_recognition.face_encodings(imgAvik)[0]
cv2.rectangle(imgAvik,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeAvikTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeAvik],encodeAvikTest)
facedis = face_recognition.face_distance([encodeAvik],encodeAvikTest)
print(results,facedis)
cv2.putText(imgTest,f'{results} {np.round(facedis, 2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

cv2.imshow('Avik',imgAvik)
cv2.imshow('Avik Test',imgTest)
cv2.waitKey(0)


