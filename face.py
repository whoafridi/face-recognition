import cv2
import numpy as np
import face_recognition as fr

imgElon = fr.load_image_file("elon.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = fr.load_image_file("elon1.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = fr.face_locations(imgElon)[0]
encodeElon = fr.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = fr.face_locations(imgTest)[0]
encodeElonTest = fr.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results = fr.compare_faces([encodeElon],encodeElonTest)
facedis = fr.face_distance([encodeElon],encodeElonTest)
print(results,facedis)
cv2.putText(imgTest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

cv2.imshow('e',imgElon)
cv2.imshow('r',imgTest)
cv2.waitKey(0)
