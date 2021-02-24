import cv2
import numpy
from picamera import PiCamera as pi
# class createuser:
#     def createuser(self,name,id):


load = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # loading the cascade for face detectionl
cap = cv2.VideoCapture(0)  # Video capturing object (opening webcame - default value 0)

name = input("Enter username:")
id = input("Enter ID:")


f = open("datatext.csv", "a")  # saving user info in a file
f.write(str(id) + " " + name + "\n")
val = 0
# for continuously detecting the face
while (1):
    status, img = cap.read()  # for reading image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting BGR to Grayscale
    faces = load.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        val = val + 1
        cv2.imwrite("Data/" + str(id)+"."+str(name) + "." + str(val) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.waitKey(10)

    cv2.imshow('FaceDetect', img)
    cv2.waitKey(1)
    if (val >= 10):
        break

f.close()
cap.release()
cv2.destroyAllWindows()