import os,cv2
import face_recognition
import numpy as np
from PIL import Image
import pickle


# class trainer:
#     def trainuser(self):
# recognizer =  cv2.face.LBPHFaceRecognizer_create()
from imutils import paths
import face_recognition
#import argparse
import pickle
import cv2
import os


# our images are located in the dataset folder
print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("Data"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
NameId=[]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = str(os.path.split(imagePath)[-1].split('.')[1])
	id = str(os.path.split(imagePath)[-1].split('.')[0])

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model="hog")

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodingsbook.save(str(month) + '.xlsx')
		knownEncodings.append(encoding)
		knownNames.append(name)
		NameId.append(id)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames,"id":NameId}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

# users, faces = img(path)
# recognizer.train(faces, np.array(users))  # training
# recognizer.save('recognizer/TraningData.yml')  # saving histogram data
# cv2.destroyAllWindows()





