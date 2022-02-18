import cv2

# image
img_file = 'Car.png'
video = cv2.VideoCapture('tesla.mp4')

# pre-trained car classifier
classifier_file = 'car_detector.xml'

# opencv image
img = cv2.imread(img_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# display the image with car spotted
cv2.imshow('Clever PRogrammer Cars Detector', img)

# Dont austoclose (Wait here in the code and listen for a key press)
cv2.waitKey()

print ("Code Completed!")