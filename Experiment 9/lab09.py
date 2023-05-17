import numpy as np
import cv2

# load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

img = cv2.imread("D:/Wallpapers/bear.jpg")  # Load image
cv2.imshow("Image", img)  # Display the image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all windows

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
cv2.imshow("Grayscale Image", gray)  # Display the grayscale image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all windows



# detect
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# draw bounding boxes for each face detected
for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w] # roi: region of interest
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread("D:/Wallpapers/bear.jpg")  # Load image
cv2.imshow("Image", img)  # Display the image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all windows


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray scale

# detect
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# draw bounding boxes for each face detected
for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w] # roi: region of interest
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    smiles = smile_cascade.detectMultiScale(roi_gray)
    for (sx, sy, sw, sh) in smiles:
      cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()