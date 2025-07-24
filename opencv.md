
# senku open cv docs

basics n more.

---

## video capture (WEBCAM/FILE)

```python
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("test.mp4")  # use 0/1 to access webcam
# cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()  # framebyframe captures 
    if not ret:
        break
    img = cv2.resize(img, (frameWidth, frameHeight))  # resize to fixed size
    cv2.imshow("result (press Q to exit)", img)  # DISPLAYING IT
    if cv2.waitKey(1) & 0xFF == ord('q'):  # PRESS Q TO EXIT FRAME
        break
cap.release()
cv2.destroyAllWindows()
```

---

## basic operations

```python
import cv2
import numpy as np

img = cv2.imread("car.png")
kernel = np.ones((5, 5), np.uint8)  # A matrix of ones used for dilation/erosion 

# uint8 means from 0 to 255 which is basically RANGE OF 8 BIT COLORS
# 0 - black
# 255 - white

# conv image from color to grayscale (CV uses BGR naming scheme insteadf of standard RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CHANNEL IS BASICALLY NUMBER OF NUMBER OF COLOR COMPONENTS PER PIXEL

# theory regarding canny filter
# threshold1 (lower threshold):
# Edge pixels with a gradient higher than this are considered as strong edges.
# threshold2 (upper threshold):
# Edge pixels with a gradient between threshold1 and threshold2 are considered as weak edges and are included only if they are connected to strong edges.

# guassian blur is used to reduce noise and detail in the image, which helps in better edge detection.
# The kernel size is 7 pixels wide and 7 pixels tall.
# The kernel is a square window that slides over the image to apply the blur.
# Larger kernel sizes (like 7x7) produce a stronger blur; smaller sizes (like 3x3) produce a weaker blur.

# Gaussian Blur: Smooths the image and reduces noise by averaging pixels with a weighted kernel.
# Canny: Detects edges in an image using gradient intensity and thresholding.
# Dilation: Expands white regions (edges) in a binary image, making features thicker.
# Erosion: Shrinks white regions in a binary image, making features thinner.


# Apply Gaussian blur to smooth out the image and reduce noise
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(img, 150, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

cv2.imshow("Gray", imgGray)
cv2.imshow("Blur", imgBlur)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Dilation", imgDilation)
cv2.imshow("Eroded", imgEroded)
cv2.waitKey(0)
```

---

## resize and crop image

```python
img = cv2.imread("shapes.png")
print(img.shape)  # will give (height,width,hannels) 

imgResize = cv2.resize(img, (1000, 500))  # Resize image to 1000x500
imgCropped = img[0:200, 200:500]  # array slicing: for crop [rows, columns]

cv2.imshow("Original", img)
cv2.imshow("Resized", imgResize)
cv2.imshow("Cropped", imgCropped)
cv2.waitKey(0)
```

---

## 🧱 Create Shapes and Text on Image

```python


# img = np.zeros((512,512,3),np.uint8) # create a black image of 512x512 pixels with 3 color channels (RGB)

# # 0,0 is start cordinate 
# # while img.shape[1] is the width of the image (number of columns, x-axis) img.shape[0] is the height of the image (number of rows, y-axis).
# # mSo, (img.shape[1], img.shape[0]) is the bottom-right corner of the image

img = np.zeros((512, 512, 3), np.uint8)  # creates a blank (black image)
cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0,255,0), 3)
cv2.rectangle(img, (0,0), (250,350), (0,0,255), 2)
cv2.circle(img, (400,50), 30, (255,255,0), 5) # 400,50 CENTRE - 30 RADIUS - 5 THICKNESS
cv2.putText(img, " OPENCV  ", (300,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,150,0), 3)

# for above codes,( ) ( ), 3(or 2 or 5) specify thickness of the border/line/circle/text etc
cv2.imshow("random shapes", img)
cv2.waitKey(0)
```

---
## Warp Perspective

```python
img = cv2.imread("cards.jpg")

# Define source points from the original image
pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])

# Define destination points to map to
pts2 = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])

# Compute the transformation matrix and apply it
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (300, 400))

cv2.imshow("Original", img)
cv2.imshow("Warped", imgOutput)
cv2.waitKey(0)
```

---

## joining images (TP)

```python
img = cv2.imread("car.jpg")
# stack images horizontally and vertically
imgHor = np.hstack((img, img))
imgVer = np.vstack((img, img))

cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVer)
cv2.waitKey(0)
```

---

## color detection using trackbar

```python
def empty(a): pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# Create trackbars to adjust HSV values
# Trackbars in OpenCV create interactive sliders in a named window, allowing you to adjust values (like HSV color ranges) in real time.
# You use cv2.createTrackbar() to add a trackbar, and cv2.getTrackbarPos() to read its current value in your loop.
# This is useful for tuning parameters (e.g., color thresholds) without restarting your code.

cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Saturation Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Saturation Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Value Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Value Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread("car.jpg")
    img_resized = cv2.resize(img, (400, 300))
    imgHSV = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Get values from trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Value Max", "TrackBars")

    # Create a mask based on the HSV range
    lower = np.array([h_min, s_min, v_min]) # Lower bound of HSV values selected using trackbars
    upper = np.array([h_max, s_max, v_max]) # Upper bound of HSV values selected using trackbars
    mask = cv2.inRange(imgHSV, lower, upper)  # Creates a binary mask where pixels in range = 255 (white), others = 0 (black)
    result = cv2.bitwise_and(img, img, mask=mask) # Keeps only the parts of the image where mask == 255
 
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## contour and shape detc
```python
def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3: objectType = "Triangle"
            elif objCor == 4: objectType = "Rectangle"
            elif objCor > 4: objectType = "Circle"
            else: objectType = "None"

            cv2.drawContours(imgContour, [approx], -1, (0, 255, 0), 3)
            cv2.putText(imgContour, objectType, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

img = cv2.imread("shapes.png")
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)

cv2.imshow("Contours", imgContour)
cv2.waitKey(0)
```

---

## face detec (HAAR CASCADE SOLO)

```python
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("face.jpg")
# Haar cascade works only on single-channel (grayscale) images because it uses intensity patterns
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect faces using Haar feature-based cascade
# - 1.1: scaleFactor – how much the image size is reduced at each image scale (1.1 = 10% smaller)
# - 4: minNeighbors – how many neighbors each candidate rectangle should have to be considered a valid detection
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)




# Loop over all detected faces and draw rectangles
# Each face is represented as a rectangle: (x, y) is top-left corner, w and h are width and height
for (x, y, w, h) in faces:
    # Draw a blue rectangle around the detected face
    # Parameters: image, top-left point, bottom-right point, color (BGR), thickness
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 



cv2.imshow("Face Detection", img)
cv2.waitKey(0)
```

---

## requirments

```bash
pip install opencv-python numpy
```

---

## links
[OpenCV Docs](https://docs.opencv.org/).
[G2G Cheat Sheet](https://www.geeksforgeeks.org/python/python-opencv-cheat-sheet/).
