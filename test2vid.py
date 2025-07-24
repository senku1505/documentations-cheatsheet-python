# VIDEO CAPTURE FROM WEBCAM
# framewidth = 640
# frameheight = 480  
# cap = cv2.VideoCapture("ali.mp4")  
# while True:
#     ret, img = cap.read()  # Read the video frame
#     img = cv2.resize(img, (framewidth, frameheight))  # Resize the frame
#     cv2.imshow("result (press Q to exit)", img)  # Display the image
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Q to exit
#         break     


# BASIC OPERATIONS ON IMAGE
import cv2
import numpy as np
 
# img = cv2.imread("car.png")
# kernel = np.ones((5,5),np.uint8)
 
# theory regarding canny filter
# threshold1 (lower threshold):
# Edge pixels with a gradient higher than this are considered as strong edges.
# threshold2 (upper threshold):
# Edge pixels with a gradient between threshold1 and threshold2 are considered as weak edges and are included only if they are connected to strong edges.

# guassian blur is used to reduce noise and detail in the image, which helps in better edge detection.
# The kernel size is 7 pixels wide and 7 pixels tall.
# The kernel is a square window that slides over the image to apply the blur.
# Larger kernel sizes (like 7x7) produce a stronger blur; smaller sizes (like 3x3) produce a weaker blur.


# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
# imgCanny = cv2.Canny(img,150,200) # img, threshold1, threshold2 



# imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
# imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
 
# cv2.imshow("Gray Image",imgGray)
# cv2.imshow("Blur Image",imgBlur)
# cv2.imshow("Canny Image",imgCanny)
# cv2.imshow("Dialation Image",imgDialation)
# cv2.imshow("Eroded Image",imgEroded)
# cv2.waitKey(0)

# img = cv2.imread("shapes.png")
# print(img.shape) #gives x*y of img 3 is number of color channels (RGB)

# imgResize = cv2.resize(img,(1000,500)) #resize image to 1000x500
# print(imgResize.shape) 

# imgCropped = img[0:200, 200:500]  # Crop the image from (x1,y1) to (x2,y2)

# cv2.imshow("Image", img)
# cv2.imshow("Image Resize", imgResize)
# cv2.imshow("Image Cropped", imgCropped)

# cv2.waitKey(0)  #keypress to exit


# img = np.zeros((512,512,3),np.uint8) # create a black image of 512x512 pixels with 3 color channels (RGB)

# # 0,0 is start cordinate 
# # while img.shape[1] is the width of the image (number of columns, x-axis) img.shape[0] is the height of the image (number of rows, y-axis).
# # mSo, (img.shape[1], img.shape[0]) is the bottom-right corner of the image
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3) # Draw a green line from top-left to bottom-right
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
# cv2.circle(img,(400,50),30,(255,255,0),5)
# cv2.putText(img," OPENCV  ",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),3)

# cv2.imshow("IMAGE", img)
# cv2.waitKey(0)








# img = cv2.imread("car.jpg")
# # stack images horizontally and vertically



# img_resized = cv2.resize(img, (400, 300))
# imgHor = np.hstack((img_resized, img_resized))
# imgVer = np.vstack((img_resized, img_resized))

# cv2.imshow("Horizontal", imgHor)
# cv2.imshow("Vertical", imgVer)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def empty(a): pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# Create trackbars to adjust HSV values
# Trackbars in OpenCV create interactive sliders in a named window, allowing you to adjust values (like HSV color ranges) in real time.
# You use cv2.createTrackbar() to add a trackbar, and cv2.getTrackbarPos() to read its current value in your loop.
# This is useful for tuning parameters (e.g., color thresholds) without restarting your code.

# V IMP - 

# HSV STANDS FOR HUE, SATURATION AND VALUE
# HUE - represents the color type (e.g., red, green, blue)
# SATURATION - represents the intensity of the color (0 is gray, 255 is full color)
# VALUE - represents the brightness of the color (0 is black, 255 is full brightness) 

# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# cv2.createTrackbar("Saturation Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Saturation Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Value Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Value Max", "TrackBars", 255, 255, empty)

# while True:
#     img = cv2.imread("car.jpg")
#     img_resized = cv2.resize(img, (400, 300))
#     imgHSV = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

#     # Get values from trackbars
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Value Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Value Max", "TrackBars")

#     # Create a mask based on the HSV range
#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(imgHSV, lower, upper)
#     result = cv2.bitwise_and(img_resized, img_resized, mask=mask)

#     cv2.imshow("Mask", mask)
#     cv2.imshow("Result", result)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


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