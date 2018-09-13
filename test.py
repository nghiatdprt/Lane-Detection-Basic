import cv2
import numpy as np
# Input Image
def detect_road(image):
    cx = -1
    cy = -1
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Define range of white color in HSV
    lower_white = np.array([0, 0, 212])
    upper_white = np.array([131, 255, 255])
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Remove noise
    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6,6),np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
    # Find the different contours
    im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area (keep only the biggest one)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    if len(contours) > 0:
        M = cv2.moments(contours[0])
        # Centroid
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return (cx, cy)
def crop_cam(frame, ratio):
    h, w, _ = frame.shape
    y = min(ratio*h, h-50)
    center_x = w/2
    x = center_x - 320/2
    y = int(y)
    x = int(x)
    img = frame[y:y+50, x:(x+320)]
    return (img, x, y)
vc = cv2.VideoCapture("video1.mp4")
while vc.isOpened():
    ret, frame = vc.read()
    if ret:
        img, x, y = crop_cam(frame, 0.3)
        coord = detect_road(img)
        cv2.circle(frame ,(coord[0]+x, coord[1]+y), 3, (0,255,0), 2)
        img, x, y = crop_cam(frame, 0.4)
        coord = detect_road(img)
        cv2.circle(frame ,(coord[0]+x, coord[1]+y), 3, (0,255,0), 2)
        img, x, y = crop_cam(frame, 0.5)
        coord = detect_road(img)
        cv2.circle(frame ,(coord[0]+x, coord[1]+y), 3, (0,255,0), 2)
        cv2.imshow("preview captures", frame)
    if cv2.waitKey(30) == ord('q'):
        break