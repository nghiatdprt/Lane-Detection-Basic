import project_config
import toulouse_dataset
import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np
import cv2
def preprocess_data(img_list, width= 320, height=50):
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    res = img_list
    # create a big 1D-array
    
    # for img in img_list:
    #     # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # mask_white = cv2.inRange(img, 140, 255)
    #     res.append(cv2.resize(img, (int(width/2), int(height/2))))
        
    res = np.array(res)
    # data_shape = res.shape
    # res = np.reshape(res, [data_shape[0], data_shape[1], data_shape[2], -1])
    # print(res.shape)
    # Normalize
    res = res / 255. # values in [0, 1]
    res -= 0.5 # values in [-0.5, 0.5]
    res *= 2 # values in [-1, 1]
    return res
# x_train, y_train, x_test, y_test = toulouse_dataset.load_toulouse_dataset()
predict_fn = predictor.from_saved_model("model\\1536260057")
def detect_road(features):
    data = np.array(features)
    data_shape = data.shape
    # data = np.reshape(data, [data_shape[0], data_shape[1], data_shape[2], -1])
    # data = normalize_data(data)
    # print(data.shape)
    # print(data[0])
    return predict_fn({"input": data})["coordinate"]
vc = cv2.VideoCapture("video2.mp4")
while vc.isOpened():
    ret, frame = vc.read()
    if ret:
        # print(frame.shape)
        # print(frame)
        # if ret:
        h, w, _ = frame.shape
        y = min(0.4*h, h-50)
        center_x = w/2
        x = center_x - 320/2
        y = int(y)
        x = int(x)
        img = frame[y:y+50, x:(x+320)]
        img = cv2.resize(img, (320, 50))
        img = preprocess_data([img])[0]
        coord = detect_road([img])[0]
        # print(coord)
        # fr = img[0]
        cv2.circle(frame ,(coord[0]+x, coord[1]+y), 3, (0,255,0), 2)
        # cv2.circle(img ,(int(coord[0]/2), int(coord[1]/2)), 3, (0,255,0), 2)
        cv2.imshow("preview captures", frame)
        # cv2.imshow("captures", img)
        # print(img.shape)

    if cv2.waitKey(30) == ord('q'):
        break
    