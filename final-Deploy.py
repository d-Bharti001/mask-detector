##################################################
# Imports

import argparse
import numpy as np
import cv2 as cv
from keras.models import load_model
from keras.preprocessing import image

##################################################
# Detection models

# load mask detection model
maskNet = load_model("mask_cnn.h5")

# build pre-trained face detection model
prototxt_path = "face_detector/deploy.prototxt.txt"  # architecture
weights_path = "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel" # weights
faceNet = cv.dnn.readNet(prototxt_path, weights_path)

##################################################
# Function for generating output frame
# by detecting faces and masks

def detect(frame):
    
    # construct a blob from frame
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # initialize list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections
        if confidence > 0.4:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(FRAME_WIDTH - 1, endX), min(FRAME_HEIGHT - 1, endY))

            # extract the face ROI and do some preprocessing
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (128,128), cv.INTER_AREA)
            face = image.img_to_array(face)
            face /= 255

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # predictions can be done only when atleast one face if detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    nums = len(locs)
    count_violations = 0

    # traverse through all recognized bounding boxes
    for i in range(nums):
        # check if mask detected for corresponding face
        # and set color and text for bounding boxes accordingly
        pred = preds[i][0]
        if pred < 0.7:  # no mask detected
            color = (0,0,255) # RED in B,G,R order
            text = "no mask: {:.2f}%".format((1-pred)*100)
            count_violations += 1
        else:  # mask detected
            color = (0,255,0) # GREEN in B,G,R order
            text = "mask: {:.2f}%".format(pred*100)

        # get the box coordinates
        (startX, startY, endX, endY) = locs[i]
        # draw box around face
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # write text above box
        cv.putText(frame, text, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)

    # print total number of `no mask` faces at the bottom
    cv.putText(frame, f"Total violations: {count_violations}", (10, FRAME_HEIGHT - 10),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255,102,0), 2, cv.LINE_AA)


##################################################
# DRIVER CODE

# Get a specific video file
# otherwise camera 0 (webcam) stream will be used

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str,
                    help="path to video file for detection (relative path)")
args = parser.parse_args()

# set stream source to webcam stream
# if video file input is not passed
# else set it to that video file
src = 0 if not args.video else args.video


# start capturing video
cap = cv.VideoCapture(src)

# capture initial frame
ret, frame = cap.read()
if not ret:
        print("Stream not available! Exiting...")
        cap.release()
        exit(0)

# store the height and width of the input stream
FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]

# capture frame by frame
while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Stream not available! Exiting...")
        break

    # pass the frame for detection
    detect(frame)

    # show the output on screen
    cv.imshow('Result', frame)
    
    # wait for 30 milliseconds
    # if `Esc` key is pressed, exit
    if cv.waitKey(30) & 0xFF == 27:
        break

# release the capture and close the output window
cap.release()
cv.destroyAllWindows()
