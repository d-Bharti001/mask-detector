""" Real-time Mask Detector

This program takes as input a live camera stream or a video source
and detects faces and whether there is a face-mask on them or not
and displays the result in a GUI window.

It also counts how many faces are detected without a face-mask
and shows the count as "Number of violations".

Additional python libraries required:
    numpy           (v1.19.2 used)
    tensorflow      (v2.1.0 used)
    keras           (v2.3.1 used)
    opencv-python   (v4.5.1.48 used)

Python version used: 3.6.12

Usage:
    python mask-detector.py [--source <path/to/video/file>]
    
    Press ESC key at any time to exit the program.

    Set FACE_CONFIDENCE to some value between 0 and 1 to vary
    face detection accuracy. All faces detected with
    probability > FACE_CONFIDENCE are considered for further processing.
    Rest detected faces are rejected.

    Set MASK_CONFIDENCE to some value between 0 and 1 to vary
    mask detection accuracy. All faces with classified with
    probability > MASK_CONFIDENCE are considered to have face-mask.
    Rest are considered to be without mask.
"""


import argparse
import numpy as np
import cv2 as cv
from keras.models import load_model
from keras.preprocessing import image


def detect(frame):
    """Gets the camera/video frame and modifies it with
    detected faces and masks

    First, detects faces in the frame and stores only those faces
    and their corresponding bounding boxes whose confidence is greater
    than FACE_CONFIDENCE.

    Then, for all the faces, probabilities of having a mask is calculated.

    Then, bounding boxes with a proper label and probability are drawn
    around every face using prediction values and box coordinates.

    The modified frame contains bounding boxes over faces
    and a label "mask" or "no mask" with corresponding
    probabilities.
    At bottom left of the frame, there is total number
    of violations, i.e. number of faces with no mask.

    Parameters
    ----------
    frame : numpy.ndarray
        frame of camera stream / video file
    """

    # construct a blob from frame
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initialize list of faces, their corresponding locations,
    # and the list of predictions from our face mask network.
    faces = []
    locs = []
    preds = []


    # Loop over the face detections
    for i in range(detections.shape[2]):

        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > FACE_CONFIDENCE:

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(FRAME_WIDTH - 1, endX), min(FRAME_HEIGHT - 1, endY))

            # Extract the face ROI and do some preprocessing
            face = frame[startY:endY, startX:endX]
            face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (128,128), cv.INTER_AREA)
            face = image.img_to_array(face)
            face /= 255

            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))


    # Predictions can be done only when atleast one face is detected
    if len(faces) > 0:

        # For faster inference, prediction is performed on *all* faces
        # in batch mode rather than one-by-one predictions.
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)


    # Now draw rectangular bounding boxes and
    # print some relevant info in the frame

    nums = len(locs)
    count_violations = 0

    for i in range(nums):

        # Check if mask detected for corresponding face
        # and set color and text for bounding boxes accordingly

        # Get current prediction for face-mask
        pred = preds[i][0]

        if pred <= MASK_CONFIDENCE:  # No mask detected

            color = (32,0,176) # Material Red

            text = "no mask: {:.2f}%".format((1-pred)*100)

            # Increase the count
            count_violations += 1

        else:  # Mask detected

            color = (255,54,3) # Material Green

            text = "mask: {:.2f}%".format(pred*100)

        # Get the box coordinates
        (startX, startY, endX, endY) = locs[i]

        # Draw box around face
        cv.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Write text above box
        cv.putText(frame, text, (startX, startY - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)


    # At last, print total number of 'no mask' faces at the bottom
    cv.putText(frame,
               f"Total violations: {count_violations}",
               (10, FRAME_HEIGHT - 10),
               cv.FONT_HERSHEY_SIMPLEX,
               1, (30,81,244), 2,
               cv.LINE_AA)


############### DRIVER CODE ###############


# Load mask detection model
maskNet = load_model("mask_cnn.h5")


# Build face detection model from pre-trained
prototxt_path = "face_detector/deploy.prototxt.txt"  # architecture
weights_path = "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel"  # weights
faceNet = cv.dnn.readNet(prototxt_path, weights_path)


# Set threshold confidence for face-detection and mask-detection
FACE_CONFIDENCE = 0.3
MASK_CONFIDENCE = 0.8


# Get a specific video file
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str,
                    help="path to custom source (e.g. video file) for detection (relative path)")
args = parser.parse_args()


# If source file is not provided,
# then set stream source to webcam stream.
# Else set it to that video file.
src = 0 if not args.source else args.source


# Start capturing video
cap = cv.VideoCapture(src)


# Capture initial frame
ret, frame = cap.read()
if not ret:
        print("Stream not available! Exiting...")
        cap.release()
        exit(0)


# Store the height and width of the input stream
FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]


# Capture frame by frame
while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Stream ended! Exiting...")
        break

    # Pass the frame for detection
    detect(frame)

    # Show the output on screen
    cv.imshow('Result', frame)
    
    # Wait for 15 milliseconds before capturing next frame
    # Break if `Esc` key is pressed
    if cv.waitKey(15) & 0xFF == 27:
        print("Keyboard interrupt! Exiting...")
        break

# Release the capture and close the output window
cap.release()
cv.destroyAllWindows()
