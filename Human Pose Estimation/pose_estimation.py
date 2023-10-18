import cv2 as cv
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--thr', default=0.2, type=float)
parser.add_argument('--width', default=368, type=int)
parser.add_argument('--height', default=368, type=int)

args = parser.parse_args()

# Define body parts indices and pose pairs
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
             "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
             "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
             "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Set input width and height
inWidth = args.width
inHeight = args.height

# Load pre-trained OpenPose model
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Open a video capture object
cap = cv.VideoCapture(args.input if args.input else 0)

# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create main display window
cv.namedWindow('OpenPose using OpenCV', cv.WINDOW_NORMAL)
cv.resizeWindow('OpenPose using OpenCV', (frame_width, frame_height))

# Create pose estimation window
cv.namedWindow('Pose Estimation', cv.WINDOW_NORMAL)
cv.resizeWindow('Pose Estimation', (frame_width, frame_height))

while cv.waitKey(1) < 0:
    # Read a frame from the video capture
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Set the input for the OpenPose network
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Update the main display window
    cv.imshow('OpenPose using OpenCV', frame)

    # Create a frame for pose estimation
    pose_frame = np.zeros_like(frame)
    for i, point in enumerate(points):
        if point:
            cv.circle(pose_frame, point, 5, (0, 0, 255), thickness=-1)
            cv.putText(pose_frame, f"{i}", point, cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    # Add green lines for pose estimation
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(pose_frame, points[idFrom], points[idTo], (0, 255, 0), 2)

    # Update the pose estimation window
    cv.imshow('Pose Estimation', pose_frame)
