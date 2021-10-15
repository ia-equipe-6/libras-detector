import sys
import cv2
import math
import time
import signal 
from testLibras import TestLibras
import mediapipe as mp

TIME = 1
PARTS = 5
RESIZE = False

DESIRED_HEIGHT = 200
DESIRED_WIDTH = 200

partTime = TIME / PARTS
lastFrameTime = time.time()

print("Start")

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

print("Capture")
capture = cv2.VideoCapture(0)

testLibras = TestLibras(partTime, PARTS)

def signal_handler(signal, frame):
    testLibras.isRunning = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("Holistic")
with mp_holistic.Holistic(
            static_image_mode=True, 
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

    testLibras.start()

    print("Loop")
    while (cv2.waitKey(1) < 0):
        start = time.time()
        conected, image = capture.read()

        if (RESIZE):
            h, w = image.shape[:2]

            if h < w:
                image = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
            else:
                image = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        lastTime = start - lastFrameTime
        if (lastTime > partTime
            and testLibras.addFrame(results)):
            lastFrameTime = time.time()

        #annotated_image = cv2.cvtColor(image, cv2.COLOR_)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image=image, landmark_list=results.pose_landmarks, connections=mp_holistic.POSE_CONNECTIONS)

        end = time.time()
        totalTime = end - start
        fps = math.trunc(1 / totalTime)
        cv2.putText(image, "FPS: " + str(fps), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Libras', image)
        
        #print(fps)

