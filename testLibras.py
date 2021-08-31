import time
import traceback
import mediapipe as mp
import numpy as np
from numpy import array
from tensorflow import keras
from threading import Thread


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

class TestLibras(Thread):

    def __init__(self, partTime: float):
        super().__init__()
        self.partTime = partTime
        self.isRunning = False
        self.frames = list()
        print("Model")
        self.model = keras.models.load_model('model')


    def addFrame(self, results) -> bool:
        if (results.pose_landmarks == None):
            return False

        line = self.createLinePose(results, list())

        if (results.left_hand_landmarks != None):
            line = self.createLineHand(results.left_hand_landmarks.landmark, line)
        else:
            line = self.createLineEmptyHand(line)

        if (results.right_hand_landmarks != None):
            line = self.createLineHand(results.right_hand_landmarks.landmark, line)
        else:
            line = self.createLineEmptyHand(line)

        self.frames.append(line)

        return True


    def run(self):
        self.isRunning = True

        while (self.isRunning):
            try:
                if (len(self.frames) > 4):
                    self.frames = self.frames[(len(self.frames)-4):]
                    timeProcess = array([self.frames[0] + self.frames[1] + self.frames[2] + self.frames[3]])

                    pred1 = self.model.predict(timeProcess)
                    #pred2 = self.model.predict_classes(timeProcess)

                    print("----------------------------------------")
                    print(str(np.argmax(pred1[0], axis=0)) + " - " + str(max(pred1[0])))
                    #print(str(len(pred1[0])))
                    #print(str(pred1[0]))
                    #print(str(pred2))

            except:
                print("Ops!!!!\n" + traceback.format_exc(1))
                raise Exception('I know Python!')
                



            time.sleep(self.partTime)



    def createLinePose(self, results, line: list) -> list:

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
        
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y)

        return line

    def createLineEmptyHand(self, line: list) -> list:
        for x in range(40):
            line.append(0)

        return line


    def createLineHand(self, landmark, line: list) -> list:
        #line.append(landmark[mp_holistic.HandLandmark.WRIST].x)
        #line.append(landmark[mp_holistic.HandLandmark.WRIST].y)

        line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].y)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].y)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)

        line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].y)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].y)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].y)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].y)

        return line
        
