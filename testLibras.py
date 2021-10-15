import time
import traceback
import mediapipe as mp
import numpy as np
from numpy import array
from tensorflow import keras
from threading import Thread
from sklearn.preprocessing import LabelEncoder


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
WITH_Z = False
POSE_MODE = True
TRACK_MODE = False
TRACK_SIMPLE = False
MODEL_PATH = "model"
WORDS_ENCODER = "words_encoder"
#MODEL_PATH = "model_track" if TRACK_MODE else "model"
#WORDS_ENCODER = "words_encoder_track" if TRACK_MODE else "words_encoder"

#ACTIVATION = "relu"
#ACTIVATION = "tanh"

#MODEL_PATH += "_" + ACTIVATION
#WORDS_ENCODER += "_" + ACTIVATION

class TestLibras(Thread):

    def __init__(self, partTime: float, parts: int):
        super().__init__()
        self.partTime = partTime
        self.parts = parts
        self.isRunning = False
        self.frames = list()
        print("Model")
        self.model = keras.models.load_model(MODEL_PATH)

        print("Word Encoder")
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(WORDS_ENCODER + ".npy", allow_pickle=True)


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

    def calculateTrackDiff(self, framePos, framePos2):
        diff = framePos - framePos2
        
        if (TRACK_SIMPLE):
            if (diff > 0.001):
                return 1.0
            elif (diff < -0.001):
                return -1.0
            else:
                return 0.0
        else:
            return diff


    def lineTrackMode(self, frame: list, frame1: list, frame2: list, frame3: list) -> list:
        size = len(frame)

        trackList1 = list()
        trackList2 = list()
        trackList3 = list()

        for pos in range(0, size):
            if (frame[pos] == 0.0 or frame1[pos] == 0.0):
                trackList1.append(0.0)
            else:
                trackList1.append(self.calculateTrackDiff(frame[pos], frame1[pos]))

            if (frame1[pos] == 0.0 or frame2[pos] == 0.0):
                trackList2.append(0.0)
            else:
                trackList2.append(self.calculateTrackDiff(frame1[pos], frame2[pos]))

            if (frame2[pos] == 0.0 or frame3[pos] == 0.0):
                trackList3.append(0.0)
            else:
                trackList2.append(self.calculateTrackDiff(frame2[pos], frame3[pos]))

        line = frame + trackList1 + trackList2 + trackList3

        return line

    def run(self):
        self.isRunning = True

        while (self.isRunning):
            try:
                if (len(self.frames) > self.parts):
                    self.frames = self.frames[(len(self.frames) - self.parts):]
                    timeProcess = list()

                    if (TRACK_MODE):
                        timeProcess = array([self.lineTrackMode(self.frames[3], self.frames[2], self.frames[1], self.frames[0])])
                        #timeProcess = array([self.lineTrackMode(self.frames[0], self.frames[1], self.frames[2], self.frames[3])])
                    else:
                        for f in range(self.parts - 1, -1, -1):
                            timeProcess += self.frames[f]
                        
                        timeProcess = array([timeProcess])
                        #timeProcess = array([self.frames[3] + self.frames[2] + self.frames[1] + self.frames[0]])
                        #timeProcess = array([self.frames[0] + self.frames[1] + self.frames[2] + self.frames[3]])

                    pred1 = list(self.model.predict(timeProcess)[0])
                    #pred2 = self.model.predict_classes(timeProcess)

                    maxValue = max(pred1)
                    wordEncode = pred1.index(maxValue)
                    word = self.encoder.classes_[wordEncode]

                    print("----------------------------------------")
                    
                    print(str(wordEncode) + "-" + word + ": " + str(maxValue))
                    #print(str(len(pred1[0])))
                    #print(str(pred1[0]))
                    #print(str(pred2))

            except:
                print("Ops!!!!\n" + traceback.format_exc(1))
                raise Exception('Break')
                



            time.sleep(self.partTime)



    def createLinePose(self, results, line: list) -> list:

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z)
        
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].z)

        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x)
        line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y)
        if WITH_Z: line.append(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].z)

        return line

    def createLineEmptyHand(self, line: list) -> list:
        size = 40
        if WITH_Z: size = 60

        for x in range(size):
            line.append(0)

        return line

    def createLineHand(self, landmark, line: list) -> list:
        #line.append(landmark[mp_holistic.HandLandmark.WRIST].x)
        #line.append(landmark[mp_holistic.HandLandmark.WRIST].y)

        line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_CMC].z)

        line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_MCP].z)

        line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_IP].z)

        line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.THUMB_TIP].z)

        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP].z)

        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_PIP].z)

        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_DIP].z)

        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP].z)

        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].z)

        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].z)

        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_DIP].z)

        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].z)

        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_MCP].z)

        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_PIP].z)

        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_DIP].z)

        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.RING_FINGER_TIP].z)

        line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_MCP].z)

        line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_PIP].z)

        line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_DIP].z)

        line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].x)
        line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].y)
        if WITH_Z: line.append(landmark[mp_holistic.HandLandmark.PINKY_TIP].z)

        return line
        
