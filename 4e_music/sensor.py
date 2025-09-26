# sensor.py
import cv2
import mediapipe as mp
from typing import Callable, Optional
from contracts import PoseFrame, Landmark, now_ms

class PoseSensor:
    def __init__(self, cam_index: int = 0, width: int = 640, height: int = 480):
        self.cam_index = cam_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.cb: Optional[Callable[[PoseFrame], None]] = None
        self.width, self.height = width, height
        self.running = False

        self.mp_pose  = mp.solutions.pose.Pose(model_complexity=1, enable_segmentation=False)
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def on_frame(self, cb: Callable[[PoseFrame], None]):
        self.cb = cb

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # corpo
            pose_res = self.mp_pose.process(image)
            body = []
            if pose_res.pose_landmarks:
                for lm in pose_res.pose_landmarks.landmark:
                    body.append(Landmark(lm.x, lm.y, lm.z, lm.visibility))

            # mãos
            hands_res = self.mp_hands.process(image)
            left, right = [], []
            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                for lmks, handed in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                    arr = []
                    for lm in lmks.landmark:
                        arr.append(Landmark(lm.x, lm.y, lm.z, 1.0))
                    if handed.classification[0].label.lower() == "left":
                        left = arr
                    else:
                        right = arr

            pf = PoseFrame(t=now_ms(), left_hand=left, right_hand=right, body=body)

            if self.cb:
                self.cb(pf)

            # opcional: mostrar preview
            cv2.imshow("PoseSensor Preview (q para sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

        self._cleanup()

    def stop(self):
        self.running = False

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.mp_pose.close()
        self.mp_hands.close()
