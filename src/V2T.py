import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a hand landmarker instance with the image mode:

hands_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/pose_landmarker_full.task'),
    running_mode=VisionRunningMode.IMAGE)

# Initialize hand landmarker
hands_landmarker = HandLandmarker.create_from_options(hands_options)
poses_landmarker = PoseLandmarker.create_from_options(pose_options)


def video2text(video_path):
    video_capture = cv2.VideoCapture(video_path)
    data = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        hand_results = hands_landmarker.detect(mp_img)
        pose_results = poses_landmarker.detect(mp_img)
        if len(hand_results.hand_landmarks) and len(pose_results.pose_landmarks):
            hand_data = [[i.x, i.y, i.z] for i in hand_results.hand_landmarks[0]]
            pose_data = [[i.x, i.y, i.z] for i in pose_results.pose_landmarks[0]]
            if hand_results.handedness[0][0].category_name == "Left":
                data.append(hand_data + [pose_data[i] for i in range(13, 22, 2)])
            else:
                data.append(hand_data + [pose_data[i] for i in range(14, 22, 2)])
    video_capture.release()
    return np.array(data, dtype=np.float32)

if "__main__" == __name__:
    print(video2text("qqdownload").shape)
