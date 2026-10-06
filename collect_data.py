import cv2
import mediapipe as mp
import csv
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

camera = cv2.VideoCapture(0)

os.makedirs("data", exist_ok=True)

label = input("Enter sign name: ")

file = open(f"data/{label}.csv", "a", newline="")
writer = csv.writer(file)

print("Press S to save a hand sample")
print("Press Q to quit")

while True:
    success, frame = camera.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        for landmark in hand:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            row = []
            for landmark in hand:
                row.extend([landmark.x, landmark.y, landmark.z])

            writer.writerow(row)
            print("Sample saved!")

    cv2.imshow("Collect Sign Data", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

file.close()
camera.release()
cv2.destroyAllWindows()