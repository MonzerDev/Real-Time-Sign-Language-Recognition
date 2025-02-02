import os
import mediapipe
import cv2
from PIL import Image
import pandas as pd
import numpy as np


# Initialize the MediaPipe Hands class for hand tracking.
handTracker = mediapipe.solutions.hands
handDetector = handTracker.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.2)

# DATA_FOLDER = '../Data/alphabet_data'
# DATA_FOLDER = '../Data/alphabet_testing_data'
# DATA_FOLDER = '../Data/numbers_data'
DATA_FOLDER = '../Data/numbers_testing_data'
# Ensure UTF-8 encoding for the environment.
os.environ["PYTHONIOENCODING"] = "utf-8"

coordinates = []  # List to store data for all characters.
index = 0  # Index for group value.
# Process each character folder.
for file in os.listdir(DATA_FOLDER):
    for imgPath in os.listdir(os.path.join(DATA_FOLDER, str(file))):
        fullImgPath = os.path.join(DATA_FOLDER, file, imgPath).replace('\\', '/')
        image = Image.open(fullImgPath)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if img is None:
            print(f"Failed to load image from {fullImgPath}")
            continue  # Skip this iteration if the image failed to load.

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgMediapipe = handDetector.process(imgRGB)

        x_Coordinates = []
        y_Coordinates = []
        z_Coordinates = []

        if imgMediapipe.multi_hand_landmarks:
            for handLandmarks in imgMediapipe.multi_hand_landmarks:
                data = {}
                data['CHARACTER'] = file
                data['GROUPVALUE'] = index
                # Collect coordinates and normalize.
                for i in range(len(handLandmarks.landmark)):
                    lm = handLandmarks.landmark[i]
                    x_Coordinates.append(lm.x)
                    y_Coordinates.append(lm.y)
                    z_Coordinates.append(lm.z)
                # Apply Min-Max normalization.
                for i, landmark in enumerate(handTracker.HandLandmark):
                    lm = handLandmarks.landmark[i]
                    data[f'{landmark.name}_x'] = lm.x - min(x_Coordinates)
                    data[f'{landmark.name}_y'] = lm.y - min(y_Coordinates)
                    data[f'{landmark.name}_z'] = lm.z - min(z_Coordinates)
                coordinates.append(data)
    index += 1
# Convert the collected data to a DataFrame.
df = pd.DataFrame(coordinates)

# excel_path = "alphabet_data.xlsx"
# excel_path = "alphabet_testing_data.xlsx"
# excel_path = "numbers_data.xlsx"
excel_path = "numbers_testing_data.xlsx"

# Save the DataFrame to an Excel file.
df.to_excel(excel_path, index=False)
