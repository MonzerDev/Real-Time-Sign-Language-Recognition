import cv2
import mediapipe
import torch
import pandas as pd
import numpy as np
from CNNModel import CNNModel

# Load the model.
model = CNNModel()
model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth"))
# model.load_state_dict(torch.load("CNN_model_number_SIBI.pth"))

# Set up video capture and MediaPipe Hands.
cap = cv2.VideoCapture(0)
handTracker = mediapipe.solutions.hands
drawing = mediapipe.solutions.drawing_utils
drawingStyles = mediapipe.solutions.drawing_styles

# Initialize the MediaPipe Hands detector.
handDetector = handTracker.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Define the classes.

classes = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25
}

# classes = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8}
model.eval()

while True:
    ret, frame = cap.read()
    # Flip the frame horizontally for a mirrored view.
    height, width, _ = frame.shape
    # Convert the frame to RGB for MediaPipe processing.
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgMediapipe = handDetector.process(frameRGB)

    coordinates = []
    x_Coordinates = []
    y_Coordinates = []
    z_Coordinates = []

    if imgMediapipe.multi_hand_landmarks:
        for handLandmarks in imgMediapipe.multi_hand_landmarks:
            # Draw the hand landmarks on the frame.
            drawing.draw_landmarks(
                frame,
                handLandmarks,
                handTracker.HAND_CONNECTIONS,
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style())

            data = {}
            # Extract and normalize landmark coordinates.
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
        # Bounding box around the hand.
        x1 = int(min(x_Coordinates) * width) - 10
        y1 = int(min(y_Coordinates) * height) - 10
        x2 = int(max(x_Coordinates) * width) - 10
        y2 = int(max(y_Coordinates) * height) - 10

        predictions = []
        # Convert landmarks to model input.
        coordinates = pd.DataFrame(coordinates)
        coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
        coordinates = torch.from_numpy(coordinates).float()

        # Predict the class.
        with torch.no_grad():
            outputs = model(coordinates)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        predicted_character = [key for key, value in classes.items() if value == predictions[0]][0]
        # Display the prediction.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    # Show the frame.
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed.
# Release resources.
cap.release()
cv2.destroyAllWindows()
