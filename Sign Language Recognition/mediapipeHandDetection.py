import cv2
import mediapipe


# Configure the MediaPipe Hands instance.
cap = cv2.VideoCapture(0)
# Initialize MediaPipe Hands and drawing utilities.
handTracker = mediapipe.solutions.hands
drawing = mediapipe.solutions.drawing_utils
drawingStyles = mediapipe.solutions.drawing_styles

# Configure the MediaPipe Hands instance.
handDetector = handTracker.Hands(static_image_mode=True, min_detection_confidence=0.2)

while True:
    ret, frame = cap.read()
    # Flip the frame horizontally for a mirrored view.
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    # Convert the frame to RGB for MediaPipe processing.
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame with MediaPipe Hands.
    imgMediapipe = handDetector.process(frameRGB)
    # Draw hand landmarks if detected.
    if imgMediapipe.multi_hand_landmarks:
        for handLandmarks in imgMediapipe.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,  # Image to draw.
                handLandmarks,  # Model output.
                handTracker.HAND_CONNECTIONS,  # Hand connections.
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style())
    # Display the frame.
    cv2.imshow('frame', frame)
    # Exit the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed.
# Release resources.
cap.release()
cv2.destroyAllWindows()
