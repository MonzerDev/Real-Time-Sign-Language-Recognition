Real-Time Sign Language Recognition

This project implements a real-time recognition system for sign language alphabets and numbers using a Convolutional 
Neural Network (CNN) and MediaPipe for hand landmark detection. The system captures live video input, processes hand 
gestures, and classifies them into corresponding sign language alphabets or numbers.

---

Project Structure

- CNNModel.py: Defines the Convolutional Neural Network (CNN) architecture used for classifying hand gestures.
- training.py: Script used for training the CNN model on datasets of hand gestures or numbers.
- testCNN.py: Script for testing the performance of the trained CNN model on a test dataset.
- mediapipeHandDetection.py: Integrates MediaPipe to perform real-time hand detection and display landmarks through the webcam.
- realTime.py: Main script that integrates the CNN model and MediaPipe for real-time hand gesture recognition and classification.
- handLandMarks.py: Processes MediaPipe's hand landmarks for generating datasets suitable for training the CNN model.
- numbers_testing_data.xlsx: Example processed dataset for testing numeric gesture recognition.
- CNN_model_alphabet_SIBI.pth: Pre-trained CNN model weights for sign language alphabets classification.
- CNN_model_number_SIBI.pth: Pre-trained CNN model weights for numeric gesture classification.

---

How to Run the Project

1. Install Dependencies

Ensure Python is installed. Install the required Python packages using:

pip install -r requirements.txt

If you don't have a requirements.txt file, manually install the necessary packages:

pip install opencv-python mediapipe torch numpy pandas

2. Running Real-Time Recognition

For real-time sign language or numeric gesture recognition, run:

python realTime.py

This will activate your webcam and start detecting and classifying hand gestures in real-time.

3. Training the Model (Optional)

To train the CNN model from scratch using a dataset of hand gestures, run:

python training.py

4. Testing the Model (Optional)

To evaluate the trained CNN model's performance on a test dataset, run:

python testCNN.py

---

How It Works

1. Hand Landmark Detection: 
   - MediaPipe detects and tracks hand landmarks in real-time from the webcam feed.

2. Feature Extraction:
   - Hand landmarks are processed and normalized to be used as input features for the CNN model.

3. Gesture Classification:
   - The CNN model classifies the input features into one of the predefined sign language alphabets (A-Z) or numeric gestures (1-9).

4. Real-Time Feedback:
   - The classified gesture is displayed in real-time, providing immediate feedback to the user.

---

Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyTorch
- Pandas
- NumPy

---

Notes

- The system supports both alphabetic and numeric gestures based on the pre-trained model loaded 
  (CNN_model_alphabet_SIBI.pth or CNN_model_number_SIBI.pth).
- Ensure the training and test datasets are preprocessed and structured correctly as required by the CNN model.

---

Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

---

Contact

For any questions or suggestions, feel free to contact me at (monzerkoukou@gmail.com).
