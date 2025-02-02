from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns
from CNNModel import CNNModel


# Function to plot the confusion matrix.
def plotConfusionMatrix(confusionMatrix, classNames):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusionMatrix, annot=True, fmt='g', xticklabels=classNames, yticklabels=classNames)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# Load the trained model.
model = CNNModel()
# model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth"))
model.load_state_dict(torch.load("CNN_model_number_SIBI.pth"))
model.eval()

# Load the testing dataset.
# data = pd.read_excel("../Data/alphabet_testing_data.xlsx", header=0)
data = pd.read_excel("../Data/numbers_testing_data.xlsx", header=0)

data.pop("CHARACTER")  # Remove unnecessary column.
groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()

# Reshape features to match model input.
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()
coordinates = [coordinates]
groupValue = groupValue.to_numpy()

# Make predictions.
predictions = []
with torch.no_grad():
    for inputs in coordinates:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

# Calculate metrics.
accuracy = accuracy_score(predictions, groupValue)
precision = precision_score(groupValue, predictions, average='weighted', zero_division=0)
recall = recall_score(groupValue, predictions, average='weighted', zero_division=0)
f1 = f1_score(groupValue, predictions, average='weighted', zero_division=0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}")
print(f"Recall: {recall * 100:.2f}")
print(f"F1-Score: {f1 * 100:.2f}")

# Define class names for the confusion matrix.
# classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# Compute the confusion matrix.
confusionMatrix = confusion_matrix(groupValue, predictions)

# Plot the confusion matrix.
plotConfusionMatrix(confusionMatrix, classNames)
