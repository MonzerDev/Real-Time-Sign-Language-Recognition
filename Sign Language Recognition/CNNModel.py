from torch.nn import Linear, ReLU, Sequential, Conv1d, MaxPool1d, Module, BatchNorm1d, Dropout


class CNNModel(Module):
    """
    A Convolutional Neural Network (CNN) model for sequence processing.
    This model consists of multiple convolutional layers followed by batch normalization,
    ReLU activations, dropout, and fully connected layers for classification.
    """
    def __init__(self):
        """
        Initialize the CNN model with convolutional and linear layers.
        """
        super().__init__()

        # Define the convolutional layers with batch normalization, activation, and pooling.
        self.cnnLayers = Sequential(
            Conv1d(63, 32, 3, 1, 2),
            BatchNorm1d(32),  # Batch normalization for stable training.
            ReLU(),  # ReLU activation for non-linearity.

            Conv1d(32, 64, 3, 1, 2),
            BatchNorm1d(64),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            Conv1d(64, 128, 3, 1, 2),
            BatchNorm1d(128),  # Output channels (128) match the filter size of the previous layer.
            ReLU(),
            Dropout(p=0.3),  # Dropout to prevent over-fitting.

            Conv1d(128, 256, 3, 1, 2),
            BatchNorm1d(256),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            Conv1d(256, 512, 5, 1, 2),
            BatchNorm1d(512),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=2),

            Conv1d(512, 512, 5, 1, 2),
            BatchNorm1d(512),
            ReLU(),
            Dropout(p=0.4),
        )

        # Define the linear layers for classification.
        self.linearLayers = Sequential(
            Linear(512, 26),
            BatchNorm1d(26),
            ReLU(),
        )

    # Define the forward pass
    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_classes).
        """
        # Pass the input through the convolutional layers.
        x = self.cnnLayers(x)

        # Flatten the output for the linear layers.
        x = x.view(x.size(0), -1)

        # Pass the flattened output through the linear layers.
        x = self.linearLayers(x)

        return x
