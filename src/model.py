import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    """
    Downscaling block: MaxPool followed by two Conv2d layers with dropout,
    batch normalization, and ReLU activations.
    """

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upscaling block: Upsample followed by concatenation and two Conv2d layers
    with dropout, batch normalization, and ReLU activations.
    """

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)  # Skip connection
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.

    Args:
        input_channels (int): Number of input channels.
        output_classes (int): Number of output segmentation classes.
        hidden_channels (int): Number of base filters (doubles at each down layer).
        dropout_probability (float): Dropout probability in conv layers.
        kernel_size (int or tuple): Convolution kernel size for initial block.
    """

    def __init__(self, input_channels, output_classes, hidden_channels, dropout_probability, kernel_size):
        super().__init__()

        # Initial convolution block
        self.initial_block = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Downscaling path
        self.down1 = Down(hidden_channels, hidden_channels * 2, dropout_probability)
        self.down2 = Down(hidden_channels * 2, hidden_channels * 4, dropout_probability)
        self.down3 = Down(hidden_channels * 4, hidden_channels * 8, dropout_probability)
        self.down4 = Down(hidden_channels * 8, hidden_channels * 8, dropout_probability)

        # Upscaling path
        self.up1 = Up(hidden_channels * 16, hidden_channels * 4, dropout_probability)
        self.up2 = Up(hidden_channels * 8, hidden_channels * 2, dropout_probability)
        self.up3 = Up(hidden_channels * 4, hidden_channels, dropout_probability)
        self.up4 = Up(hidden_channels * 2, hidden_channels, dropout_probability)

        # Output layer
        self.out_conv = nn.Conv2d(hidden_channels, output_classes, kernel_size=1)
        self.activation = nn.Sigmoid()  # For binary/multilabel segmentation

    def forward(self, x):
        # Encoding
        x1 = self.initial_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        # Output
        logits = self.out_conv(x9)
        return self.activation(logits)