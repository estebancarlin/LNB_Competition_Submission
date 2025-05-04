import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

from src.model import UNet
from src.data import create_generators
from src.params import NUM_EPOCHS, LEARNING_RATE
from src.loss import MaskedMSELoss as MSE

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Fix random seed
seed = 1
torch.manual_seed(seed)

# Model initialization
model = UNet(
    input_channels=3,
    output_classes=1,
    hidden_channels=32,
    dropout_probability=0.0,
    kernel_size=(3, 3)
)

# Data generators
train_generator, val_generator, _ = create_generators()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    model.to(device)

    # Print model summary
    summary(model, input_size=(3, 256, 256))

    # Masked MSE loss with ignored indices:
    # 1 = Saturated or defective
    # 2 = Cast shadows
    # 3 = Cloud shadows
    # 6 = Water
    # 7 = Unclassified
    # 8 = Cloud (medium probability)
    # 9 = Cloud (high probability)
    # 10 = Thin cirrus (thin clouds)
    # 11 = Snow or ice
    mse = MSE(ignore_indices=[1, 2, 3, 8, 9, 10])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch: {epoch}')
        train_loss = []
        model.train()
        train_range = tqdm(train_generator)

        for X, Y in train_range:
            X = X.to(torch.float).to(device)
            Y_true = Y[:, 0, :, :].to(torch.float).to(device)
            Y_mask = Y[:, 1, :, :].to(torch.float).to(device)

            optimizer.zero_grad()
            S2_pred = model(X)
            loss = mse(S2_pred, Y_true, Y_mask)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_range.set_description(f"TRAIN -> Epoch: {epoch:4d} || Loss: {np.mean(train_loss):.4f}")
            train_range.refresh()

        train_losses.append(np.mean(train_loss))

        # Validation phase
        model.eval()
        val_loss = []
        with torch.no_grad():
            for image, target in tqdm(val_generator, desc='Validation'):
                image = image.to(device).to(torch.float)
                y_true = target[:, 0, :, :].to(torch.float).to(device)
                y_mask = target[:, 1, :, :].to(torch.float).to(device)
                y_pred = model(image)

                loss = mse(y_pred, y_true, y_mask)
                val_loss.append(loss.item())

        val_losses.append(np.mean(val_loss))
        print("Validation Loss:", np.mean(val_loss))

    # Plot loss curves
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.show()


# ===========================
# Manual inference visualization block (CPU-only)
# ===========================

# This block should be run in CPU context without Colab

# from src.utils import image_similarity2

# similarity_scores = []
#
# for k in range(TESTING_BATCH_SIZE):
#     ex = X_batch[k].to("cpu").unsqueeze(0)
#     res = model(ex.to(torch.float))
#
#     image = res.to('cpu').detach().numpy().squeeze()
#     reconstruct = image
#     output_image = Y_batch[k][0].to('cpu').detach().numpy().squeeze()
#     mask = Y_batch[k][1].to('cpu').detach().numpy().squeeze()
#
#     # Similarity calculation
#     similarity = image_similarity2(reconstruct, output_image, mask, [k], k)
#     if similarity != 0:
#         similarity_scores.append(similarity)
#     print(f"Similarity between predicted and true image: {similarity}")
#
#     # Mask cleaning
#     mask[(mask != 0) & (mask != 1) & (mask != 2) & (mask != 3) &
#          (mask != 8) & (mask != 9)] = 0
#
#     fig, axs = plt.subplots(1, 6, figsize=(20, 10))
#     axs[0].imshow(X_batch[k][2])
#     axs[1].imshow(X_batch[k][1])
#     axs[2].imshow(X_batch[k][0])
#     axs[3].imshow(Y_batch[k][0].cpu())
#     axs[4].imshow(reconstruct)
#     axs[5].imshow(mask)
#
#     axs[0].set_title('VH Polarization')
#     axs[1].set_title('VV Polarization')
#     axs[2].set_title('S2(t-1)')
#     axs[3].set_title('S2(t) Ground Truth')
#     axs[4].set_title('S2(t) Prediction')
#     axs[5].set_title('S2(t) Mask')
#
#     for ax in axs:
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     plt.show()
#
# # Mean similarity score
# if similarity_scores:
#     mean_similarity = sum(similarity_scores) / len(similarity_scores)
#     print("Mean similarity score:", mean_similarity)
# else:
#     print("EMPTY LIST")