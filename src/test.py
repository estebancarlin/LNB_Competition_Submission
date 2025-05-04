import os
import pickle
import numpy as np
import torch

import src.params as PARAM
from src.model import UNet
from src.data import create_prediction


def test(csv_path: str, save_infers_under: str):
    """
    Run inference on a dataset specified by a CSV file and save predictions.

    Args:
        csv_path (str): Path to the CSV file containing test image triplets.
        save_infers_under (str): Directory where the prediction results will be saved.
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load checkpoint
    model = UNet(
        input_channels=3,
        output_classes=1,
        hidden_channels=PARAM.HIDDEN_CHANNELS,
        dropout_probability=PARAM.DROPOUT_RATE,
        kernel_size=(3, 3)
    ).to(device)

    checkpoint_path = os.path.join('checkpoint', 'unet2_24+1+15epochs25%Dropout2000data.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load test data
    loader = create_prediction(csv_path)
    total_batches = len(loader)
    results = {"outputs": [], "paths": []}

    # Inference loop
    for i, (X, moy, image_name) in enumerate(loader):
        X = X.to(torch.float).to(device)
        moy = moy.to(torch.float).to(device)

        with torch.no_grad():
            prediction = model(X)
            combined_result = (prediction + moy)[0]  # Remove batch dimension

        results["outputs"].append(combined_result.detach().cpu().numpy())
        results["paths"] += image_name

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"[{i + 1}/{total_batches}] Inference complete.")

    # Format results for saving
    results["outputs"] = np.expand_dims(np.concatenate(results["outputs"], axis=0), axis=-1)

    # Prepare save path
    csv_base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_dir = os.path.join(save_infers_under, csv_base_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save results to pickle
    with open(os.path.join(save_dir, "results.pickle"), "wb") as f:
        pickle.dump(results, f)