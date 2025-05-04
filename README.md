# README.md

Title: Leaf Area Index Prediction - Transfer Learning Competition 2023

Description: UNet-based deep learning solution for cloud-robust satellite image prediction

---

## Overview

This project was developed for the **"Leaf Nothing Behind"** competition hosted on [transfer-learning.org](https://transfer-learning.org/competition). The task focuses on **predicting the Leaf Area Index (LAI)** at time `t` from multi-temporal, multi-modal satellite imagery data provided by Sentinel-1 and Sentinel-2 missions.

The competition challenged participants to design models capable of inferring reliable vegetation index maps even under **cloud occlusion**, leveraging both radar and optical data. You can find more context in the official competition rules [here (PDF)](https://drive.google.com/drive/folders/1u31bpBWvbvrKgCgbavGWp_mNOzZb3zQ4).

---

## Data Description

The dataset includes:
- **Sentinel-2 LAI images** at timesteps t-2 and t-1 (256×256×1)
- **Sentinel-1 radar data** (VV, VH) at timesteps t-2, t-1, and t (256×256×2)
- **Segmentation masks** for cloud detection
- Ground truth LAI image at timestep t

This data is highly correlated spatially and temporally and structured to simulate real-world satellite prediction challenges.

---

## Model Architecture

The project is based on a **U-Net architecture** implemented in PyTorch:
- 4 downsampling + 4 upsampling blocks
- Multi-scale feature fusion
- Support for dropout, batch normalization, and residual inputs
- Custom masked loss to ignore cloud and defective pixels

---

## Project Structure

```
CompetitionLNB/
├── Results_Sub_20_04/                                          # PNG files showing prediction quality and similarity scores
├── src/
│ ├── data.py                                                   # Dataset and dataloader definitions
│ ├── loss.py                                                   # Custom masked MSE loss
│ ├── model.py                                                  # U-Net model definition
│ ├── params.py                                                 # Hyperparameters and paths
│ ├── test.py                                                   # Inference code
│ ├── train.py                                                  # Training loop
│ ├── traitement.py                                             # Utility functions (normalization, masking, loading)
│
├── main.py                                                     # CLI entry point
├── requirements.txt                                            # Python dependencies
├── Competition rules 2023.pdf                                  # Official rulebook from the competition
```


---

## Results

We visualize prediction accuracy using similarity scores between predicted and true LAI maps. See the `Results_Sub_20_04` folder for qualitative evaluation.

- Example Similarity Scores:  
  - Image 1: 0.94  
  - Image 2: 0.94, 0.98  
  - Image 3: 0.98, 0.99  
  - Image 4: 0.87
  - Image 5: 0.96

These results demonstrate the model's generalization to both cloudy and clear conditions.

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

Training
```bash
python main.py --mode train
```

Inference
```bash
python main.py --mode infer --csv_path PATH_TO_CSV --save_infers_under OUTPUT_DIR
```

## Competition Info
Organized by: Jules Salzinger under Transfer Learning

Official site: [transfer-learning.org/competition](https://transfer-learning.org/competition)

Drive (submission guide, resources): [Google Drive](https://drive.google.com/drive/folders/1u31bpBWvbvrKgCgbavGWp_mNOzZb3zQ4)

## License & Publication
Participation implies agreement to co-publish the winning method in a scientific paper with the organizers. See the official rules PDF for more details.
