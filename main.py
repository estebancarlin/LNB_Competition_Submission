import argparse
from src.train import train
from src.test import test


def main(mode: str, csv_path: str = None, save_path: str = None):
    if mode == 'train':
        train()
    elif mode == 'infer':
        if not csv_path or not save_path:
            raise ValueError("Both --csv_path and --save_infers_under must be provided in 'infer' mode.")
        test(csv_path, save_path)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'infer'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run inference on the model.")

    parser.add_argument(
        '--mode', default='infer', choices=['train', 'infer'],
        help="Choose 'train' to train the model or 'infer' to run inference."
    )
    parser.add_argument(
        '--csv_path', type=str,
        help="Path to the CSV file containing the test data (same format and structure as training data)."
    )
    parser.add_argument(
        '--save_infers_under', type=str,
        help="Path to the directory where inference results will be saved."
    )

    args = parser.parse_args()
    main(args.mode, args.csv_path, args.save_infers_under)