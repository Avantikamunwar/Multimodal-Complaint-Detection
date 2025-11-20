import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/processed/dataset.csv"
MODEL_SAVE_PATH = "models/best_model.pt"
ENCODER_SAVE_PATH = "models/label_encoders.pkl"

MAX_LENGTH = 512
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-5
SEED = 42
