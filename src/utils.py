from torchvision import transforms
import torch
import json
from model import FoodRegressor
from tqdm import tqdm

# returns standard image preprocessing
# ResNet is input size 224x224
def get_transforms(img_size = 224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize to ResNet input
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # Derived from ImageNet
    ])
    return transform

# Function to load model and normalization stats
def load_model_and_stats(model_path, metadata_path, device = "cpu"):

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Set target mean and std dev
    target_mean = torch.tensor(metadata["target_mean"], dtype = torch.float32).to(device)
    target_std = torch.tensor(metadata["target_std"], dtype = torch.float32).to(device)

    # Initialize the model
    model = FoodRegressor(
        pretrained_model_name="resnet18",
        num_dishes=300,    
        hidden_dim=512,
        dish_emb_dim=32,
        dropout=0.0,
        out_dim=5,
        freeze_pretrained=False
    ).to(device)

    # Load weights
    state_dict = torch.load(model_path, map_location = device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, target_mean, target_std

# Function to evaluate the model
def evaluate_model(model, dataloader, target_mean, target_std, device = "cpu"):
    model.eval()

    mae_total = 0.0
    mse_total = 0.0
    n = 0

    with torch.no_grad():
        for images, dish_ids, targets in tqdm(dataloader, desc = "Evaluating"):

            # Load data
            images = images.to(device)
            dish_ids = dish_ids.to(device)
            targets = targets.to(device)

            # Predict nutrition
            preds_norm = model(images, dish_ids)

            # Denormalize
            preds = preds_norm * target_std + target_mean

            # Sum error and squared error
            mae_total += torch.sum(torch.abs(preds - targets)).item()
            mse_total += torch.sum((preds - targets) ** 2).item()
            n += targets.numel()
    
    # Calculate mae and mse
    mae = mae_total / n
    mse = mse_total / n

    return {"MAE": mae, "MSE": mse}
