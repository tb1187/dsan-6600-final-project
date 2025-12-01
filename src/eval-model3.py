import sys
sys.path.append("../src")

import torch
import numpy as np
import json

from data_loader import FoodDataset
from utils import get_transforms, load_model_and_stats, evaluate_model
from torch.utils.data import Subset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = get_transforms()

with open("../models/model3_metadata.json", "r") as f:
    meta3 = json.load(f)

test_indices = meta3["test_indices"]
target_mean_raw = meta3["target_mean"]
target_std_raw = meta3["target_std"]

test_dataset = Subset(FoodDataset(
    csv_path = "../data/cleaned-food-data.csv",
    img_dir   = "../data/data/images",
    transform = transform,
    target_mean = target_mean_raw,
    target_std  = target_std_raw
),
test_indices)

print("Eval dataset size:", len(test_dataset))

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model3, mean3, std3 = load_model_and_stats(
    "../models/model_3.pt",
    "../models/model3_metadata.json",
    device=device
)

results = evaluate_model(model3, test_loader, mean3, std3, device=device)
print("MAE:", results["MAE"])
print("MSE:", results["MSE"])

target_mean = torch.tensor(target_mean_raw, dtype=torch.float32, device=device)
target_std  = torch.tensor(target_std_raw,  dtype=torch.float32, device=device)

all_preds = []
all_targets = []

model3.eval()

with torch.no_grad():
    for images, dish_ids, targets in test_loader:

        # Load data
        images = images.to(device)
        dish_ids = dish_ids.to(device)
        targets = targets.to(device)

        # Predict nutrition
        preds_norm = model3(images, dish_ids)

        # Denormalize
        preds_denorm = preds_norm * target_std + target_mean
        targets_denorm = targets * target_std + target_mean

        all_preds.append(preds_denorm.cpu().numpy())
        all_targets.append(targets_denorm.cpu().numpy())

# Stack predictions and targets
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# Calculate per target MAE and RMSE
mae_per_target = np.mean(np.abs(all_preds - all_targets), axis = 0)
rmse_per_target = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis = 0))

target_names = ["portion_total_g", "calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]

print("\nPer-target errors (Model 3, test set):")
for name, mae, rmse in zip(target_names, mae_per_target, rmse_per_target):
    print(f"{name:16s}  MAE = {mae:8.2f}   RMSE = {rmse:8.2f}")

# Save path
save_path = "../outputs/model3_eval.json"

# Convert to floats
mae_per_target_list  = [float(x) for x in mae_per_target]
rmse_per_target_list = [float(x) for x in rmse_per_target]

# Output format
results_dict = {
    "overall_MAE": results["MAE"],
    "overall_MSE": results["MSE"],
    "per_target": {
        name: {
            "MAE": mae,
            "RMSE": rmse
        }
        for name, mae, rmse in zip(target_names, mae_per_target_list, rmse_per_target_list)
    }
}

# Save file
with open(save_path, "w") as f:
    json.dump(results_dict, f, indent=4)

print(f"Saved evaluation results to {save_path}")