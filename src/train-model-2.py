import sys
sys.path.append("../src")
from model import FoodRegressor
from data_loader import FoodDataset
from utils import get_transforms
from optimizer import Optimizer
import torch
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = get_transforms()

base_dataset = FoodDataset(
    csv_path="../data/cleaned-food-data.csv",
    img_dir="../data/data/images",
    transform=transform,
    target_mean=None,
    target_std=None
)

print("Filtered dataset size:", len(base_dataset))

n = len(base_dataset) # total samples
n_val = int(n * 0.2) # 80-20 train val split
n_train = n - n_val

all_indices = torch.randperm(n) # shuffle the data
train_indices = all_indices[:n_train].tolist() # indices for training samples
val_indices = all_indices[n_train:].tolist() # indices for val samples

targets_all = base_dataset.targets # all of the targets in the base dataset
train_targets = targets_all[train_indices]

# Calculate mean and std dev of training set to normalize targets
target_mean = train_targets.mean(axis = 0).astype("float32")
target_std = train_targets.std(axis = 0).astype("float32")

# Check output
print("Training set mean: ", target_mean)
print("Training set standard deviation: ", target_std)

dataset = FoodDataset(
    csv_path="../data/cleaned-food-data.csv",
    img_dir="../data/data/images",
    transform=transform,
    target_mean=target_mean,
    target_std=target_std
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

num_dishes = len(dataset.dish_to_id) # Get the number of total dishes in the dataset

model = FoodRegressor(
    pretrained_model_name = "resnet18",
    num_dishes = num_dishes,
    hidden_dim = 512,
    dish_emb_dim = 32,
    dropout = 0.0,
    out_dim = 5, 
    freeze_pretrained = False
    ).to(device)

opt = Optimizer(
    model = model,
    train_data = train_loader,
    val_data = val_loader,
    epochs = 20,
    lr = 3e-5,
    device = device,
    use_scheduler = True,
    patience = 3 
)

trained_model = opt.train()

save_path = "../models/model_2.pt"
torch.save(trained_model.state_dict(), save_path)
print(f"Saved model to {save_path}")

plot_save_path = "../report/plots/model_2_output.png"
opt.plot(save_path=plot_save_path)
print(f"Saved plot to {plot_save_path}")

results = {
    "train_losses": opt.training_losses,
    "val_losses": opt.validation_losses,
    "target_mean": target_mean.tolist(),
    "target_std": target_std.tolist()
}

with open("../models/model2_metadata.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Saved plot to {plot_save_path}")