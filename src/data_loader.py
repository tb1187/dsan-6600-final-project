import os
from typing import Sequence, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FoodDataset(Dataset):
    # Initialize
    def __init__(
            self,
            csv_path: str, # where cleaned-food-data.csv is stored
            img_dir: str, # data/data/images
            portion_macro_cols: Sequence[str]= ("portion_total_g", "calories_kcal", "protein_g", "fat_g", "carbohydrate_g"),
            target_mean = None,
            target_std = None,
            transform: Optional[transforms.Compose] = None
    ):
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.portion_macro_cols = list(portion_macro_cols)
        self.transform = transform

        # Putting in some error handling in case there is a mismatch between the dataset and the downloaded images
        valid_mask = self.df["image_url"].apply(lambda name: os.path.exists(os.path.join(self.img_dir, name)))
        self.df = self.df[valid_mask].reset_index(drop = True)
        self.targets = self.df[self.portion_macro_cols].values.astype("float32")

        # Normalize targets
        if target_mean is not None:
            self.target_mean = target_mean
            self.target_std = target_std
            self.targets = (self.targets - target_mean) / target_std
        else:
            self.target_mean = None
            self.target_std = None

        # Create dish name -> ID mapping for future embedding in the model
        dish_names = sorted(self.df["dish_name"].unique())
        self.dish_to_id = {dish: i for i, dish in enumerate(dish_names)}
        self.id_to_dish = {i: dish for dish, i in self.dish_to_id.items()}

        self.dish_ids = self.df["dish_name"].apply(lambda d: self.dish_to_id[d]).values.astype("int64")


    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["image_url"] # name of the image file
        img_path = os.path.join(self.img_dir, img_name) # get the full path to the image

        image = Image.open(img_path).convert("RGB") # load the image, add convert in case some are grayscale for whatever reason

        # Apply transforms if indicated
        if self.transform is not None:
            image = self.transform(image)

        # Extract dish IDs
        dish_id = self.dish_ids[idx]
        dish_id = torch.tensor(dish_id, dtype=torch.long) # convert to tensor for future handling

        target = torch.from_numpy(self.targets[idx]) # Targets

        return image, dish_id, target
    
    def __len__(self):
        return len(self.df)
    