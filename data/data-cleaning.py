import pandas as pd
import numpy as np
import os
import re
import logging
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from ast import literal_eval

## Helper Functions
def ensure_dict(x):
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except:
        return {}
    
def normalize_text(text):
    if pd.isna(text):
        return "unknown"

    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"

def parse_portion(p):

    lst = literal_eval(p)   # safer than literal_eval
    parts = {}
    total = 0

    for item in lst:
        try:
            food, val = item.split(":")
            food = food.strip()
            # Handle gram case
            if val.endswith("g"):
                g = float(val.replace("g", "").strip())
            # Handle liters (convert L → mL → grams)
            elif val.lower().endswith("l"):
                try:
                    liters = float(val.replace("L", "").replace("l", "").strip())
                    g = liters * 1000   # assume density of 1000g/L
                except:
                    g = 0
            # Handle ml
            elif val.lower().endswith("ml"):
                try:
                    ml = float(val.replace("mL", "").replace("ml", "").strip())
                    g = ml              # assume density = 1 g/mL
                except:
                    g = 0
            # Handle pieces (unknown mass → 0)
            elif "piece" in val.lower():
                g = 0
            # Unknown format → 0
            else:
                g = 0

        except Exception:
            g = 0

        parts[food] = g
        total += g

    return parts, total

def download_images(df):

    save_dir = "data/images/"
    os.makedirs(save_dir, exist_ok=True)

    # track number of images per dish
    dish_counters = {}

    for _, row in tqdm(df.iterrows(), total=len(df)):

        url = row["image_url"]          # safer: reference by column name
        dish_name= normalize_text(row["dish_name"])

        # initialize counter
        if dish_name not in dish_counters:
            dish_counters[dish_name] = 1

        try:
            resp = requests.get(url, timeout=10)

            if resp.status_code != 200:
                print(f"BAD STATUS CODE ({resp.status_code}): {url}")
                continue

            # Load as image
            try:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception:
                print(f"CORRUPTED IMAGE: {url}")
                continue

            # create unique filename
            count = dish_counters[dish_name]
            filename = f"{dish_name}_{count:04d}.jpg"
            dish_counters[dish_name] += 1

            # save
            img.save(os.path.join(save_dir, filename))

            # change url name in dataframe to the new filename
            df.loc[_, "image_url"] = filename

        except Exception as e:
            print(f"ERROR downloading {url}: {e}")
            continue

    print(f"\nAll downloads complete! Images saved to: {save_dir}")

def clean_filter_dataset(df):

    ## Drop unnecessary columns
    df = df.drop(columns=["camera_or_phone_prob", "food_prob", "sub_dt", "food_type", "cooking_method"])
    
    ## Standardize dish name column; drop where it is unknown
    df["dish_name"] = df["dish_name"].apply(normalize_text)
    df = df[df["dish_name"] != "unknown"].reset_index(drop=True)

    # Filter dataset to only include 300 most popular dish names; keep only 50 images for each dish
    top_500 = (df["dish_name"].value_counts().nlargest(300).index)
    df = df[df["dish_name"].isin(top_500)].reset_index(drop=True)
    df = (df.groupby("dish_name").head(50).reset_index(drop=True))

    # Parse ingredients
    df["ingredients"] = df["ingredients"].apply(lambda x: literal_eval(x))

    # Extract portion data; break into a dictionary of grams per part and total grams
    df["portion_dict"], df["portion_total_g"] = zip(*df["portion_size"].map(parse_portion))
    df = df.drop(columns=["portion_size"])

    # Clean nutritional profile (break up dictionary into individual columns)
    df["nutritional_profile"] = df["nutritional_profile"].apply(ensure_dict)
    df = pd.concat([df.drop(columns=["nutritional_profile"]),
                df["nutritional_profile"].apply(pd.Series)], axis=1)
    
    return df

def main():
    # Load in dataset from Hugging Face
    df = pd.read_csv("hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv")

    # Clean dataset
    df = clean_filter_dataset(df)

    # Download images
    download_images(df)

    # Save dataset
    df.to_csv("./cleaned-food-data.csv", index = False)

if __name__ == "__main__":
    main()