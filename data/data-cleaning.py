import pandas as pd

## Download the data; save to csv in data folder
df = pd.read_csv("hf://datasets/Codatta/MM-Food-100K/MM-Food-100K.csv")

## Do necessary cleaning here

## Save dataframe to csv
df.to_csv("./data/cleaned-food-data.csv", index=False)