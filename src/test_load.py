import pandas as pd

print("Pandas version:", pd.__version__)

df = pd.read_csv("data/monday.csv")

print("Shape:", df.shape)
print(df.head())


