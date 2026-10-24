import pandas as pd

# df = pd.read_csv("py-data.csv") #check the idoft data
df = pd.read_csv("empirical-testsoverview.csv") #check the idoft data

print("columns in csv:", df.columns.tolist())
print(df.head(2))