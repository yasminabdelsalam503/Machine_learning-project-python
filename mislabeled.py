import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\Karim\\Desktop\\NETW504\\data.csv")
df_numerical = df.select_dtypes(include=np.number)

for col in df_numerical:
    data = df_numerical[col]
    if(data.dtype!=np.int64 and [x%1 for x in data.dropna()].count(0)-len(data.dropna())==0):
        print(col)
