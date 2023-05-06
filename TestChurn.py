import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from sys import exit

### Variables to play with
# include = ["all"]
include = ["rev_Mean","actvsubs","months"]
exclude = ["churn","Customer_ID"]
threshold = 0.5
### There are 4 modes for output:
## 1) verbose: For each customer, prints their number, their calculated churn probability, their actual churn value
##             and the result of the prediction (Correct / False Positive / False Negative).
## 2) lite: For each customer, prints only the result of the prediction.
## 3) update: Acts like verbose but only for every 'printEvery' clients. (1, printEvery, printEvery*2, ...)
## 4) results: Prints only the accuracy of the predictions at the very end.
mode = "lite"
printEvery = 1000

### Beginning of driver code
def closestValue(df, val):
    dist = abs(df.keys() - val)
    return df[dist==min(dist)].values[0]
path = "data.csv"
df = pd.read_csv(path)
with open("pdfs.pkl","rb") as f:
    pdfs = pickle.load(f)
with open("pdfs_churn.pkl","rb") as f:
    pdfs_churn = pickle.load(f)
with open("pmfs.pkl","rb") as f:
    pmfs = pickle.load(f)
with open("pmfs_churn.pkl","rb") as f:
    pmfs_churn = pickle.load(f)
test_df = df.T[pdfs["test_df"]].T.infer_objects()
training_df = df.drop(test_df.T)

test_numerical = test_df.select_dtypes(include=np.number)
test_categorical = test_df.select_dtypes(exclude=np.number)
mislabeled = ["avg6mou","avg6qty","avg6rev","phones","models","truck","rv","lor","adults","income","numbcars","forgntvl","eqpdays"]

prChurn = len(training_df[training_df["churn"]==1])/len(training_df)
currNumber = 0
correct = 0
result = ""
for row in test_df.T:
    base = prChurn
    for col in test_numerical:
        if(col in include or ("all" in include and col not in exclude)):
            val = test_df.T[row][col]
            if not pd.isnull(val):
                if test_numerical[col].dtype!=np.int64 and col not in mislabeled:
                    base = base * closestValue(pdfs_churn[col][0],val)/closestValue(pdfs[col][0],val)
                else:
                    base = base * closestValue(pmfs[col],val)/closestValue(pmfs_churn[col],val)
    for col in test_categorical:
        if(col in include or ("all" in include and col not in exclude)):
            val = test_df.T[row][col]
            if not pd.isnull(val):
                base = base * pmfs_churn[col][val]/pmfs[col][val]
    churn = test_df["churn"].iloc[currNumber]
    if base > threshold:
        if(churn == 1):
            correct += 1
            result = "Correct Prediction"
        else:
            result = "False Positive"
    else:
        if(churn == 0):
            correct += 1
            result = "Correct Prediction"
        else:
            result = "False Negative"
    currNumber += 1
    if mode == "verbose" or (mode == "update" and (currNumber==1 or currNumber%printEvery==0)):
        print("Customer",currNumber)
        print("Probability of churn: {:.3f}".format(base))
        print("Actual Churn Value:",churn)
        print("Prediction result:",result)
        print()
    elif mode == "lite":
        print("Customer {}: {}\n".format(currNumber,result))
print("Accuracy: {:.2f}".format(correct/currNumber*100))
