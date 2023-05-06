import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
import warnings
import pickle
from sys import exit

def pmf(data):
    return (data.value_counts().sort_index()/len(data))
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    excludeDists = ['levy_stable', 'studentized_range','ncf','genhyperbolic','ksone','geninvgauss','tukeylambda','ncx2','nct','norminvgauss']
    for ii, distribution in enumerate([d for d in _distn_names if not d in excludeDists]):

        print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))

        except Exception:
            pass

    return sorted(best_distributions, key=lambda x:x[2])
def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get same start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

path = "data.csv"
df = pd.read_csv(path)
test_df = df.sample(frac=0.2)   # Randomly choose 20% of the dataset
training_df = df.drop(test_df.T)   # Drop the chosen 20% to get the remaining 80%
training_numerical = training_df.select_dtypes(include=np.number)
training_categorical = training_df.select_dtypes(exclude=np.number)
# These are integer columns that were falsely labeled as int due to existance of NaNs
mislabeled = ["avg6mou","avg6qty","avg6rev","phones","models","truck","rv","lor","adults","income","numbcars","forgntvl","eqpdays"]
pdfs = {"test_df": test_df.T.keys().to_list()}

for col in training_numerical:
    if training_numerical[col].dtype != np.int64 and col not in mislabeled:
        # Limit dataset by Chebyshev to include roughly 93.75% of the dataset
        mean = np.mean(training_numerical[col])
        std = np.std(training_numerical[col])
        k = 4
        data = pd.Series([x for x in training_numerical[col] if mean-k*std<=x<=mean+k*std])
        # The first element in the returned list is the distribution with the least SSE (Sum of Square Error)
        best_dist = best_fit_distribution(data, 200)[0]
        # best_dist is a tuple of two elements. The first element is the distribution type (Exponential, etc...)
        # And the second element is the distibution parameters (Lambda, etc...)
        pdfs[col] = [make_pdf(best_dist[0], best_dist[1])]

        param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
        dist_str = '{}({})'.format(best_dist[0].name, param_str) # Chosen distribution with its parameters
        pdfs[col].append(dist_str)
        print(len(pdfs)-1)

with open("pdfs.pkl","wb") as f:
    # Save the computed pdfs so that they need not be recomputed every time
    pickle.dump(pdfs,f)
with open("pdfs.txt","w+") as f:
    # Save the chosen pdf type and parameters for each column
    for col in pdfs:
        if col != "test_df":
            f.write(col+": "+pdfs[col][1]+"\n\n")

pdfs_churn = dict()
training_numerical_churn = training_numerical[training_df["churn"]==1]
for col in training_numerical_churn:
    if training_numerical_churn[col].dtype != np.int64 and col not in mislabeled:
        # Limit dataset by Chebyshev to include roughly 93.75% of the dataset
        mean = np.mean(training_numerical_churn[col])
        std = np.std(training_numerical_churn[col])
        k = 4
        data = pd.Series([x for x in training_numerical_churn[col] if mean-k*std<=x<=mean+k*std])
        # The first element in the returned list is the distribution with the least SSE (Sum of Square Error)
        best_dist = best_fit_distribution(data, 200)[0]
        # best_dist is a tuple of two elements. The first element is the distribution type (Exponential, etc...)
        # And the second element is the distibution parameters (Lambda, etc...)
        pdfs_churn[col] = [make_pdf(best_dist[0], best_dist[1])]

        param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
        dist_str = '{}({})'.format(best_dist[0].name, param_str) # Chosen distribution with its parameters
        pdfs_churn[col].append(dist_str)
        print(len(pdfs_churn))

with open("pdfs_churn.pkl","wb") as f:
    # Save the computed pdfs so that they need not be recomputed every time
    pickle.dump(pdfs_churn,f)
with open("pdfs_churn.txt","w+") as f:
    # Save the chosen pdf type and parameters for each column
    for col in pdfs_churn:
            f.write(col+": "+pdfs_churn[col][1]+"\n\n")

# Dictionaries for the pmfs and pmfs|churn
probs = dict()
probs_churn = dict()

training_categorical_churn = training_categorical[training_df["churn"]==1]
for col in training_categorical:
    probs[col] = pmf(training_categorical[col])
    probs_churn[col] = pmf(training_categorical_churn[col])
for col in training_numerical:
    data = training_numerical[col]
    data_churn = training_numerical_churn[col]
    if col in mislabeled:
        # Data is mislabeled due to occurance of NaNs. So remove them then calculate the PMF
        data = data.dropna().astype(int)
        data_churn = data_churn.dropna().astype(int)
    if data.dtype==np.int64:
        probs[col] = pmf(data)
        probs_churn[col] = pmf(data_churn)
with open("pmfs.pkl","wb") as f:
    # Save the computed pmfs so that they need not be recomputed every time
    pickle.dump(probs,f)
with open("pmfs_churn.pkl","wb") as f:
    # Save the computed pmfs so that they need not be recomputed every time
    pickle.dump(probs_churn,f)
