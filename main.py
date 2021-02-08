import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd






if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
