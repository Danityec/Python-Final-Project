import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


def corr(df):
    correlation = df.corr()
    plt.figure(figsize=(16, 12))
    plt.title('Correlation Heatmap of Rain in Australia Dataset')
    ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
    plt.show()


# def identifying_days(df):



if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
    corr(weather)
    # identifying_days(weather)