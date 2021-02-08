import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd



def CheckNanInColumn(df):
    # print(df.isnull().sum())
    Evaporation_percent = (((df['Evaporation'].isnull().sum()) / 145461) * 100)
    Sunshine_percent = (((df['Sunshine'].isnull().sum()) / 145461) * 100)
    Cloud9am_percent = (((df['Cloud9am'].isnull().sum()) / 145461) * 100)
    Cloud3pm_percent = (((df['Cloud3pm'].isnull().sum()) / 145461) * 100)

    # print("Evaporation percent", Evaporation_percent)
    # print("Sunshine percent", Sunshine_percent)
    # print("Cloud9am percent", Cloud9am_percent)
    # print("Cloud3pm percent", Cloud3pm_percent)



if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
    # CheckNanInColumn(weather)
    # print(weather.head())
    # print(weather.shape)
    print(weather.info())
