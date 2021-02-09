import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

def CheckNanInColumn(df):

    # We see there are some columns with null values.
    # Before we start pre-processing, let's find out which of the columns have maximum null values
    # print(df.count().sort_values())
    print(df.isnull().sum())

    # number of missing values for each row having more than 10 missing values
    # print((df.isnull().sum(axis=1).sort_values(ascending=False) > 10).sum())

    # Summary Stats for Categorical Variables
    # print((df.describe(include='object')))

    # Summary Stats for Numerical variables
    # print((df.describe())

    # Check how data is null of average
    # Evaporation_percent = (((df['Evaporation'].isnull().sum()) / 145461) * 100)
    # Sunshine_percent = (((df['Sunshine'].isnull().sum()) / 145461) * 100)
    # Cloud9am_percent = (((df['Cloud9am'].isnull().sum()) / 145461) * 100)
    # Cloud3pm_percent = (((df['Cloud3pm'].isnull().sum()) / 145461) * 100)
    # print("Evaporation percent", Evaporation_percent)
    # print("Sunshine percent", Sunshine_percent)
    # print("Cloud9am percent", Cloud9am_percent)
    # print("Cloud3pm percent", Cloud3pm_percent)

    # print(weather['Location'].value_counts())
    # print(weather['WindGustDir'].value_counts())
    # print(weather['WindDir9am'].value_counts())
    # print(weather['WindDir3pm'].value_counts())

    data_na = (df.isnull().sum() / len(df)) * 100
    data_miss = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=data_miss.index, y=data_miss)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()


def balance_positive_negative_class(df):
    # Plotting balance between positive and negative classes
    weather['RainTomorrow'].value_counts().plot(kind='bar')
    plt.show()


if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
    # print('Size of weather data frame is :', weather.shape)
    CheckNanInColumn(weather)
    # balance_positive_negative_class(weather)
    # print(weather.head())
    # print(weather.shape)
    # print(weather.info())
