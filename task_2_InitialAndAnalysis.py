import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import csv


def clean_data(df):
    # This columns have less than 60% data, we can ignore these four columns
    # We don't need the location column because we are going to find if it will rain in Australia(not location specific)
    # We are going to drop the date column too.

    df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'Date'], axis=1)
    df.shape



def categorical_to_numerical(data):
    RainToday_list = ['No', 'Yes']
    data['RainToday_bin'] = pd.Categorical(data.RainToday, ordered=False, categories=RainToday_list).codes

    RainTomorrow_list = ['No', 'Yes']
    data['RainTomorrow_bin'] = pd.Categorical(data.RainTomorrow, ordered=False, categories=RainTomorrow_list).codes

    return data.drop(['RainToday', 'RainTomorrow_'], axis=1)

def clean_date(df):
    # parse the dates, currently coded as strings, into datetime format
    # im have divided the date column into 3 different columns as 'day', 'month' and 'year'
    # to know which day of month in a particular year has more rainfall.
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df.drop(["Date"], inplace=True, axis=1)
    df.info()



def statistics_in_numerical(df):

    # find numerical variables
    numerical = [var for var in df.columns if df[var].dtype != 'O']
    # print('There are {} numerical variables\n'.format(len(numerical)))
    # print('The numerical variables are :', numerical)

    # view summary statistics in numerical variables
    weather_statistics_numerical = (round(df[numerical].describe()))
    pd.set_option("display.max_columns", None)
    print(weather_statistics_numerical)


def draw_boxplots(df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = df.boxplot(column='Rainfall')
    fig.set_title('')
    fig.set_ylabel('Rainfall')

    plt.subplot(2, 2, 2)
    fig = df.boxplot(column='Evaporation')
    fig.set_title('')
    fig.set_ylabel('Evaporation')

    plt.subplot(2, 2, 3)
    fig = df.boxplot(column='WindSpeed9am')
    fig.set_title('')
    fig.set_ylabel('WindSpeed9am')

    plt.subplot(2, 2, 4)
    fig = df.boxplot(column='WindSpeed3pm')
    fig.set_title('')
    fig.set_ylabel('WindSpeed3pm')

def histogram_plot(df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = df.Rainfall.hist(bins=10)
    fig.set_xlabel('Rainfall')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 2)
    fig = df.Evaporation.hist(bins=10)
    fig.set_xlabel('Evaporation')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 3)
    fig = df.WindSpeed9am.hist(bins=10)
    fig.set_xlabel('WindSpeed9am')
    fig.set_ylabel('RainTomorrow')

    plt.subplot(2, 2, 4)
    fig = df.WindSpeed3pm.hist(bins=10)
    fig.set_xlabel('WindSpeed3pm')
    fig.set_ylabel('RainTomorrow')




if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
    # weather = clean_data(weather)
    # statistics_in_numerical(weather)
    # clean_date(weather)
    # weather.to_csv('weather_new.csv')
    # draw_boxplots(weather)
    histogram_plot(weather)
    plt.show()

