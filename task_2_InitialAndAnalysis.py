import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd



def categorical_to_numerical(df):
    WindGustDir_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindGustDir_bin'] = pd.Categorical(df.WindGustDir, ordered=False, categories=WindGustDir_list).codes+1

    WindDir9am_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindDir9am_bin'] = pd.Categorical(df.WindDir9am, ordered=False, categories=WindDir9am_list).codes+1

    WindDir3pm_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindDir3pm_bin'] = pd.Categorical(df.WindDir3pm, ordered=False, categories=WindDir3pm_list).codes+1

    RainToday_list = ['No', 'Yes']
    df['RainToday_bin'] = pd.Categorical(df.RainToday, ordered=False, categories=RainToday_list).codes

    RainTomorrow_list = ['No', 'Yes']
    df['RainTomorrow_bin'] = pd.Categorical(df.RainTomorrow, ordered=False, categories=RainTomorrow_list).codes

    df['Location_bin'] = pd.Categorical(df.Location, ordered=False).codes

    return df.drop(['RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir3pm', 'WindDir9am', 'Location'], axis=1)

def clean_date(df):
    # parse the dates, currently coded as strings, into datetime format
    # im have divided the date column into 3 different columns as 'day', 'month' and 'year'
    # to know which day of month in a particular year has more rainfall.
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df.drop(["Date"], inplace=True, axis=1)


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


def rows_duplicate(df):

    # Removing rows with all duplicate values
    print(df.drop_duplicates(keep=False, inplace=True))


def different_values_same_date(df):

    # Check that there are no different values on the same date
   print(df.drop_duplicates(subset=['Year', 'Month', 'Day'], keep='first'))


def row_with_more_10_missing_values(df):

    # number of missing values for each row having more than 10 missing values
    print((df.isnull().sum(axis=1).sort_values(ascending=False) > 10).sum())
    print((df.isnull().sum(axis=1).sort_values(ascending=False) > 10))
    return df.dropna((df.isnull().sum(axis=1).sort_values(ascending=False) > 10))


def check_outliers(df):

    # return((df['MinTemp'] >= -8.0) & (df['MinTemp'] < 34.0))
    # return ((df['MaxTemp'] >= -8.0) & (df['MaxTemp'] < 34.0))
    return df['MinTemp'].between(-8.0, 34.0, inclusive=False)


def modified_heatmap(data):
    df = data.select_dtypes(np.number)
    corr_matrix = round(df.corr(), 3)
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1)
    plt.show()


def fields_average(df):
    df['MinTemp'].fillna(round((df['MinTemp'].mean()), 2), inplace=True)
    df['MaxTemp'].fillna(round((df['MaxTemp'].mean()), 2), inplace=True)
    df['WindSpeed9am'].fillna(round((df['WindSpeed9am'].mean()), 2), inplace=True)
    df['Temp9am'].fillna(round((df['Temp9am'].mean()), 2), inplace=True)
    df['Humidity9am'].fillna(round((df['Humidity9am'].mean()), 2), inplace=True)
    df['WindSpeed3pm'].fillna(round((df['WindSpeed3pm'].mean()), 2), inplace=True)
    # df['RainToday'].fillna(round((df['RainToday'].mean()), 2), inplace=True)
    df['Rainfall'].fillna(round((df['Rainfall'].mean()), 2), inplace=True)
    df['Temp3pm'].fillna(round((df['Temp3pm'].mean()), 2), inplace=True)
    # empty value in target class- what i do?
    # df['WindDir3pm'].fillna(round((df['WindDir3pm'].mean()), 2), inplace=True)
    df['Humidity3pm'].fillna(round((df['Humidity3pm'].mean()), 2), inplace=True)
    df['WindGustSpeed'].fillna(round((df['WindGustSpeed'].mean()), 2), inplace=True)
    # df['WindGustDir'].fillna(round((df['WindGustDir'].mean()), 2), inplace=True)
    df['WindDir9am'].fillna(round((df['WindDir9am'].mean()), 2), inplace=True)
    df['Pressure9am'].fillna(round((df['Pressure9am'].mean()), 2), inplace=True)
    df['Pressure3pm'].fillna(round((df['Pressure3pm'].mean()), 2), inplace=True)




if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")
    # rows_duplicate(weather)
    # clean_date(weather)
    weather = categorical_to_numerical(weather)
    # weather = row_with_more_10_missing_values(weather)
    # weather = check_outliers(weather)
    weather.to_csv('weather_new.csv')
    # different_values_same_date(weather)
    fields_average(weather)

    print(len(weather))
    weather.dropna(inplace=True)
    modified_heatmap(weather)

    # statistics_in_numerical(weather)
    # draw_boxplots(weather)
    # histogram_plot(weather)
    plt.show()

