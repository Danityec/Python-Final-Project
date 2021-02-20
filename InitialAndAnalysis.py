import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.utils import resample




def categorical_to_numerical(df):

    WindGustDir_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindGustDir_bin'] = pd.Categorical(df.WindGustDir, ordered=False, categories=WindGustDir_list).codes+1

    WindDir9am_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindDir9am_bin'] = pd.Categorical(df.WindDir9am, ordered=False, categories=WindDir9am_list).codes+1

    WindDir3pm_list = ['W', 'SE', 'N', 'SSE', 'E', 'S', 'WSW', 'SW', 'SSW', 'WNW', 'NW', 'ENE', 'ESE', 'NE', 'NNW', 'NNE']
    df['WindDir3pm_bin'] = pd.Categorical(df.WindDir3pm, ordered=False, categories=WindDir3pm_list).codes+1

    RainToday_list = ['No', 'Yes']
    df['RainToday_bin'] = pd.Categorical(df.RainToday, ordered=False, categories=RainToday_list).codes+1

    RainTomorrow_list = ['No', 'Yes']
    df['RainTomorrow_bin'] = pd.Categorical(df.RainTomorrow, ordered=False, categories=RainTomorrow_list).codes+1

    df['Location_bin'] = pd.Categorical(df.Location, ordered=False).codes

    return df.drop(['RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir3pm', 'WindDir9am', 'Location'], axis=1)


def dismantle_column_date(df):
    # parse the dates, currently coded as strings, into datetime format
    # i have divided the date column into 3 different columns as 'day', 'month' and 'year'
    # to know which day of month in a particular year has more rainfall.
    tmp = df
    tmp['Date'] = pd.to_datetime(df['Date'])
    tmp['Year'] = tmp['Date'].dt.year
    tmp['Month'] = tmp['Date'].dt.month
    tmp['Day'] = tmp['Date'].dt.day
    tmp.drop(["Date"], inplace=True, axis=1)

    return(tmp)

def statistics_in_numerical(df):

    # find numerical variables
    numerical = [var for var in df.columns if df[var].dtype != 'O']
    print('There are {} numerical variables\n'.format(len(numerical)))
    print('The numerical variables are :', numerical)

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


def remove_row_outliers(df):
    df.drop(df[(df["Rainfall"] > 250)].index, inplace=True)
    df.drop(df[(df["Evaporation"] > 85)].index, inplace=True)
    df.drop(df[(df["WindSpeed9am"] > 80)].index, inplace=True)
    df.drop(df[(df["WindSpeed9am"] > 80)].index, inplace=True)

    return df

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
   return(df.drop_duplicates(subset=['Year', 'Month', 'Day'], keep='first'))


def row_with_more_10_missing_values(df):

    # number of missing values for each row having more than 10 missing values
    print("more than 10 missing values:", ((df.isnull().sum(axis=1).sort_values(ascending=False) > 10).sum()))
    print((df.isnull().sum(axis=1).sort_values(ascending=False) > 10))
    df.dropna((df.isnull().sum(axis=1).sort_values(ascending=False) > 10))
    return df

def check_outliers_with_zero(df):
    # The only column that the value 0 cannot exist is humidity
    zero = df[(df["Humidity9am"] == 0) | (df["Humidity3pm"] == 0)]
    print("value 0 cannot exist is humidity:", len(zero))

    # df.replace(df[(df["Humidity9am"] == 0)].mean(), inplace=True)
    # df.replace(df[(df["Humidity3pm"] == 0)].mean(), inplace=True)



def modified_heatmap(data):
    df = data.select_dtypes(np.number)
    corr_matrix = round(df.corr(), 3)
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1)
    plt.show()


def fields_average(df):
    # numerical - Replace the missing values nan with the mean if the column is numerical .
    df['MinTemp'].fillna(round((df['MinTemp'].mean()), 2), inplace=True)
    df['MaxTemp'].fillna(round((df['MaxTemp'].mean()), 2), inplace=True)
    df['WindSpeed9am'].fillna(round((df['WindSpeed9am'].mean()), 2), inplace=True)
    df['Temp9am'].fillna(round((df['Temp9am'].mean()), 2), inplace=True)
    df['Humidity9am'].fillna(round((df['Humidity9am'].mean()), 2), inplace=True)
    df['WindSpeed3pm'].fillna(round((df['WindSpeed3pm'].mean()), 2), inplace=True)
    df['Rainfall'].fillna(round((df['Rainfall'].mean()), 2), inplace=True)
    df['Temp3pm'].fillna(round((df['Temp3pm'].mean()), 2), inplace=True)
    df['Humidity3pm'].fillna(round((df['Humidity3pm'].mean()), 2), inplace=True)
    df['WindGustSpeed'].fillna(round((df['WindGustSpeed'].mean()), 2), inplace=True)
    df['Pressure9am'].fillna(round((df['Pressure9am'].mean()), 2), inplace=True)
    df['Pressure3pm'].fillna(round((df['Pressure3pm'].mean()), 2), inplace=True)


    # categorical - Replace the missing values nan with the mode if the column is categorical
    df['WindDir3pm_bin'].fillna((df['WindDir3pm_bin'].mode()[0]), inplace=True)
    df['WindGustDir_bin'].fillna((df['WindGustDir_bin'].mode()[0]), inplace=True)
    df['WindDir9am_bin'].fillna((df['WindDir9am_bin'].mode()[0]), inplace=True)
    df['RainToday_bin'].fillna((df['RainToday_bin'].mode()[0]), inplace=True)
    # empty value in target class- what i do? RainTomorrow
    # df['RainTomorrow_bin'].fillna((df['RainTomorrow_bin'].mode()[0]), inplace=True)

    return(df)


def fill_nan_with_other_columns(df):
    df['Sunshine'] = df.apply(
        lambda row: fill_nan_with_other_columns(row['MaxTemp'], row['Humidity9am'], row['Humidity3pm'], row['Cloud3pm'], row['Cloud9am'], row['Temp3pm']), axis=1)
    # df['Evaporation'] = df.apply(
    #     lambda row: fill_nan_with_other_columns(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    # df['Cloud3pm'] = df.apply(
    #     lambda row: fill_nan_with_other_columns(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    # df['Cloud9am'] = df.apply(
    #     lambda row: fill_nan_with_other_columns(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    return df

def balance_target_class(df):
    # Plotting balance between positive and negative classes
    weather['RainTomorrow'].value_counts().plot(kind='bar')
    plt.show()


if __name__ == '__main__':
    weather = pd.read_csv("weather1.csv")

    # check rows with duplicate values and
    # rows_duplicate(weather)

    weather = row_with_more_10_missing_values(weather)


    # divided the date column into 3 different columns as 'day', 'month' and 'year'
    weather = dismantle_column_date(weather)

    # different values with same date
    # weather = different_values_same_date(weather)     ####### not work

    # swipe categorical value to numerical value
    weather = categorical_to_numerical(weather)

    # Filling values in columns by the average of the values in the column
    weather = fields_average(weather)
    # weather = fill_nan_with_other_columns(weather)

    # chek Exceeding the value range
    check_outliers_with_zero(weather)

    # remove all value with nan fields: available 58,237 without nan
    # weather.dropna(inplace=True)
    # weather.to_csv("tmp2.csv")

    weather = remove_row_outliers(weather)

    # chek correlation
    # modified_heatmap(weather)
    weather.to_csv("tmp1.csv")

    # statistics_in_numerical(weather)   #columns may contain outliers.
    # draw_boxplots(weather)   # plot the histograms to check distributions to find out if they are normal or skewed.
    histogram_plot(weather)

    plt.show()



