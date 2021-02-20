import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

def CheckNanInColumn(df):

    # We see there are some columns with null values.
    # Before we start pre-processing, let's find out which of the columns have maximum null values
    print(df.count().sort_values())
    # print(df.isnull().sum())

    # number of missing values for each row having more than 10 missing values
    # print((df.isnull().sum(axis=1).sort_values(ascending=False) > 10).sum())

    # Summary Stats for Categorical Variables
    # print((df.describe(include='object')))

    # Summary Stats for Numerical variables
    # print((df.describe())

    # print(weather['Location'].value_counts())
    # print(weather['WindGustDir'].value_counts())
    # print(weather['WindDir9am'].value_counts())
    # print(weather['WindDir3pm'].value_counts())
    # print(weather['RainToday'].value_counts())
    # print(weather['RainTomorrow'].value_counts())

    # data_na = (df.isnull().sum() / len(df)) * 100
    # data_miss = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    # f, ax = plt.subplots(figsize=(15, 12))
    # plt.xticks(rotation='90')
    # sns.barplot(x=data_miss.index, y=data_miss)
    # plt.xlabel('Features', fontsize=15)
    # plt.ylabel('Percent of missing values', fontsize=15)
    # plt.title('Percent missing data by feature', fontsize=15)
    # plt.show()

    # Check how data is null of average.............................................
    # number_of_rows = df.shape[0]
    # number_of_nan_in_column = df.isnull().sum(axis=0)
    # print(
    #     pd.concat([number_of_nan_in_column, (number_of_nan_in_column / number_of_rows * 100).round(1)], axis=1).rename(
    #         columns={0: 'Number of NaN', 1: 'Number of NaN in %'}))

    # Data types....................................................................
    # print('Data types of this dataset :')
    # print(list(df.dtypes.unique()))

    # Categorical and numerical valuse.............................................
    # categorical_type_columns = []
    # numerical_type_columns = []
    # for one_column_name in df:
    #     if 'object' in str(df[one_column_name].dtype):
    #         categorical_type_columns.append(one_column_name)
    #     elif 'float' in str(df[one_column_name].dtype):
    #         numerical_type_columns.append(one_column_name)
    #
    # print(categorical_type_columns)
    # print()
    # print(numerical_type_columns)
    # print()
    # print('Categorical type columns : {} / {}'.format(len(categorical_type_columns), len(df.columns)))
    # print('Numerical type columns : {} / {}'.format(len(numerical_type_columns), len(df.columns)))

    #cardinality of categorical columns ............................................
    # print("Categorical column cardinality :")
    # for var in categorical_type_columns:
    #     print('{} : {} labels'.format(var, len(df[var].unique())))



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
