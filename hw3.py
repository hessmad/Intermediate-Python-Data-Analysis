# Madison Hess
# CSE 163
# Homework 3
# The following program takes in a data set containing years, sex, education
# attainment and the relative percentages for the population percentages given
# the previous classifications.  The following program explores relationships
# over time, across different sexes, and different degree types in the effort
# to gain insights related to education trends across different groups.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set()

# Part 0 - Statistical Functions using Pandas


def completions_between_years(df, year1, year2, sex):
    """
    Takes in a dataframe, a starting and ending year, and sex and returns a
    dataframe which shows the education attainment across different racial
    groups as well as the whole population (given other restrictions) between
    the specified years.
    """
    df = df[(df['Sex'] == sex) & (df['Year'] >= year1) & (df['Year'] < year2)]
    return df


def compare_bachelors_1980(df):
    """
    Takes in a dataframe and returns a tuple containing the total percentage
    of males and females, respectively, who at least attained a bachelor's
    degree in 1980.
    """
    df = df[(df['Year'] == 1980) & (df['Min degree'] == "bachelor's")]
    df = df[(df['Sex'] == 'M') | (df['Sex'] == 'F')]
    df = df[['Sex', 'Total']]
    return tuple(list(df['Total']))


def top_2_2000s(df):
    """
    Takes in a dataframe and returns the two most commonly attained degree
    types and the relative average percentage that attained that type for the
    whole population between 2000 and 2010.
    """
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2010) & (df['Sex'] == 'A')]
    df = df.groupby('Min degree')['Total'].mean()
    result = df.nlargest(2)
    return list(result.items())


def percent_change_bachelors_2000s(df, sex='A'):
    """
    Takes in a dataframe and an optional sex variable (which defaults to all if
    unspecified) and returns the percent change in bachelor's degree attainment
    for that sex between 2000 and 2010.
    """
    df = df[(df['Sex'] == sex)]
    df_00 = df[(df['Year'] == 2000)
               & (df['Min degree'] == "bachelor's")].Total.item()
    df_10 = df[(df['Year'] == 2010)
               & (df['Min degree'] == "bachelor's")].Total.item()
    return (df_10 - df_00)


# Plotting with seaborn


def line_plot_bachelors(df):
    """
    Takes in a dataframe and saves a plot to a .png file which graphically
    represents the change in bachelor's degree attainment for the whole
    population since the 1940s.
    """
    df = df[(df['Sex'] == 'A') & (df['Min degree'] == "bachelor's")]
    sns.lineplot(x='Year', y='Total', data=df)
    plt.savefig('line_plot_bachelors.png', facecolor='w')


def bar_chart_high_school(df):
    """
    Takes in a dataframe and saves a bar chart to a .png file which graphically
    represents the total, male, and female percentages of people who recieced
    a high school diploma in 2009.
    """
    df = df[(df['Min degree'] == 'high school') & (df['Year'] == 2009)]
    df = df[['Sex', 'Total']]
    sns.catplot(x='Sex', y='Total', data=df, kind='bar')
    plt.savefig('bar_chart_high_school.png', facecolor='w')


def plot_hispanic_min_degrees(df):
    """
    Takes in a dataframe and saves a line plot to a .png file which graphically
    represents the change in both percentage attainment of high school diplomas
    and bachelor's degrees amongst Hispanics from 1990 to 2010.
    """
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2010) & (df['Sex'] == 'A')]
    df = df[(df['Min degree'] == 'high school')
            | (df['Min degree'] == "bachelor's")]
    df = df[['Year', 'Min degree', 'Hispanic']]
    sns.lineplot(x='Year', y='Hispanic', hue='Min degree', data=df)
    plt.xlim(1990, 2010)
    plt.savefig('plot_hispanic_min_degrees.png', facecolor='w')


# Part 2: Machine Learning using scikit-learn


def fit_and_predict_degrees(df):
    """
    Develops a simple machine learning program using decision tree regression
    and the year, sex, and min degree from our data to develop a model to
    predict the total percent attainment for each group.
    """
    df = df.loc[:, ('Year', 'Sex', 'Min degree', 'Total')].dropna()
    X = df.loc[:, df.columns != 'Total']
    X = pd.get_dummies(X)
    y = df['Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = tree.DecisionTreeRegressor()
    model = model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print('Training MSE: ', mean_squared_error(y_train, y_train_pred))
    print('Test MSE', mean_squared_error(y_test, y_test_pred))


def main():
    data = pd.read_csv('hw3-nces-ed-attainment.csv', na_values='---')
    completions_between_years(data, 2007, 2008, 'F')
    compare_bachelors_1980(data)
    top_2_2000s(data)
    percent_change_bachelors_2000s(data)
    line_plot_bachelors(data)
    bar_chart_high_school(data)
    plot_hispanic_min_degrees(data)
    fit_and_predict_degrees(data)


if __name__ == '__main__':
    main()
