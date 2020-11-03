#! python3

import wiki_scrap
from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

url = 'https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Ukraine'

source = requests.get(url)
soup = BeautifulSoup(source.text, 'html.parser')

tb1 = 'wikitable mw-datatable collapsible collapsed'
table = soup.find_all('table', {'class': tb1})

tb2 = 'wikitable mw-datatable collapsible'
table_july = soup.find('table', class_=tb2)

df_april = wiki_scrap.make_df(table[1])
df_may = wiki_scrap.make_df(table[2])
df_june = wiki_scrap.make_df(table[3])
df_july = wiki_scrap.make_july_df(table_july)

df_all = df_april, df_may, df_june, df_july
df = pd.DataFrame()
df = df.append((df_all), ignore_index=True)
df.columns = ['Date', 'New Cases', 'Total Cases', 'New Deaths', 'Total Deaths']
# pd.set_option("display.max_rows", None, "display.max_columns", None)
df1 = df.drop(['Date'], axis=1)
df1 = df1.astype(float)

# df1 = df1.values
# plt.style.use('ggplot')
# plt.plot(df1['New Cases'][:21])
# plt.show()
# print(df1.describe().round(2))
# print(df1.corr().round(2))
# df1.plot(kind='scatter', x='New Deaths', y='New Cases', figsize =(8,6))
# plt.show()

X = df1[['Total Cases']]
Y = df1['New Cases']

model = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
	test_size=0.3, random_state=0)

model.fit(X_train, Y_train)
# print(model.intercept_.round(2))
# print(model.coef_.round(2))
y_test_pred = model.predict(X_test)
# print(y_test_pred)
# print(Y_test)

# Outputs: [24.42606323]
# print(model.score(X_test, Y_test).round(2))
# from sklearn.metrics import mean_squared_error
# print(mean_squared_error(Y_test, y_test_pred).round(2))

# plt.scatter(X_test, Y_test, 
#   label='testing data');
# plt.plot(X_test, y_test_pred,
#   label='prediction', linewidth=3)
# plt.xlabel('Total Cases'); plt.ylabel('New Cases')
# plt.legend(loc='upper left')
# plt.show()
# X2 = df1[['Total Deaths', 'Total Cases']]
# X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y,
# 	test_size = 0.3, random_state=1)
# model2 = LinearRegression()
# model2.fit(X2_train, Y_train)

# print(model2.intercept_.round(2))
# print(model2.coef_.round(2))

# y_test_pred2 = model2.predict(X2_test)
# print(mean_squared_error(Y_test, y_test_pred2).round(2))
# print(model2.score(X2_test, Y_test).round(2))
