import pandas as pd


df = pd.read_csv('./data/data.csv', parse_dates=True, index_col='Date').dropna()
df = df['2016'] #Taking 12 month data as mentioned in the paper

df.reset_index()
df.to_csv('./data/sp500_2016.csv')

df = df[['A', 'AAL', 'AAP', 'AAPL']]

df.to_csv('./data/sp500_2016_test.csv')