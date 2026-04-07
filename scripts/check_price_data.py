"""
检查价格数据格式
"""
import pandas as pd
import os

# 检查AAPL价格数据
aapl_path = 'data/cache/AAPL_2017-12-02_2021-06-10.csv'
if os.path.exists(aapl_path):
    df = pd.read_csv(aapl_path, index_col=0, parse_dates=True)
    print('AAPL数据:')
    print(df.head())
    print(f'Shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')
    print(f'Close类型: {df["Close"].dtype}')
    print()
else:
    print('AAPL数据文件不存在')

# 检查BTC-USD价格数据
btc_path = 'data/cache/BTC-USD_2017-12-02_2021-06-10.csv'
if os.path.exists(btc_path):
    df = pd.read_csv(btc_path, index_col=0, parse_dates=True)
    print('BTC-USD数据:')
    print(df.head())
    print(f'Shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')
    print(f'Close类型: {df["Close"].dtype}')
else:
    print('BTC-USD数据文件不存在')
