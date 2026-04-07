import pandas as pd
import yfinance as yf
from datetime import datetime


def load_real_news_dataset(filepath='data/real/real_news_dataset.csv'):
    """加载真实新闻数据集"""
    news_df = pd.read_csv(filepath)
    return news_df


def fetch_price_data(asset='AAPL', start_date='2023-01-01', end_date='2024-12-31'):
    """从yfinance下载价格数据"""
    ticker = yf.Ticker(asset)
    price_df = ticker.history(start=start_date, end=end_date)
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    return price_df


def align_news_and_prices(news_df, price_df):
    """对齐新闻和价格数据（T+1交易）"""
    aligned_data = []
    
    for _, news_row in news_df.iterrows():
        try:
            news_date = pd.to_datetime(news_row['date'])
        except:
            continue
        
        future_trading_days = price_df.index[price_df.index > news_date]
        
        if len(future_trading_days) > 0:
            t1_date = future_trading_days[0]
            future_price_days = price_df.index[price_df.index >= t1_date]
            
            if len(future_price_days) > 1:
                future_price_date = future_price_days[1]
                
                trade_price = price_df.loc[t1_date]['Close']
                future_price = price_df.loc[future_price_date]['Close']
                
                aligned_data.append({
                    'news_date': news_row['date'],
                    'news_text': news_row.get('full_text', news_row.get('description', news_row.get('title', ''))),
                    'source': news_row.get('source', ''),
                    'trade_date': t1_date.strftime('%Y-%m-%d'),
                    'trade_price': trade_price,
                    'future_price_date': future_price_date.strftime('%Y-%m-%d'),
                    'future_price': future_price
                })
    
    aligned_df = pd.DataFrame(aligned_data)
    return aligned_df
