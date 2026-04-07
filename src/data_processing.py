import pandas as pd
import yfinance as yf
from datetime import datetime, time
import pytz


def load_real_news_dataset(filepath='data/real/real_news_dataset.csv'):
    """加载真实新闻数据集"""
    news_df = pd.read_csv(filepath)
    return news_df


def fetch_price_data(asset='AAPL', start_date='2023-01-01', end_date='2024-12-31'):
    """从yfinance下载价格数据"""
    ticker = yf.Ticker(asset)
    price_df = ticker.history(start=start_date, end=end_date)
    # 转换为UTC时区
    price_df.index = pd.to_datetime(price_df.index).tz_localize('UTC')
    return price_df


def align_news_and_prices(news_df, price_df):
    """对齐新闻和价格数据（T+1交易）"""
    aligned_data = []
    
    for _, news_row in news_df.iterrows():
        try:
            news_date = pd.to_datetime(news_row['date'])
            # 转换为UTC时区
            if news_date.tzinfo is None:
                news_date = news_date.tz_localize('UTC')
            else:
                news_date = news_date.tz_convert('UTC')
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
                
                # 标注市场开闭时间（UTC时间，美股：9:30-16:00 ET = 13:30-20:00 UTC）
                market_open = t1_date.replace(hour=13, minute=30, second=0, microsecond=0)
                market_close = t1_date.replace(hour=20, minute=0, second=0, microsecond=0)
                
                aligned_data.append({
                    'news_date': news_date.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'news_text': news_row.get('full_text', news_row.get('description', news_row.get('title', ''))),
                    'source': news_row.get('source', ''),
                    'trade_date': t1_date.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'trade_price': trade_price,
                    'market_open': market_open.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'market_close': market_close.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'future_price_date': future_price_date.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'future_price': future_price
                })
    
    aligned_df = pd.DataFrame(aligned_data)
    return aligned_df
