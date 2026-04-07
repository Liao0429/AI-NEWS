import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import yfinance as yf
import pytz


class DataProcessor:
    """数据处理类，支持数据质量控制、多资产支持和数据缓存"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = os.path.join(self.base_dir, 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.utc = pytz.UTC
    
    def load_price_data(self, ticker, start_date, end_date):
        """
        加载价格数据，支持缓存
        
        Args:
            ticker: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 价格数据
        """
        # 生成缓存文件名
        cache_file = os.path.join(self.cache_dir, f'{ticker}_{start_date}_{end_date}.csv')
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            print(f'✓ 从缓存加载 {ticker} 价格数据')
            price_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # 转换为UTC时区
            price_data.index = price_data.index.tz_localize('UTC')
            return price_data
        
        # 下载数据
        print(f'下载 {ticker} 价格数据...')
        price_data = yf.download(ticker, start=start_date, end=end_date)
        
        # 数据质量控制
        price_data = self._process_price_data(price_data)
        
        # 转换为UTC时区
        price_data.index = price_data.index.tz_localize('UTC')
        
        # 保存到缓存
        price_data.to_csv(cache_file)
        print(f'✓ 保存 {ticker} 价格数据到缓存')
        
        return price_data
    
    def _process_price_data(self, price_data):
        """
        处理价格数据，包括缺失值和异常值
        
        Args:
            price_data: 原始价格数据
            
        Returns:
            DataFrame: 处理后的价格数据
        """
        # 如果数据有MultiIndex或特殊结构，先扁平化
        if isinstance(price_data.columns, pd.MultiIndex):
            price_data.columns = price_data.columns.get_level_values(0)
        
        # 将价格列转换为数值类型
        for col in ['Close', 'High', 'Low', 'Open']:
            if col in price_data.columns:
                price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
        
        # 处理缺失值
        price_data = price_data.dropna()
        
        # 处理异常值（价格突变）
        if len(price_data) > 1:
            # 计算价格变化率
            price_changes = price_data['Close'].pct_change()
            # 移除极端值（超过5个标准差）
            std_dev = price_changes.std()
            if pd.notna(std_dev) and std_dev > 0:
                mask = abs(price_changes) < 5 * std_dev
                price_data = price_data[mask]
        
        return price_data
    
    def convert_to_trading_format(self, raw_data, price_data):
        """
        将原始新闻数据转换为交易格式，确保时间顺序正确
        
        核心要求：news_time < prediction_time < trade_time < future_price_time
        
        Args:
            raw_data: 原始新闻数据（DataFrame）
            price_data: 价格数据（DataFrame，index为日期）
        
        Returns:
            DataFrame: 转换后的交易格式数据
        """
        print('='*80)
        print('Step 1: 数据金融化改造')
        print('='*80)
        print()
        
        trading_data = []
        
        for idx, row in raw_data.iterrows():
            # 获取新闻时间
            if 'date' in row:
                news_date_str = row['date']
            elif 'news_date' in row:
                news_date_str = row['news_date']
            else:
                continue
            
            try:
                news_datetime = datetime.strptime(news_date_str, '%Y-%m-%d')
                # 转换为UTC时区
                news_datetime = self.utc.localize(news_datetime)
            except (ValueError, TypeError):
                continue
            
            # 设置时间（新闻在当天收盘后）
            news_time = news_datetime.replace(hour=17, minute=0, second=0)
            
            # prediction_time: 新闻时间+1小时（分析时间）
            prediction_time = news_time + timedelta(hours=1)
            
            # 找到下一个交易日
            trade_date = None
            future_date = None
            
            for i in range(1, 10):
                check_date = news_datetime + timedelta(days=i)
                check_date_str = check_date.strftime('%Y-%m-%d')
                
                # 检查日期是否在价格数据中
                if check_date_str in price_data.index.strftime('%Y-%m-%d'):
                    if trade_date is None:
                        trade_date = check_date_str
                        # 转换为UTC时间
                        trade_time = self.utc.localize(datetime.strptime(trade_date, '%Y-%m-%d')).replace(hour=9, minute=30, second=0)
                        # 标注市场开闭时间（UTC时间，美股：9:30-16:00 ET = 13:30-20:00 UTC）
                        market_open = self.utc.localize(datetime.strptime(trade_date, '%Y-%m-%d')).replace(hour=13, minute=30, second=0)
                        market_close = self.utc.localize(datetime.strptime(trade_date, '%Y-%m-%d')).replace(hour=20, minute=0, second=0)
                    elif future_date is None:
                        future_date = check_date_str
                        future_price_time = self.utc.localize(datetime.strptime(future_date, '%Y-%m-%d')).replace(hour=16, minute=0, second=0)
                        break
            
            if trade_date and future_date:
                # 验证时间顺序
                assert news_time < prediction_time, f"news_time >= prediction_time: {news_time} >= {prediction_time}"
                assert prediction_time < trade_time, f"prediction_time >= trade_time: {prediction_time} >= {trade_time}"
                assert trade_time < future_price_time, f"trade_time >= future_price_time: {trade_time} >= {future_price_time}"

                # 获取价格（处理MultiIndex）
                try:
                    close_series = price_data.loc[trade_date, 'Close']
                    if hasattr(close_series, 'iloc'):
                        trade_price = float(close_series.iloc[0])
                    else:
                        trade_price = float(close_series)
                except (KeyError, TypeError):
                    continue

                try:
                    close_series = price_data.loc[future_date, 'Close']
                    if hasattr(close_series, 'iloc'):
                        future_price = float(close_series.iloc[0])
                    else:
                        future_price = float(close_series)
                except (KeyError, TypeError):
                    continue

                # 计算真实return（不使用原数据的label！）
                if trade_price == 0 or pd.isna(trade_price):
                    continue
                price_return = (future_price - trade_price) / trade_price
                
                trading_data.append({
                    'news_time': news_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'news_date': news_date_str,
                    'news_text': row.get('full_text', row.get('title', '')),
                    'source': row.get('source', 'Unknown'),
                    'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'trade_time': trade_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'trade_date': trade_date,
                    'trade_price': trade_price,
                    'market_open': market_open.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'market_close': market_close.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'future_price_time': future_price_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    'future_price_date': future_date,
                    'future_price': future_price,
                    'price_return': price_return
                })
        
        df_trading = pd.DataFrame(trading_data)
        
        print(f'✓ 转换了 {len(df_trading)} 条数据')
        print(f'✓ 时间顺序验证通过：news_time < prediction_time < trade_time < future_price_time')
        print()
        
        return df_trading
    
    def load_from_news_to_forecast_data(self, ticker='BTC-USD', sample_size=150):
        """
        加载From_News_to_Forecast的数据并进行金融化改造
        
        Args:
            ticker: 资产代码（AAPL, MSFT, GOOGL, BTC-USD）
            sample_size: 采样数量
            
        Returns:
            DataFrame: 转换后的交易格式数据
        """
        print('='*80)
        print(f'加载From_News_to_Forecast数据 ({ticker})')
        print('='*80)
        print()
        
        # 加载新闻数据
        news_path = os.path.join(self.base_dir, 'data', 'data', 'raw_news_data', 'bitcoin_news.json')
        
        if not os.path.exists(news_path):
            raise FileNotFoundError(f'News data not found at: {news_path}')
        
        with open(news_path, 'r', encoding='utf-8') as f:
            bitcoin_news = json.load(f)
        
        print(f'✓ 加载了 {len(bitcoin_news)} 条新闻')
        
        # 处理新闻数据
        processed_news = []
        for news in bitcoin_news:
            pub_time = news.get('publication_time', '')
            if pub_time:
                try:
                    date_obj = datetime.strptime(pub_time, '%Y-%m-%d %H:%M:%S')
                    date_str = date_obj.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            processed_news.append({
                'date': date_str,
                'title': news.get('title', ''),
                'full_text': news.get('full_article', ''),
                'source': 'From_News_to_Forecast',
                'url': news.get('link', '')
            })
        
        df_news = pd.DataFrame(processed_news)
        df_news = df_news.sort_values('date').reset_index(drop=True)
        
        # 采样
        if len(df_news) > sample_size:
            sample_step = len(df_news) // sample_size
            selected_indices = range(0, len(df_news), sample_step)[:sample_size]
            df_news = df_news.iloc[selected_indices].copy()
        
        print(f'✓ 使用 {len(df_news)} 条新闻')
        print(f'✓ 日期范围: {df_news["date"].min()} to {df_news["date"].max()}')
        
        # 加载价格数据
        print()
        print('加载价格数据...')
        
        min_date = df_news['date'].min()
        max_date = df_news['date'].max()
        
        start_date = (datetime.strptime(min_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = (datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
        
        price_data = self.load_price_data(ticker, start_date, end_date)
        
        print(f'✓ 下载了 {len(price_data)} 天价格数据')
        
        # 转换为交易格式
        df_trading = self.convert_to_trading_format(df_news, price_data)
        
        return df_trading
    
    def load_multi_asset_data(self, tickers=['AAPL', 'MSFT', 'GOOGL', 'BTC-USD'], sample_size=150):
        """
        加载多资产数据
        
        Args:
            tickers: 资产代码列表
            sample_size: 每个资产的采样数量
            
        Returns:
            Dict[str, DataFrame]: 各资产的交易格式数据
        """
        multi_asset_data = {}
        
        for ticker in tickers:
            print(f'\n' + '='*100)
            print(f'处理资产: {ticker}')
            print('='*100)
            
            try:
                df_trading = self.load_from_news_to_forecast_data(ticker, sample_size)
                multi_asset_data[ticker] = df_trading
                print(f'✓ 成功加载 {ticker} 数据')
            except Exception as e:
                print(f'✗ 加载 {ticker} 数据失败: {e}')
        
        return multi_asset_data


# 兼容旧接口
def convert_to_trading_format(raw_data, price_data):
    """
    兼容旧接口
    """
    processor = DataProcessor()
    return processor.convert_to_trading_format(raw_data, price_data)


def load_and_prepare_from_news_to_forecast_data():
    """
    兼容旧接口
    """
    processor = DataProcessor()
    return processor.load_from_news_to_forecast_data()


def load_multi_asset_data(tickers=['AAPL', 'MSFT', 'GOOGL', 'BTC-USD'], sample_size=150):
    """
    加载多资产数据
    """
    processor = DataProcessor()
    return processor.load_multi_asset_data(tickers, sample_size)

