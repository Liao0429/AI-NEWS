from src.strategies import BaseStrategy
import pandas as pd
import numpy as np


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略（Mean Reversion Strategy）
    
    核心逻辑：基于价格偏离均值，"高抛低吸"
    - 计算最近N天的移动平均
    - 价格高于均值则做空，低于则做多
    
    关键参数：
    - time_window: 时间窗口（默认20天）
    """
    
    def __init__(self, random_seed: int = 42, time_window: int = 20):
        super().__init__("MeanReversion", random_seed)
        self.time_window = time_window
    
    def generate_signal(self, news_text: str) -> int:
        """
        生成交易信号（基于均值回归策略）
        
        注意：均值回归策略实际需要价格数据，这里为了保持统一接口，
        我们在run方法中处理价格数据，这里仅返回默认信号
        
        Args:
            news_text: 新闻文本
            
        Returns:
            signal: +1 (做多), -1 (做空)
        """
        # 均值回归策略实际依赖价格数据，这里返回默认信号
        # 实际信号生成在run方法中处理
        return 1
    
    def run(self, aligned_df: pd.DataFrame) -> tuple:
        """
        运行均值回归策略
        
        Args:
            aligned_df: 对齐的数据框，包含价格数据
            
        Returns:
            signals: 信号列表
            returns: 收益率列表
        """
        from src.strategies import calculate_percentage_return
        
        # 计算移动平均线
        prices = aligned_df['trade_price'].values
        signals = []
        
        for idx, row in aligned_df.iterrows():
            # 对于均值回归策略，我们需要计算移动平均
            if idx >= self.time_window:
                # 计算过去time_window天的移动平均（只使用T-1及以前的数据）
                window_prices = prices[max(0, idx - self.time_window):idx]
                moving_avg = np.mean(window_prices)
                
                # 价格高于均值则做空，低于则做多
                if row['trade_price'] > moving_avg:
                    signal = -1  # 做空
                else:
                    signal = 1  # 做多
            else:
                # 数据不足，默认做多
                signal = 1
            
            signals.append(signal)
        
        # 计算收益率
        returns = []
        for idx, row in aligned_df.iterrows():
            if row['trade_price'] == 0:
                returns.append(0.0)
                continue
            
            signal = signals[idx]
            pct_return = calculate_percentage_return(row['future_price'], row['trade_price'])
            strategy_return = signal * pct_return
            returns.append(strategy_return)
        
        return signals, returns
