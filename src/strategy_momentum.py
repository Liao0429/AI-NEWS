import pandas as pd
import numpy as np
from typing import List, Tuple


class MomentumStrategy:
    """动量策略"""
    
    def __init__(self, time_window: int = 5, random_seed: int = 42):
        """
        初始化动量策略
        
        Args:
            time_window: 时间窗口
            random_seed: 随机种子
        """
        self.time_window = time_window
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def run(self, df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """
        运行动量策略
        
        Args:
            df: 交易数据
            
        Returns:
            Tuple[List[int], List[float]]: 信号列表和收益列表
        """
        signals = []
        returns = []
        
        for i in range(len(df)):
            if i < self.time_window:
                # 数据不足，返回中性信号
                signal = 0
            else:
                # 计算过去time_window天的收益率
                past_prices = df['trade_price'].iloc[i-self.time_window:i].values
                if len(past_prices) > 1:
                    returns_past = (past_prices[1:] - past_prices[:-1]) / past_prices[:-1]
                    avg_return = np.mean(returns_past)
                    
                    # 如果平均收益为正，买入；否则卖出
                    if avg_return > 0:
                        signal = 1
                    else:
                        signal = -1
                else:
                    signal = 0
            
            signals.append(signal)
            
            # 计算收益
            if i < len(df) - 1:
                trade_price = df['trade_price'].iloc[i]
                future_price = df['future_price'].iloc[i]
                if trade_price != 0:
                    pct_return = (future_price - trade_price) / trade_price
                    returns.append(signal * pct_return)
                else:
                    returns.append(0)
            else:
                returns.append(0)
        
        return signals, returns
    
    def generate_signal(self, news_text: str) -> int:
        """
        生成交易信号（兼容接口）
        
        Args:
            news_text: 新闻文本
            
        Returns:
            int: 交易信号 (-1=卖出, 0=持有, 1=买入)
        """
        # 动量策略不依赖新闻文本，返回随机信号
        return np.random.choice([-1, 0, 1])