from src.strategies import BaseStrategy
import pandas as pd
import numpy as np


class MomentumStrategy(BaseStrategy):
    """动量策略（Momentum Strategy）
    
    核心逻辑：基于价格趋势，"追涨杀跌"
    - 计算最近N天的价格变化率
    - 大于0则做多，小于0则做空
    
    关键参数：
    - time_window: 时间窗口（默认5天）
    """
    
    def __init__(self, random_seed: int = 42, time_window: int = 5):
        super().__init__("Momentum", random_seed)
        self.time_window = time_window
    
    def generate_signal(self, news_text: str) -> int:
        """
        生成交易信号（基于动量策略）
        
        注意：动量策略实际需要价格数据，这里为了保持统一接口，
        我们在run方法中处理价格数据，这里仅返回默认信号
        
        Args:
            news_text: 新闻文本
            
        Returns:
            signal: +1 (做多), -1 (做空)
        """
        # 动量策略实际依赖价格数据，这里返回默认信号
        # 实际信号生成在run方法中处理
        return 1
    
    def run(self, aligned_df: pd.DataFrame) -> tuple:
        """
        运行动量策略
        
        Args:
            aligned_df: 对齐的数据框，包含价格数据
            
        Returns:
            signals: 信号列表
            returns: 收益率列表
        """
        from src.strategies import calculate_percentage_return
        
        # 计算价格变化率（动量）
        price_changes = []
        signals = []
        
        for idx, row in aligned_df.iterrows():
            # 对于动量策略，我们需要过去N天的价格数据
            # 这里简化处理，使用trade_price和过去的价格比较
            # 实际应用中，应该使用专门的价格数据
            
            # 简化的动量计算：如果当前价格高于过去价格则做多，否则做空
            # 这里使用一个简单的逻辑：基于价格趋势
            if idx > 0:
                prev_row = aligned_df.iloc[idx-1]
                price_change = (row['trade_price'] - prev_row['trade_price']) / prev_row['trade_price']
                if price_change > 0:
                    signal = 1  # 做多
                else:
                    signal = -1  # 做空
            else:
                # 第一条数据，默认做多
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
