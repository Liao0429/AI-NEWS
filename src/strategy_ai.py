import random
import warnings

warnings.warn(
    "strategy_ai.py is deprecated and will be removed in future versions. "
    "Please use src.strategies.KeywordStrategy instead.",
    DeprecationWarning,
    stacklevel=2
)



def generate_signal_ai(news_text):
    """AI策略：从新闻生成信号"""
    positive_keywords = ['earnings', 'beat', 'growth', 'record', 'dividend', 'increase', 
                        'announces', 'partnership', 'unveils', 'high', 'boost', 'expands']
    negative_keywords = ['delay', 'concerns', 'downgraded', 'lawsuit', 'investigation', 
                        'decline', 'issues', 'problems', 'below', 'cannibalization', 'pressure']
    
    news_lower = str(news_text).lower()
    
    positive_count = sum(1 for kw in positive_keywords if kw in news_lower)
    negative_count = sum(1 for kw in negative_keywords if kw in news_lower)
    
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return -1
    else:
        return 1 if random.random() < 0.5 else -1


def run_strategy_ai(aligned_df, random_seed=42):
    """运行AI策略"""
    random.seed(random_seed)
    
    signals = []
    for idx, row in aligned_df.iterrows():
        signal = generate_signal_ai(row['news_text'])
        signals.append(signal)
    
    returns = []
    for idx, row in aligned_df.iterrows():
        signal = signals[idx]
        ret = signal * (row['future_price'] - row['trade_price'])
        returns.append(ret)
    
    return signals, returns
