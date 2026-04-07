import hashlib
import re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


POSITIVE_WORDS = ['rise', 'gain', 'positive', 'strong', 'beat', 'up', 'growth',
                  'increase', 'success', 'bull', 'soar', 'surge', 'rally', 'jump']
NEGATIVE_WORDS = ['fall', 'drop', 'negative', 'weak', 'miss', 'down', 'decline',
                  'decrease', 'failure', 'bear', 'crash', 'plunge', 'slump', 'tumble']

_positive_patterns = [re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) for word in POSITIVE_WORDS]
_negative_patterns = [re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) for word in NEGATIVE_WORDS]


def count_keywords(text: str, patterns: List[re.Pattern]) -> int:
    """
    统计文本中关键词的数量（使用单词边界匹配，预编译正则表达式）
    
    Args:
        text: 输入文本
        patterns: 预编译的正则表达式列表
        
    Returns:
        count: 匹配到的关键词数量
    """
    return sum(1 for pattern in patterns if pattern.search(text))


def get_deterministic_hash(text: str) -> int:
    """
    生成确定性的哈希值
    
    Args:
        text: 输入文本
        
    Returns:
        hash_value: 确定性的整数哈希值
    """
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)


def calculate_percentage_return(future_price: float, trade_price: float) -> float:
    """
    计算百分比收益率
    
    Args:
        future_price: 未来价格
        trade_price: 交易价格
        
    Returns:
        return: 百分比收益率
    """
    if trade_price == 0:
        return 0.0
    return (future_price - trade_price) / trade_price


class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str, random_seed: int = 42):
        self.name = name
        self.random_seed = random_seed
    
    def generate_signal(self, news_text: str) -> int:
        """
        生成交易信号
        
        Args:
            news_text: 新闻文本
            
        Returns:
            signal: +1 (做多), -1 (做空)
        """
        raise NotImplementedError
    
    def run(self, aligned_df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """
        运行策略
        
        Args:
            aligned_df: 对齐的数据框
            
        Returns:
            signals: 信号列表
            returns: 收益率列表
        """
        # 使用apply代替iterrows，提高性能
        signals = aligned_df['news_text'].apply(self.generate_signal).tolist()
        
        # 向量化计算收益率
        pct_returns = aligned_df.apply(
            lambda row: calculate_percentage_return(row['future_price'], row['trade_price']),
            axis=1
        ).tolist()
        
        returns = [signal * pct for signal, pct in zip(signals, pct_returns)]
        
        return signals, returns


class KeywordStrategy(BaseStrategy):
    """关键词策略（原LLM策略）"""
    
    def __init__(self, random_seed: int = 42):
        super().__init__("Keyword", random_seed)
    
    def generate_signal(self, news_text: str) -> int:
        pos_count = count_keywords(news_text, _positive_patterns)
        neg_count = count_keywords(news_text, _negative_patterns)
        
        if pos_count > neg_count:
            return 1
        elif neg_count > pos_count:
            return -1
        else:
            text_hash = get_deterministic_hash(news_text)
            return 1 if (text_hash % 2 == 0) else -1


class HashStrategy(BaseStrategy):
    """Hash策略（原Rule策略）"""

    def __init__(self, random_seed: int = 42):
        super().__init__("Hash", random_seed)

    def generate_signal(self, news_text: str) -> int:
        text_hash = get_deterministic_hash(news_text)
        return 1 if (text_hash % 2 == 0) else -1


class RandomStrategy(BaseStrategy):
    """随机策略（基线策略）"""

    def __init__(self, random_seed: int = 42):
        super().__init__("Random", random_seed)
        import random
        random.seed(random_seed)

    def generate_signal(self, news_text: str) -> int:
        import random
        return 1 if random.random() >= 0.5 else -1


def calculate_stats(returns: List[float]) -> Dict[str, float]:
    """
    计算策略统计指标
    
    Args:
        returns: 收益率列表
        
    Returns:
        stats: 统计指标字典
    """
    if not returns:
        return {
            'mean': 0.0,
            'std': 1e-8,
            'sharpe': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0
        }
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    std_return = std_return if std_return > 1e-8 else 1e-8
    
    # 计算年化Sharpe Ratio（假设每日收益率）
    annualized_sharpe = mean_return / std_return * np.sqrt(252)  # 252个交易日
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) * 100
    
    # 计算最大回撤
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    # 确保peak不为0，避免除零错误
    peak = np.maximum(peak, 1e-8)  # 当peak为0或负数时，使用一个小的正数
    drawdown = (peak - cumulative) / peak
    max_drawdown = np.max(drawdown) * 100  # 转换为百分比
    
    # 计算卡玛比率
    calmar_ratio = (mean_return * 252) / (max_drawdown / 100 + 1e-8) if max_drawdown > 0 else 0.0  # 年化均值
    
    # 计算索提诺比率
    negative_returns = [r for r in returns if r < 0]
    downside_std = np.std(negative_returns) if negative_returns else 1e-8
    sortino_ratio = mean_return / downside_std * np.sqrt(252)  # 年化索提诺
    
    return {
        'mean': mean_return,
        'std': std_return,
        'sharpe': annualized_sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio
    }
