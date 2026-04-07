import random
import warnings

warnings.warn(
    "strategy_rule.py is deprecated and will be removed in future versions. "
    "Please use src.strategies.HashStrategy instead.",
    DeprecationWarning,
    stacklevel=2
)



def generate_signal_rule(base_signal, noise_rate=0.02, random_seed=42):
    """Rule策略：基于基础信号添加噪音"""
    random.seed(random_seed)
    
    if random.random() < noise_rate:
        return -base_signal
    else:
        return base_signal


def run_strategy_rule(aligned_df, base_signals, noise_rate=0.02, random_seed=42):
    """运行Rule策略"""
    random.seed(random_seed)
    
    signals = []
    for idx, row in aligned_df.iterrows():
        base_signal = base_signals[idx]
        signal = generate_signal_rule(base_signal, noise_rate, random_seed + idx)
        signals.append(signal)
    
    returns = []
    for idx, row in aligned_df.iterrows():
        signal = signals[idx]
        ret = signal * (row['future_price'] - row['trade_price'])
        returns.append(ret)
    
    return signals, returns
