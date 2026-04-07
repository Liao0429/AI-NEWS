import numpy as np
import random
import yaml
import os
from src.strategies import KeywordStrategy, HashStrategy, calculate_percentage_return, calculate_stats
from src.strategy_momentum import MomentumStrategy
from src.strategy_meanreversion import MeanReversionStrategy


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"警告: 加载配置文件失败: {e}")
        # 返回默认配置
        return {
            'random_seed': 42,
            'num_runs': 100,
            'noise_rate': 0.02,
            'trading_cost': 0.001,  # 0.1% per trade
            'slippage': 0.0005,  # 0.05%
            'max_position': 1  # 每次交易最大1单位
        }


def calculate_net_return(gross_return, trade_price, trading_cost, slippage):
    """计算净收益（扣除交易成本和滑点）"""
    # 交易成本：0.1% per trade
    cost = trade_price * trading_cost
    # 滑点：0.05%
    slippage_cost = trade_price * slippage
    # 总成本
    total_cost = cost + slippage_cost
    # 净收益 = 毛收益 - 总成本
    net_return = gross_return - total_cost
    return net_return


def run_strategy_with_cost(aligned_df, strategy, random_seed=42, trading_cost=0.001, slippage=0.0005, max_position=1):
    """运行策略（包含交易成本和滑点）"""
    # 生成信号
    signals = aligned_df['news_text'].apply(strategy.generate_signal).tolist()
    # 应用仓位限制
    signals = [min(max(signal, -max_position), max_position) for signal in signals]
    
    # 向量化计算
    trade_prices = aligned_df['trade_price'].values
    future_prices = aligned_df['future_price'].values
    
    # 计算收益
    gross_returns = []
    net_returns = []
    
    for i, (signal, trade_price, future_price) in enumerate(zip(signals, trade_prices, future_prices)):
        if trade_price == 0:
            gross_returns.append(0)
            net_returns.append(0)
            continue
        
        # 计算百分比毛收益
        pct_return = calculate_percentage_return(future_price, trade_price)
        gross_return = signal * pct_return * trade_price  # 转换为绝对收益
        gross_returns.append(gross_return)
        
        # 计算净收益
        net_return = calculate_net_return(gross_return, trade_price, trading_cost, slippage)
        net_returns.append(net_return)
    
    return signals, gross_returns, net_returns


def run_backtest(aligned_df, num_runs=100, random_seed=42):
    """运行回测（多次重复）"""
    # 加载配置
    config = load_config()
    
    # 获取交易参数
    trading_cost = config.get('trading_cost', 0.001)  # 默认0.1% per trade
    slippage = config.get('slippage', 0.0005)  # 默认0.05%
    max_position = config.get('max_position', 1)  # 默认每次交易max 1 unit
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    all_keyword_gross_returns = []
    all_keyword_net_returns = []
    all_hash_gross_returns = []
    all_hash_net_returns = []
    all_momentum_gross_returns = []
    all_momentum_net_returns = []
    all_meanreversion_gross_returns = []
    all_meanreversion_net_returns = []
    
    for run in range(num_runs):
        # 运行Keyword策略（带成本）
        keyword_strategy = KeywordStrategy(random_seed=random_seed + run)
        keyword_signals, keyword_gross_returns, keyword_net_returns = run_strategy_with_cost(
            aligned_df, 
            keyword_strategy,
            random_seed + run, 
            trading_cost, 
            slippage, 
            max_position
        )
        
        # 运行Hash策略（带成本）
        hash_strategy = HashStrategy(random_seed=random_seed + run + 1000)
        hash_signals, hash_gross_returns, hash_net_returns = run_strategy_with_cost(
            aligned_df, 
            hash_strategy,
            random_seed + run + 1000, 
            trading_cost, 
            slippage, 
            max_position
        )
        
        # 运行Momentum策略（带成本）
        momentum_strategy = MomentumStrategy(random_seed=random_seed + run + 2000)
        momentum_signals, momentum_gross_returns, momentum_net_returns = run_strategy_with_cost(
            aligned_df, 
            momentum_strategy,
            random_seed + run + 2000, 
            trading_cost, 
            slippage, 
            max_position
        )
        
        # 运行MeanReversion策略（带成本）
        meanreversion_strategy = MeanReversionStrategy(random_seed=random_seed + run + 3000)
        meanreversion_signals, meanreversion_gross_returns, meanreversion_net_returns = run_strategy_with_cost(
            aligned_df, 
            meanreversion_strategy,
            random_seed + run + 3000, 
            trading_cost, 
            slippage, 
            max_position
        )
        
        # 计算总收益
        all_keyword_gross_returns.append(np.sum(keyword_gross_returns))
        all_keyword_net_returns.append(np.sum(keyword_net_returns))
        all_hash_gross_returns.append(np.sum(hash_gross_returns))
        all_hash_net_returns.append(np.sum(hash_net_returns))
        all_momentum_gross_returns.append(np.sum(momentum_gross_returns))
        all_momentum_net_returns.append(np.sum(momentum_net_returns))
        all_meanreversion_gross_returns.append(np.sum(meanreversion_gross_returns))
        all_meanreversion_net_returns.append(np.sum(meanreversion_net_returns))
    
    return {
        'keyword_gross_returns': all_keyword_gross_returns,
        'keyword_net_returns': all_keyword_net_returns,
        'hash_gross_returns': all_hash_gross_returns,
        'hash_net_returns': all_hash_net_returns,
        'momentum_gross_returns': all_momentum_gross_returns,
        'momentum_net_returns': all_momentum_net_returns,
        'meanreversion_gross_returns': all_meanreversion_gross_returns,
        'meanreversion_net_returns': all_meanreversion_net_returns
    }


def print_backtest_results(results):
    """打印回测结果"""
    print("=" * 80)
    print("回测结果")
    print("=" * 80)
    
    # Keyword策略
    keyword_gross = np.mean(results['keyword_gross_returns'])
    keyword_net = np.mean(results['keyword_net_returns'])
    print(f"Keyword策略:")
    print(f"  Before cost: {keyword_gross:.4f}")
    print(f"  After cost:  {keyword_net:.4f}")
    print(f"  成本影响:  {keyword_net - keyword_gross:.4f}")
    print()
    
    # Hash策略
    hash_gross = np.mean(results['hash_gross_returns'])
    hash_net = np.mean(results['hash_net_returns'])
    print(f"Hash策略:")
    print(f"  Before cost: {hash_gross:.4f}")
    print(f"  After cost:  {hash_net:.4f}")
    print(f"  成本影响:  {hash_net - hash_gross:.4f}")
    print()
    
    # Momentum策略
    if 'momentum_gross_returns' in results:
        momentum_gross = np.mean(results['momentum_gross_returns'])
        momentum_net = np.mean(results['momentum_net_returns'])
        print(f"Momentum策略:")
        print(f"  Before cost: {momentum_gross:.4f}")
        print(f"  After cost:  {momentum_net:.4f}")
        print(f"  成本影响:  {momentum_net - momentum_gross:.4f}")
        print()
    
    # MeanReversion策略
    if 'meanreversion_gross_returns' in results:
        meanreversion_gross = np.mean(results['meanreversion_gross_returns'])
        meanreversion_net = np.mean(results['meanreversion_net_returns'])
        print(f"MeanReversion策略:")
        print(f"  Before cost: {meanreversion_gross:.4f}")
        print(f"  After cost:  {meanreversion_net:.4f}")
        print(f"  成本影响:  {meanreversion_net - meanreversion_gross:.4f}")
        print()
    
    print("=" * 80)
