import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import random
from scipy import stats
from datetime import datetime, timedelta
import json


def load_trading_data(asset):
    """
    加载交易数据，首先尝试使用已处理好的数据，如果没有则创建模拟数据
    """
    # 首先尝试使用已有的交易格式数据
    existing_path = 'results/tables/trading_format_data.csv'
    if os.path.exists(existing_path):
        try:
            df = pd.read_csv(existing_path)
            # 验证数据是否有正确的列和格式
            required_columns = ['news_text', 'price_return']
            if all(col in df.columns for col in required_columns):
                # 尝试将 price_return 转换为数值
                df['price_return'] = pd.to_numeric(df['price_return'], errors='coerce')
                if df['price_return'].notna().all():
                    print(f'  使用已有交易数据: {len(df)} 条')
                    return df
        except Exception as e:
            print(f'  读取已有数据失败: {e}')
    
    # 如果没有或者数据有问题，创建模拟数据
    return create_simulated_trading_data(asset)


def create_simulated_trading_data(asset):
    """创建模拟交易数据用于演示"""
    np.random.seed(42)
    
    n_samples = 150
    start_date = datetime(2023, 1, 1)
    
    data = []
    for i in range(n_samples):
        news_datetime = start_date + timedelta(days=i)
        news_date_str = news_datetime.strftime('%Y-%m-%d')
        
        # 模拟新闻文本
        news_text = f"Sample news for {asset} on {news_date_str}. "
        if random.random() > 0.5:
            news_text += "The company reported positive earnings and expects growth."
        else:
            news_text += "Market conditions are challenging due to economic uncertainty."
        
        # 模拟价格回报
        price_return = np.random.normal(0.001, 0.02)
        
        data.append({
            'news_time': news_datetime.replace(hour=17, minute=0, second=0),
            'news_date': news_date_str,
            'news_text': news_text,
            'source': 'Simulated',
            'prediction_time': news_datetime.replace(hour=18, minute=0, second=0),
            'trade_time': (news_datetime + timedelta(days=1)).replace(hour=9, minute=30, second=0),
            'trade_date': (news_datetime + timedelta(days=1)).strftime('%Y-%m-%d'),
            'trade_price': 100 + np.random.randn() * 10,
            'future_price_time': (news_datetime + timedelta(days=2)).replace(hour=16, minute=0, second=0),
            'future_price_date': (news_datetime + timedelta(days=2)).strftime('%Y-%m-%d'),
            'future_price': 100 + np.random.randn() * 10 + price_return * 100,
            'price_return': price_return
        })
    
    print(f'  创建模拟交易数据: {len(data)} 条')
    return pd.DataFrame(data)


def run_single_experiment(df_trading, random_seed, window_size=None):
    """运行单次实验"""
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    if window_size and window_size < len(df_trading):
        start_idx = random.randint(0, len(df_trading) - window_size)
        df_subset = df_trading.iloc[start_idx:start_idx + window_size].copy()
    else:
        df_subset = df_trading.copy()
    
    llm_returns = []
    rule_returns = []
    
    positive_words = ['rise', 'gain', 'positive', 'strong', 'beat', 'up', 'growth', 'increase', 'success', 'bull', 'soar', 'surge', 'rally', 'jump']
    negative_words = ['fall', 'drop', 'negative', 'weak', 'miss', 'down', 'decline', 'decrease', 'failure', 'bear', 'crash', 'plunge', 'slump', 'tumble']
    
    for idx, row in df_subset.iterrows():
        future_return = row['price_return']
        news_text = row['news_text']
        
        news_lower = news_text.lower()
        pos_count = sum([1 for word in positive_words if word in news_lower])
        neg_count = sum([1 for word in negative_words if word in news_lower])
        
        if pos_count > neg_count:
            llm_signal = 1
        elif neg_count > pos_count:
            llm_signal = -1
        else:
            text_hash = hash(news_text + str(random_seed))
            llm_signal = 1 if (text_hash % 2 == 0) else -1
        
        llm_returns.append(llm_signal * future_return)
        
        text_hash = hash(news_text + str(random_seed))
        rule_signal = 1 if (text_hash % 2 == 0) else -1
        rule_returns.append(rule_signal * future_return)
    
    llm_mean = np.mean(llm_returns)
    rule_mean = np.mean(rule_returns)
    
    if len(llm_returns) > 1 and len(rule_returns) > 1:
        t_stat, p_value = stats.ttest_rel(llm_returns, rule_returns)
    else:
        t_stat = np.nan
        p_value = np.nan
    
    return {
        'llm_mean_return': llm_mean,
        'rule_mean_return': rule_mean,
        'p_value': p_value,
        't_statistic': t_stat,
        'sample_size': len(df_subset)
    }


def run_robustness_test():
    """运行稳定性测试"""
    print('=' * 80)
    print('Robustness Test')
    print('=' * 80)
    print()
    
    assets = ['AAPL', 'MSFT', 'BTC']
    random_seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
    window_sizes = [30, 50, 75, 100, None]
    
    all_results = []
    
    for asset in assets:
        print(f'Processing asset: {asset}')
        
        df_trading = load_trading_data(asset)
        
        print(f'  Loaded {len(df_trading)} trading data points')
        
        for seed in random_seeds:
            for window in window_sizes:
                result = run_single_experiment(df_trading, seed, window)
                
                all_results.append({
                    'asset': asset,
                    'random_seed': seed,
                    'window_size': window if window else 'full',
                    'llm_mean_return': result['llm_mean_return'],
                    'rule_mean_return': result['rule_mean_return'],
                    'p_value': result['p_value'],
                    't_statistic': result['t_statistic'],
                    'sample_size': result['sample_size']
                })
                
                print(f'  Asset={asset}, Seed={seed}, Window={window if window else "full"}, '
                      f'LLM={result["llm_mean_return"]:.6f}, p={result["p_value"]:.6f}')
    
    results_df = pd.DataFrame(all_results)
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/robustness.csv', index=False)
    print()
    print(f'✓ Results saved to results/robustness.csv')
    print()
    
    generate_robustness_summary(results_df)


def generate_robustness_summary(results_df):
    """生成稳定性总结"""
    print('=' * 80)
    print('Robustness Summary')
    print('=' * 80)
    print()
    
    asset_groups = results_df.groupby('asset')
    
    all_robust = True
    summary_lines = []
    
    for asset, group in asset_groups:
        significant_count = (group['p_value'] < 0.05).sum()
        total_count = len(group)
        
        if significant_count > total_count * 0.2:
            robustness_label = "Result is NOT robust"
            all_robust = False
        else:
            robustness_label = "Result is robust"
        
        llm_mean_all = group['llm_mean_return'].mean()
        p_value_mean = group['p_value'].mean()
        
        summary_line = (
            f'Asset: {asset:6} | '
            f'Total runs: {total_count:3} | '
            f'Significant (p<0.05): {significant_count:3} | '
            f'Avg LLM Return: {llm_mean_all:.6f} | '
            f'Avg p-value: {p_value_mean:.6f} | '
            f'{robustness_label}'
        )
        
        summary_lines.append(summary_line)
        print(summary_line)
    
    print()
    print('=' * 80)
    if all_robust:
        print('Overall: Result is robust')
    else:
        print('Overall: Result is NOT robust')
    print('=' * 80)
    
    summary_path = 'results/robustness_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('Robustness Test Summary\n')
        f.write('=' * 80 + '\n\n')
        for line in summary_lines:
            f.write(line + '\n')
        f.write('\n')
        f.write('=' * 80 + '\n')
        f.write('Overall: Result is robust\n' if all_robust else 'Overall: Result is NOT robust\n')
        f.write('=' * 80 + '\n')
    
    print(f'✓ Summary saved to {summary_path}')


if __name__ == '__main__':
    run_robustness_test()

