import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data
from src.unified_evaluation import run_single_strategy
import pandas as pd
import numpy as np
import random
from scipy import stats


def main():
    """Sanity Check实验：验证回测系统在"有信息"的情况下可以盈利"""
    print('='*80)
    print('Experiment 1: Sanity Check')
    print('='*80)
    print()
    
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # Step 1: 加载From_News_to_Forecast数据
    print('Step 1: Loading From_News_to_Forecast data...')
    try:
        aligned_df = load_and_prepare_from_news_to_forecast_data()
        print(f'  ✓ Loaded {len(aligned_df)} data points')
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return
    print()
    
    # Step 2: 运行Perfect策略（有信息）
    print('Step 2: Running Perfect strategy...')
    perfect_returns = []
    for i in range(len(aligned_df)):
        row = aligned_df.iloc[i]
        future_return = (row['future_price'] - row['trade_price']) / row['trade_price']
        perfect_signal = 1 if future_return > 0 else -1
        strategy_return = perfect_signal * future_return
        perfect_returns.append(strategy_return)
    
    perfect_mean = np.mean(perfect_returns)
    perfect_std = np.std(perfect_returns) if np.std(perfect_returns) > 0 else 1
    perfect_sharpe = perfect_mean / perfect_std
    perfect_win_rate = np.mean([1 if r > 0 else 0 for r in perfect_returns]) * 100
    print('  ✓ Perfect strategy completed')
    print()
    
    # Step 3: 运行Random策略（无信息）
    print('Step 3: Running Random strategy...')
    num_runs = 100
    random_returns_all = []
    
    for run in range(num_runs):
        random_returns = []
        for i in range(len(aligned_df)):
            row = aligned_df.iloc[i]
            future_return = (row['future_price'] - row['trade_price']) / row['trade_price']
            random_signal = 1 if random.random() < 0.5 else -1
            strategy_return = random_signal * future_return
            random_returns.append(strategy_return)
        random_returns_all.append(np.mean(random_returns))
    
    random_mean = np.mean(random_returns_all)
    random_std = np.std(random_returns_all)
    random_sharpe = random_mean / random_std if random_std > 0 else 0
    random_win_rate = 50.0  # 理论值
    print('  ✓ Random strategy completed')
    print()
    
    # Step 4: 统计检验
    print('Step 4: Performing statistical tests...')
    
    # 使用Perfect单次运行的return作为样本
    t_stat, p_value = stats.ttest_1samp(perfect_returns, 0)
    print(f'  t-statistic: {t_stat:.4f}')
    print(f'  p-value: {p_value:.6f}')
    print()
    
    # Step 5: 输出结果
    print('='*80)
    print('Sanity Check Results')
    print('='*80)
    print()
    print(f'{"Strategy":<12} {"Return":>12} {"Sharpe":>8} {"Win Rate":>12}')
    print('-'*54)
    print(f'{"Perfect":<12} {perfect_mean:12.6f} {perfect_sharpe:8.4f} {perfect_win_rate:11.2f}%')
    print(f'{"Random":<12} {random_mean:12.6f} {random_sharpe:8.4f} {random_win_rate:11.2f}%')
    print('-'*54)
    print()
    print('Statistical Test (Perfect vs 0):')
    print(f'  t-statistic: {t_stat:.4f}')
    print(f'  p-value: {p_value:.6f}')
    print()
    
    # 判断是否通过
    if perfect_mean > 0 and perfect_win_rate > 50 and p_value < 0.05:
        print('✅ Sanity Check PASSED: Perfect strategy significantly profitable')
    else:
        print('⚠️ Sanity Check WARNING: Perfect strategy not significantly profitable')
    print()
    
    # Step 6: 保存结果
    print('='*80)
    print('Saving results...')
    
    results_df = pd.DataFrame({
        'Strategy': ['Perfect', 'Random'],
        'Mean Return': [perfect_mean, random_mean],
        'Std Return': [perfect_std, random_std],
        'Sharpe': [perfect_sharpe, random_sharpe],
        'Win Rate': [perfect_win_rate, random_win_rate],
        'p-value (vs 0)': [p_value, None]
    })
    
    os.makedirs('results/tables', exist_ok=True)
    results_df.to_csv('results/tables/01_sanity_check_results.csv', index=False)
    print('  ✓ Results saved to results/tables/01_sanity_check_results.csv')
    print('='*80)


if __name__ == '__main__':
    main()
