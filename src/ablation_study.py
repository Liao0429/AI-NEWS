import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
from src.strategies import KeywordStrategy, HashStrategy, calculate_stats, calculate_percentage_return
from src.stats import perform_statistical_test


class AblationStudy:
    """
    Ablation Study模块（使用统一策略接口）
    
    目标：验证策略是否真的使用了新闻信息
    
    严格遵循机器学习研究规范：
    - 严禁数据泄露（no look-ahead bias）
    - 所有实验可复现（固定随机种子）
    - 输出包含统计检验（t-test / bootstrap）
    """
    
    def __init__(self, df_trading, base_random_seed=42):
        """
        初始化Ablation Study
        
        Args:
            df_trading: 对齐后的交易数据
            base_random_seed: 基础随机种子
        """
        self.df_trading = df_trading.copy()
        self.base_random_seed = base_random_seed
        
        self._validate_data()
        
        np.random.seed(base_random_seed)
        random.seed(base_random_seed)
    
    def _validate_data(self):
        """验证数据完整性"""
        required_columns = ['news_text', 'trade_price', 'future_price']
        for col in required_columns:
            if col not in self.df_trading.columns:
                raise ValueError(f"缺少必需的列: {col}")
        
        if len(self.df_trading) < 30:
            raise ValueError(f"样本数量不足: {len(self.df_trading)} < 30")
        
        print("✓ 数据验证通过！")
    
    def shuffle_news_texts(self, run_seed):
        """
        随机打乱新闻文本
        
        Args:
            run_seed: 本次运行的随机种子
        
        Returns:
            list: 打乱后的新闻文本列表
        """
        np.random.seed(self.base_random_seed + run_seed)
        shuffled_indices = np.random.permutation(len(self.df_trading))
        shuffled_news = self.df_trading['news_text'].values[shuffled_indices].tolist()
        return shuffled_news
    
    def run_single_backtest(self, run_seed, strategy, use_shuffled_news=False):
        """
        运行单次回测
        
        Args:
            run_seed: 本次运行的随机种子
            strategy: 策略实例
            use_shuffled_news: 是否使用打乱的新闻
        
        Returns:
            dict: 单次回测结果
        """
        # 不再需要设置随机种子，因为策略都是确定性的
        
        if use_shuffled_news:
            news_texts = self.shuffle_news_texts(run_seed)
        else:
            news_texts = self.df_trading['news_text'].values.tolist()
        
        returns = []
        
        for idx, row in self.df_trading.iterrows():
            if row['trade_price'] == 0:
                returns.append(0.0)
                continue
            
            news_text = news_texts[idx]
            signal = strategy.generate_signal(news_text)
            
            pct_return = calculate_percentage_return(row['future_price'], row['trade_price'])
            strategy_return = signal * pct_return
            returns.append(strategy_return)
        
        return calculate_stats(returns)
    
    def run_multiple_backtests(self, n_runs=100):
        """
        运行多次回测
        
        Args:
            n_runs: 回测次数
        
        Returns:
            dict: 多次回测结果
        """
        all_real_results = []
        all_shuffled_results = []
        all_hash_results = []
        
        keyword_strategy = KeywordStrategy(random_seed=self.base_random_seed)
        hash_strategy = HashStrategy(random_seed=self.base_random_seed)
        
        for run_seed in range(n_runs):
            real_result = self.run_single_backtest(run_seed, keyword_strategy, use_shuffled_news=False)
            shuffled_result = self.run_single_backtest(run_seed, keyword_strategy, use_shuffled_news=True)
            hash_result = self.run_single_backtest(run_seed, hash_strategy, use_shuffled_news=False)
            
            all_real_results.append(real_result)
            all_shuffled_results.append(shuffled_result)
            all_hash_results.append(hash_result)
        
        return {
            'real': all_real_results,
            'shuffled': all_shuffled_results,
            'hash': all_hash_results
        }
    
    def bootstrap_confidence_interval(self, data, n_bootstraps=1000):
        """Bootstrap置信区间"""
        bootstrapped_means = []
        
        np.random.seed(self.base_random_seed)
        for _ in range(n_bootstraps):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_means.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrapped_means, 2.5)
        ci_upper = np.percentile(bootstrapped_means, 97.5)
        
        return ci_lower, ci_upper
    
    def check_validity(self, all_results):
        """
        检查有效性
        
        如果检测到问题，必须停止并报错
        """
        real_returns = [r['mean'] for r in all_results['real']]
        shuffled_returns = [r['mean'] for r in all_results['shuffled']]
        hash_returns = [r['mean'] for r in all_results['hash']]
        
        if len(real_returns) < 10 or len(shuffled_returns) < 10 or len(hash_returns) < 10:
            raise RuntimeError("运行次数不足")
        
        print("✓ 有效性检查通过！")
    
    def aggregate_results(self, all_results):
        """聚合多次回测结果"""
        real_returns = [r['mean'] for r in all_results['real']]
        real_sharpes = [r['sharpe'] for r in all_results['real']]
        real_win_rates = [r['win_rate'] for r in all_results['real']]
        
        shuffled_returns = [r['mean'] for r in all_results['shuffled']]
        shuffled_sharpes = [r['sharpe'] for r in all_results['shuffled']]
        shuffled_win_rates = [r['win_rate'] for r in all_results['shuffled']]
        
        hash_returns = [r['mean'] for r in all_results['hash']]
        hash_sharpes = [r['sharpe'] for r in all_results['hash']]
        hash_win_rates = [r['win_rate'] for r in all_results['hash']]
        
        real_mean = np.mean(real_returns)
        real_std = np.std(real_returns)
        real_ci_lower, real_ci_upper = self.bootstrap_confidence_interval(real_returns)
        
        shuffled_mean = np.mean(shuffled_returns)
        shuffled_std = np.std(shuffled_returns)
        shuffled_ci_lower, shuffled_ci_upper = self.bootstrap_confidence_interval(shuffled_returns)
        
        hash_mean = np.mean(hash_returns)
        hash_std = np.std(hash_returns)
        hash_ci_lower, hash_ci_upper = self.bootstrap_confidence_interval(hash_returns)
        
        test_stat, p_value, test_type = perform_statistical_test(real_returns, shuffled_returns)
        
        if p_value > 0.05:
            print()
            print("="*100)
            print("⚠️  警告：Keyword did not extract meaningful information")
            print(f"   Real ≈ Shuffled (p-value = {p_value:.6f})")
            print("="*100)
            print()
        else:
            print()
            print("="*100)
            print("✅ 结论：Keyword extracted useful signal")
            print(f"   Real > Shuffled (p-value = {p_value:.6f})")
            print("="*100)
            print()
        
        aggregated_results = {
            'real': {
                'mean_return': real_mean,
                'std_return': real_std,
                'ci_lower': real_ci_lower,
                'ci_upper': real_ci_upper,
                'mean_sharpe': np.mean(real_sharpes),
                'mean_win_rate': np.mean(real_win_rates)
            },
            'shuffled': {
                'mean_return': shuffled_mean,
                'std_return': shuffled_std,
                'ci_lower': shuffled_ci_lower,
                'ci_upper': shuffled_ci_upper,
                'mean_sharpe': np.mean(shuffled_sharpes),
                'mean_win_rate': np.mean(shuffled_win_rates)
            },
            'hash': {
                'mean_return': hash_mean,
                'std_return': hash_std,
                'ci_lower': hash_ci_lower,
                'ci_upper': hash_ci_upper,
                'mean_sharpe': np.mean(hash_sharpes),
                'mean_win_rate': np.mean(hash_win_rates)
            },
            'statistical_test': {
                'test_type': test_type,
                'test_statistic': test_stat,
                'p_value': p_value
            }
        }
        
        return aggregated_results
    
    def print_results(self, aggregated_results):
        """打印Ablation Study结果"""
        print('='*100)
        print('Ablation Study结果')
        print('='*100)
        print()
        
        print('┌' + '─'*100 + '┐')
        print('│' + ' Setup '.center(15) + '│' + ' Mean Return '.center(15) + '│' + 
              ' 95% CI '.center(25) + '│' + ' Sharpe '.center(12) + '│' + ' Win Rate '.center(14) + '│')
        print('├' + '─'*100 + '┤')
        
        real = aggregated_results['real']
        real_return_str = f"{real['mean_return']:.6f}"
        real_ci_str = f"[{real['ci_lower']:.6f}, {real['ci_upper']:.6f}]"
        real_sharpe_str = f"{real['mean_sharpe']:.2f}"
        real_win_rate_str = f"{real['mean_win_rate']:.1f}%"
        print(f'│ {"Real News":13} │ {real_return_str:13} │ {real_ci_str:23} │ {real_sharpe_str:10} │ {real_win_rate_str:12} │')
        
        shuffled = aggregated_results['shuffled']
        shuffled_return_str = f"{shuffled['mean_return']:.6f}"
        shuffled_ci_str = f"[{shuffled['ci_lower']:.6f}, {shuffled['ci_upper']:.6f}]"
        shuffled_sharpe_str = f"{shuffled['mean_sharpe']:.2f}"
        shuffled_win_rate_str = f"{shuffled['mean_win_rate']:.1f}%"
        print(f'│ {"Shuffled":13} │ {shuffled_return_str:13} │ {shuffled_ci_str:23} │ {shuffled_sharpe_str:10} │ {shuffled_win_rate_str:12} │')
        
        hash_data = aggregated_results['hash']
        hash_return_str = f"{hash_data['mean_return']:.6f}"
        hash_ci_str = f"[{hash_data['ci_lower']:.6f}, {hash_data['ci_upper']:.6f}]"
        hash_sharpe_str = f"{hash_data['mean_sharpe']:.2f}"
        hash_win_rate_str = f"{hash_data['mean_win_rate']:.1f}%"
        print(f'│ {"Hash":13} │ {hash_return_str:13} │ {hash_ci_str:23} │ {hash_sharpe_str:10} │ {hash_win_rate_str:12} │')
        
        print('└' + '─'*100 + '┘')
        print()
        
        print('统计检验 (Real vs Shuffled):')
        print(f'  - 检验类型: {aggregated_results["statistical_test"]["test_type"]}')
        print(f'  - 检验统计量: {aggregated_results["statistical_test"]["test_statistic"]:.4f}')
        print(f'  - p-value: {aggregated_results["statistical_test"]["p_value"]:.6f}')
        
        if aggregated_results["statistical_test"]["p_value"] < 0.05:
            print('  ✓ Real显著优于Shuffled (p < 0.05)')
        else:
            print('  ✗ Real与Shuffled无显著差异')
        
        print()
        print('='*100)
    
    def save_results(self, aggregated_results, output_path='results/ablation_results.csv'):
        """保存Ablation Study结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        summary_data = []
        
        real = aggregated_results['real']
        summary_data.append({
            'Strategy': 'Keyword (Real News)',
            'Return': real['mean_return'],
            'Sharpe': real['mean_sharpe'],
            'WinRate': real['mean_win_rate'],
            'p-value': ''
        })
        
        shuffled = aggregated_results['shuffled']
        summary_data.append({
            'Strategy': 'Keyword (Shuffled News)',
            'Return': shuffled['mean_return'],
            'Sharpe': shuffled['mean_sharpe'],
            'WinRate': shuffled['mean_win_rate'],
            'p-value': aggregated_results['statistical_test']['p_value']
        })
        
        hash_data = aggregated_results['hash']
        summary_data.append({
            'Strategy': 'Hash (Hash Strategy)',
            'Return': hash_data['mean_return'],
            'Sharpe': hash_data['mean_sharpe'],
            'WinRate': hash_data['mean_win_rate'],
            'p-value': ''
        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        print(f'  - 结果保存到: {output_path}')
        
        return output_path
    
    def run_ablation_study(self, n_runs=100):
        """
        运行完整的Ablation Study
        
        Args:
            n_runs: 回测次数
        
        Returns:
            dict: Ablation Study结果
        """
        print('='*100)
        print('Ablation Study（消融研究）')
        print('目标：验证策略是否真的使用了新闻信息')
        print('='*100)
        print()
        
        print(f'样本数量: {len(self.df_trading)}')
        print(f'运行次数: {n_runs}')
        print(f'基础随机种子: {self.base_random_seed}')
        print()
        
        print('Step 1: 运行多次回测...')
        all_results = self.run_multiple_backtests(n_runs=n_runs)
        print(f'  ✓ 完成 {n_runs} 次回测 (Real + Shuffled)')
        print()
        
        print('Step 2: 有效性检查...')
        self.check_validity(all_results)
        print()
        
        print('Step 3: 聚合结果...')
        aggregated_results = self.aggregate_results(all_results)
        print(f'  ✓ 完成')
        print()
        
        print('Step 4: 打印结果...')
        self.print_results(aggregated_results)
        print()
        
        print('Step 5: 保存结果...')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, 'results', 'ablation_results.csv')
        self.save_results(aggregated_results, output_path)
        print()
        
        print('='*100)
        print('✓ Ablation Study完成！')
        print('='*100)
        
        return aggregated_results


def run_ablation_study(df_trading, n_runs=100, base_random_seed=42):
    """运行Ablation Study的便捷函数"""
    study = AblationStudy(df_trading, base_random_seed=base_random_seed)
    aggregated_results = study.run_ablation_study(n_runs=n_runs)
    return aggregated_results


if __name__ == '__main__':
    from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data
    
    print('加载数据...')
    df_trading = load_and_prepare_from_news_to_forecast_data()
    
    print()
    print('运行Ablation Study...')
    aggregated_results = run_ablation_study(df_trading, n_runs=100, base_random_seed=42)
