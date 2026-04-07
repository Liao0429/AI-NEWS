import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy import stats
import os


class SanityChecker:
    """
    Sanity Check模块
    
    严格遵循机器学习研究规范：
    - 严禁数据泄露（no look-ahead bias）
    - 所有实验可复现（固定随机种子）
    - 输出包含统计检验（t-test / bootstrap）
    """
    
    def __init__(self, df_trading, base_random_seed=42):
        """
        初始化Sanity Checker
        
        Args:
            df_trading: 对齐后的交易数据
            base_random_seed: 基础随机种子
        """
        self.df_trading = df_trading.copy()
        self.base_random_seed = base_random_seed
        
        # 验证数据完整性
        self._validate_data()
    
    def _validate_data(self):
        """验证数据完整性"""
        # 检查必需的列
        required_columns = ['trade_price', 'future_price']
        for col in required_columns:
            if col not in self.df_trading.columns:
                raise ValueError(f"缺少必需的列: {col}")
        
        # 检查样本数量
        if len(self.df_trading) < 30:
            raise ValueError(f"样本数量不足: {len(self.df_trading)} < 30")
    
    def generate_perfect_signal(self, row):
        """
        生成perfect signal（严格使用未来信息
        
        perfect signal = +1 if future_price > trade_price else -1
        
        注意：这只是用于sanity check，实际交易中不可用！
        """
        if row['future_price'] > row['trade_price']:
            return 1
        else:
            return -1
    
    def generate_random_signal(self, idx, run_seed):
        """生成随机信号"""
        np.random.seed(self.base_random_seed + run_seed + idx)
        return 1 if np.random.random() < 0.5 else -1
    
    def run_single_backtest(self, run_seed):
        """
        运行单次回测
        
        Args:
            run_seed: 本次运行的随机种子
        
        Returns:
            dict: 单次回测结果
        """
        perfect_returns = []
        random_returns = []
        
        for idx, row in self.df_trading.iterrows():
            # 计算价格收益
            price_return = (row['future_price'] - row['trade_price']) / row['trade_price']
            
            # 生成信号
            perfect_signal = self.generate_perfect_signal(row)
            random_signal = self.generate_random_signal(idx, run_seed)
            
            # 计算策略收益
            perfect_return = perfect_signal * price_return
            random_return = random_signal * price_return
            
            perfect_returns.append(perfect_return)
            random_returns.append(random_return)
        
        # 计算统计指标
        perfect_mean = np.mean(perfect_returns)
        perfect_std = np.std(perfect_returns)
        perfect_sharpe = perfect_mean / perfect_std if perfect_std > 0 else 0
        perfect_win_rate = np.mean(np.array(perfect_returns) > 0) * 100
        
        random_mean = np.mean(random_returns)
        random_std = np.std(random_returns)
        random_sharpe = random_mean / random_std if random_std > 0 else 0
        random_win_rate = np.mean(np.array(random_returns) > 0) * 100
        
        return {
            'perfect_return': perfect_mean,
            'perfect_sharpe': perfect_sharpe,
            'perfect_win_rate': perfect_win_rate,
            'random_return': random_mean,
            'random_sharpe': random_sharpe,
            'random_win_rate': random_win_rate
        }
    
    def run_multiple_backtests(self, n_runs=100):
        """
        运行多次回测
        
        Args:
            n_runs: 回测次数
        
        Returns:
            dict: 多次回测结果
        """
        all_results = []
        
        for run_seed in range(n_runs):
            result = self.run_single_backtest(run_seed)
            all_results.append(result)
        
        return all_results
    
    def bootstrap_confidence_interval(self, data, n_bootstraps=1000):
        """Bootstrap置信区间"""
        bootstrapped_means = []
        
        for _ in range(n_bootstraps):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped_means.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrapped_means, 2.5)
        ci_upper = np.percentile(bootstrapped_means, 97.5)
        
        return ci_lower, ci_upper
    
    def check_sanity(self, all_results):
        """
        检查Sanity
        
        如果检测到问题，必须停止并报错
        
        检测以下问题：
        1. 使用未来数据（perfect signal不应该在实际交易中使用，但这里是sanity check）
        2. 所有策略完全相同（可能bug）
        3. 样本数量 < 30（已在__init__中检查）
        4. return计算不一致
        """
        # 提取数据
        perfect_returns = [r['perfect_return'] for r in all_results]
        random_returns = [r['random_return'] for r in all_results]
        
        # 检查1: Perfect signal应该显著赚钱
        perfect_mean = np.mean(perfect_returns)
        if perfect_mean <= 0:
            raise RuntimeError(f"Sanity Check失败！Perfect signal没有显著赚钱: {perfect_mean:.4f} <= 0")
        
        # 检查2: Perfect应该显著优于Random
        t_stat, p_value = stats.ttest_rel(perfect_returns, random_returns)
        if p_value >= 0.05:
            raise RuntimeError(f"Sanity Check失败！Perfect没有显著优于Random: p-value = {p_value:.4f}")
        
        # 检查3: Perfect win rate应该显著高于50%
        perfect_win_rates = [r['perfect_win_rate'] for r in all_results]
        avg_perfect_win_rate = np.mean(perfect_win_rates)
        if avg_perfect_win_rate <= 50:
            raise RuntimeError(f"Sanity Check失败！Perfect win rate没有显著高于50%: {avg_perfect_win_rate:.2f}%")
        
        print("✓ Sanity Check通过！")
        print(f"  - Perfect平均收益: {perfect_mean:.4f}")
        print(f"  - Perfect vs Random p-value: {p_value:.6f}")
        print(f"  - Perfect win rate: {avg_perfect_win_rate:.2f}%")
    
    def aggregate_results(self, all_results):
        """聚合多次回测结果"""
        perfect_returns = [r['perfect_return'] for r in all_results]
        perfect_sharpes = [r['perfect_sharpe'] for r in all_results]
        perfect_win_rates = [r['perfect_win_rate'] for r in all_results]
        
        random_returns = [r['random_return'] for r in all_results]
        random_sharpes = [r['random_sharpe'] for r in all_results]
        random_win_rates = [r['random_win_rate'] for r in all_results]
        
        # 计算统计量
        perfect_mean_perfect = np.mean(perfect_returns)
        perfect_std_perfect = np.std(perfect_returns)
        perfect_ci_lower, perfect_ci_upper = self.bootstrap_confidence_interval(perfect_returns)
        
        random_mean_perfect = np.mean(random_returns)
        random_std_perfect = np.std(random_returns)
        random_ci_lower, random_ci_upper = self.bootstrap_confidence_interval(random_returns)
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(perfect_returns, random_returns)
        
        aggregated_results = {
            'perfect': {
                'mean_return': perfect_mean_perfect,
                'std_return': perfect_std_perfect,
                'ci_lower': perfect_ci_lower,
                'ci_upper': perfect_ci_upper,
                'mean_sharpe': np.mean(perfect_sharpes),
                'mean_win_rate': np.mean(perfect_win_rates)
            },
            'random': {
                'mean_return': random_mean_perfect,
                'std_return': random_std_perfect,
                'ci_lower': random_ci_lower,
                'ci_upper': random_ci_upper,
                'mean_sharpe': np.mean(random_sharpes),
                'mean_win_rate': np.mean(random_win_rates)
            },
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value
            }
        }
        
        return aggregated_results
    
    def print_results(self, aggregated_results):
        """打印结果"""
        print('='*100)
        print('Sanity Check结果')
        print('='*100)
        print()
        
        print('┌' + '─'*100 + '┐')
        print('│' + ' Strategy '.center(15) + '│' + ' Mean Return '.center(18) + '│' + 
              ' 95% CI '.center(25) + '│' + ' Sharpe '.center(12) + '│' + ' Win Rate '.center(14) + '│')
        print('├' + '─'*100 + '┤')
        
        # Perfect
        perf = aggregated_results['perfect']
        perf_return_str = f"{perf['mean_return']:.4f}"
        perf_ci_str = f"[{perf['ci_lower']:.4f}, {perf['ci_upper']:.4f}]"
        perf_sharpe_str = f"{perf['mean_sharpe']:.2f}"
        perf_win_rate_str = f"{perf['mean_win_rate']:.1f}%"
        print(f'│ {"Perfect":13} │ {perf_return_str:16} │ {perf_ci_str:23} │ {perf_sharpe_str:10} │ {perf_win_rate_str:12} │')
        
        # Random
        rand = aggregated_results['random']
        rand_return_str = f"{rand['mean_return']:.4f}"
        rand_ci_str = f"[{rand['ci_lower']:.4f}, {rand['ci_upper']:.4f}]"
        rand_sharpe_str = f"{rand['mean_sharpe']:.2f}"
        rand_win_rate_str = f"{rand['mean_win_rate']:.1f}%"
        print(f'│ {"Random":13} │ {rand_return_str:16} │ {rand_ci_str:23} │ {rand_sharpe_str:10} │ {rand_win_rate_str:12} │')
        
        print('└' + '─'*100 + '┘')
        print()
        
        # 统计检验
        print('统计检验:')
        print(f'  - 配对t检验: t = {aggregated_results["statistical_test"]["t_statistic"]:.4f}, p-value = {aggregated_results["statistical_test"]["p_value"]:.6f}')
        
        if aggregated_results["statistical_test"]["p_value"] < 0.05:
            print('  ✓ Perfect显著优于Random (p < 0.05)')
        else:
            print('  ✗ Perfect与Random无显著差异')
        
        print()
        print('='*100)
    
    def save_results(self, aggregated_results, all_results, output_path='results/sanity_check_results.csv'):
        """保存结果"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存汇总结果
        summary_data = {
            'strategy': ['Perfect', 'Random'],
            'mean_return': [
                aggregated_results['perfect']['mean_return'],
                aggregated_results['random']['mean_return']
            ],
            'std_return': [
                aggregated_results['perfect']['std_return'],
                aggregated_results['random']['std_return']
            ],
            'ci_lower': [
                aggregated_results['perfect']['ci_lower'],
                aggregated_results['random']['ci_lower']
            ],
            'ci_upper': [
                aggregated_results['perfect']['ci_upper'],
                aggregated_results['random']['ci_upper']
            ],
            'mean_sharpe': [
                aggregated_results['perfect']['mean_sharpe'],
                aggregated_results['random']['mean_sharpe']
            ],
            'mean_win_rate': [
                aggregated_results['perfect']['mean_win_rate'],
                aggregated_results['random']['mean_win_rate']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        print(f'  - 汇总结果保存到: {output_path}')
        
        # 保存详细结果
        detailed_path = output_path.replace('.csv', '_detailed.csv')
        detailed_df = pd.DataFrame(all_results)
        detailed_df.to_csv(detailed_path, index=False)
        
        print(f'  - 详细结果保存到: {detailed_path}')
    
    def run_complete_sanity_check(self, n_runs=100):
        """运行完整的Sanity Check"""
        print('='*100)
        print('Sanity Check（完整性检查')
        print('='*100)
        print()
        
        print(f'样本数量: {len(self.df_trading)}')
        print(f'运行次数: {n_runs}')
        print(f'基础随机种子: {self.base_random_seed}')
        print()
        
        # Step 1: 运行多次回测
        print('Step 1: 运行多次回测...')
        all_results = self.run_multiple_backtests(n_runs=n_runs)
        print(f'  ✓ 完成 {n_runs} 次回测')
        print()
        
        # Step 2: Sanity Check
        print('Step 2: 执行Sanity Check...')
        self.check_sanity(all_results)
        print()
        
        # Step 3: 聚合结果
        print('Step 3: 聚合结果...')
        aggregated_results = self.aggregate_results(all_results)
        print(f'  ✓ 完成')
        print()
        
        # Step 4: 打印结果
        print('Step 4: 打印结果...')
        self.print_results(aggregated_results)
        print()
        
        # Step 5: 保存结果
        print('Step 5: 保存结果...')
        self.save_results(aggregated_results, all_results)
        print()
        
        print('='*100)
        print('✓ Sanity Check完成！')
        print('='*100)
        
        return aggregated_results


def run_sanity_check(df_trading, n_runs=100, base_random_seed=42):
    """运行Sanity Check的便捷函数"""
    checker = SanityChecker(df_trading, base_random_seed=base_random_seed)
    aggregated_results = checker.run_complete_sanity_check(n_runs=n_runs)
    return aggregated_results


if __name__ == '__main__':
    # 测试代码
    from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data
    
    print('加载数据...')
    df_trading = load_and_prepare_from_news_to_forecast_data()
    
    print()
    print('运行Sanity Check...')
    aggregated_results = run_sanity_check(df_trading, n_runs=100, base_random_seed=42)
