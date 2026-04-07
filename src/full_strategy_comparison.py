import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy import stats
import os


class StrategyBase:
    """策略基类"""
    
    def generate_signal(self, idx, row, price_history, run_seed):
        """
        生成信号，返回 {+1, -1}
        
        Args:
            idx: 当前索引
            row: 当前行数据
            price_history: 历史价格数据（trade_time之前的数据）
            run_seed: 本次运行的随机种子
        
        Returns:
            signal: {+1, -1}
        """
        raise NotImplementedError


class LLMStrategy(StrategyBase):
    """LLM策略：基于新闻关键词"""
    
    def generate_signal(self, idx, row, price_history, run_seed):
        text_lower = row['news_text'].lower()
        
        positive_words = [
            'rise', 'gain', 'positive', 'strong', 'beat', 'up', 'growth',
            'increase', 'success', 'bull', 'soar', 'surge', 'rally', 'jump'
        ]
        negative_words = [
            'fall', 'drop', 'negative', 'weak', 'miss', 'down', 'decline',
            'decrease', 'failure', 'bear', 'crash', 'plunge', 'slump', 'tumble'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return -1
        else:
            np.random.seed(42 + run_seed + idx)
            return 1 if np.random.random() < 0.5 else -1


class MLStrategy(StrategyBase):
    """ML策略：基于关键词的逻辑回归风格"""
    
    def generate_signal(self, idx, row, price_history, run_seed):
        text_lower = row['news_text'].lower()
        
        features = {
            'has_rise': int('rise' in text_lower),
            'has_fall': int('fall' in text_lower),
            'has_gain': int('gain' in text_lower),
            'has_drop': int('drop' in text_lower),
            'has_positive': int('positive' in text_lower),
            'has_negative': int('negative' in text_lower),
            'has_strong': int('strong' in text_lower),
            'has_weak': int('weak' in text_lower),
        }
        
        score = (
            0.3 * features['has_rise'] +
            0.25 * features['has_gain'] +
            0.2 * features['has_positive'] +
            0.15 * features['has_strong'] -
            0.3 * features['has_fall'] -
            0.25 * features['has_drop'] -
            0.2 * features['has_negative'] -
            0.15 * features['has_weak']
        )
        
        np.random.seed(42 + run_seed + idx)
        score += np.random.uniform(-0.2, 0.2)
        
        if score > 0:
            return 1
        else:
            return -1


class RuleStrategy(StrategyBase):
    """Rule策略：简单规则"""
    
    def generate_signal(self, idx, row, price_history, run_seed):
        np.random.seed(42 + run_seed + idx)
        return 1 if np.random.random() < 0.5 else -1


class RandomStrategy(StrategyBase):
    """Random策略：完全随机"""
    
    def generate_signal(self, idx, row, price_history, run_seed):
        np.random.seed(42 + run_seed + idx)
        return 1 if np.random.random() < 0.5 else -1


class MomentumStrategy(StrategyBase):
    """Momentum策略：signal = sign(price_t - price_t-1)"""
    
    def generate_signal(self, idx, row, price_history, run_seed):
        if len(price_history) < 2:
            np.random.seed(42 + run_seed + idx)
            return 1 if np.random.random() < 0.5 else -1
        
        price_t = price_history[-1]
        price_t_minus_1 = price_history[-2]
        
        if price_t > price_t_minus_1:
            return 1
        elif price_t < price_t_minus_1:
            return -1
        else:
            np.random.seed(42 + run_seed + idx)
            return 1 if np.random.random() < 0.5 else -1


class MeanReversionStrategy(StrategyBase):
    """Mean Reversion策略：signal = -sign(price_t - moving_average)"""
    
    def __init__(self, ma_window=5):
        self.ma_window = ma_window
    
    def generate_signal(self, idx, row, price_history, run_seed):
        if len(price_history) < self.ma_window:
            np.random.seed(42 + run_seed + idx)
            return 1 if np.random.random() < 0.5 else -1
        
        price_t = price_history[-1]
        moving_average = np.mean(price_history[-self.ma_window:])
        
        if price_t > moving_average:
            return -1
        elif price_t < moving_average:
            return 1
        else:
            np.random.seed(42 + run_seed + idx)
            return 1 if np.random.random() < 0.5 else -1


class FullStrategyComparison:
    """
    完整策略比较模块
    
    严格遵循机器学习研究规范：
    - 严禁数据泄露（no look-ahead bias）
    - 所有实验可复现（固定随机种子）
    - 输出包含统计检验（t-test / bootstrap）
    """
    
    def __init__(self, df_trading, base_random_seed=42):
        """
        初始化策略比较器
        
        Args:
            df_trading: 对齐后的交易数据
            base_random_seed: 基础随机种子
        """
        self.df_trading = df_trading.copy()
        self.base_random_seed = base_random_seed
        
        # 验证数据完整性
        self._validate_data()
        
        # 初始化策略
        self.strategies = {
            'LLM': LLMStrategy(),
            'ML': MLStrategy(),
            'Rule': RuleStrategy(),
            'Random': RandomStrategy(),
            'Momentum': MomentumStrategy(),
            'MeanReversion': MeanReversionStrategy(ma_window=5)
        }
    
    def _validate_data(self):
        """验证数据完整性"""
        # 检查必需的列
        required_columns = ['news_text', 'trade_price', 'future_price']
        for col in required_columns:
            if col not in self.df_trading.columns:
                raise ValueError(f"缺少必需的列: {col}")
        
        # 检查样本数量
        if len(self.df_trading) < 30:
            raise ValueError(f"样本数量不足: {len(self.df_trading)} < 30")
    
    def run_single_backtest(self, run_seed):
        """
        运行单次回测
        
        Args:
            run_seed: 本次运行的随机种子
        
        Returns:
            dict: 单次回测结果
        """
        strategy_returns = {name: [] for name in self.strategies.keys()}
        price_history = []
        
        for idx, row in self.df_trading.iterrows():
            # 计算价格收益
            price_return = (row['future_price'] - row['trade_price']) / row['trade_price']
            
            # 更新历史价格（仅使用trade_time之前的数据！）
            price_history.append(row['trade_price'])
            
            # 为每个策略生成信号
            for strategy_name, strategy in self.strategies.items():
                signal = strategy.generate_signal(idx, row, price_history.copy(), run_seed)
                strategy_return = signal * price_return
                strategy_returns[strategy_name].append(strategy_return)
        
        # 计算统计指标
        single_result = {}
        for strategy_name, returns in strategy_returns.items():
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            win_rate = np.mean(np.array(returns) > 0) * 100
            
            single_result[strategy_name] = {
                'mean_return': mean_ret,
                'std_return': std_ret,
                'sharpe': sharpe,
                'win_rate': win_rate
            }
        
        return single_result
    
    def run_multiple_backtests(self, n_runs=100):
        """
        运行多次回测
        
        Args:
            n_runs: 回测次数
        
        Returns:
            dict: 多次回测结果
        """
        all_results = {name: [] for name in self.strategies.keys()}
        
        for run_seed in range(n_runs):
            single_result = self.run_single_backtest(run_seed)
            
            for strategy_name in self.strategies.keys():
                all_results[strategy_name].append(single_result[strategy_name])
        
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
    
    def check_validity(self, all_results):
        """
        检查有效性
        
        如果检测到问题，必须停止并报错
        """
        # 检查样本数量
        if len(self.df_trading) < 30:
            raise RuntimeError(f"样本数量不足: {len(self.df_trading)} < 30")
        
        # 检查是否所有策略完全相同
        all_returns = []
        for strategy_name in self.strategies.keys():
            returns = [r['mean_return'] for r in all_results[strategy_name]]
            all_returns.append(np.array(returns))
        
        # 检查所有策略是否几乎相同
        all_same = True
        for i in range(1, len(all_returns)):
            if not np.allclose(all_returns[0], all_returns[i], rtol=1e-3):
                all_same = False
                break
        
        if all_same:
            raise RuntimeError("所有策略完全相同！可能存在bug")
        
        print("✓ 有效性检查通过！")
    
    def aggregate_results(self, all_results):
        """聚合多次回测结果"""
        aggregated_results = {}
        
        for strategy_name in self.strategies.keys():
            returns = [r['mean_return'] for r in all_results[strategy_name]]
            sharpes = [r['sharpe'] for r in all_results[strategy_name]]
            win_rates = [r['win_rate'] for r in all_results[strategy_name]]
            
            # 计算统计量
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            ci_lower, ci_upper = self.bootstrap_confidence_interval(returns)
            
            # 计算vs Random的p-value
            random_returns = [r['mean_return'] for r in all_results['Random']]
            t_stat, p_value = stats.ttest_rel(returns, random_returns)
            
            aggregated_results[strategy_name] = {
                'mean_return': mean_return,
                'std_return': std_return,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean_sharpe': np.mean(sharpes),
                'mean_win_rate': np.mean(win_rates),
                'p_value_vs_random': p_value
            }
        
        return aggregated_results
    
    def print_results(self, aggregated_results):
        """打印结果"""
        print('='*120)
        print('完整策略比较结果')
        print('='*120)
        print()
        
        print('┌' + '─'*120 + '┐')
        print('│' + ' Strategy '.center(15) + '│' + ' Mean Return '.center(15) + '│' + 
              ' 95% CI '.center(25) + '│' + ' Sharpe '.center(12) + '│' + 
              ' Win Rate '.center(12) + '│' + ' p-value (vs Random) '.center(25) + '│')
        print('├' + '─'*120 + '┤')
        
        for strategy_name in ['LLM', 'ML', 'Rule', 'Momentum', 'MeanReversion', 'Random']:
            if strategy_name in aggregated_results:
                res = aggregated_results[strategy_name]
                return_str = f"{res['mean_return']:.4f}"
                ci_str = f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]"
                sharpe_str = f"{res['mean_sharpe']:.2f}"
                win_rate_str = f"{res['mean_win_rate']:.1f}%"
                
                if strategy_name == 'Random':
                    p_value_str = '-'
                else:
                    p_value = res['p_value_vs_random']
                    p_value_str = f"{p_value:.6f}"
                    if p_value < 0.05:
                        p_value_str += ' *'
                
                print(f'│ {strategy_name:13} │ {return_str:13} │ {ci_str:23} │ {sharpe_str:10} │ {win_rate_str:10} │ {p_value_str:23} │')
        
        print('└' + '─'*120 + '┘')
        print()
        
        print('='*120)
    
    def save_results(self, aggregated_results, output_path='results/full_strategy_comparison.csv'):
        """保存结果"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存汇总结果
        summary_data = []
        for strategy_name in ['LLM', 'ML', 'Rule', 'Momentum', 'MeanReversion', 'Random']:
            if strategy_name in aggregated_results:
                res = aggregated_results[strategy_name]
                summary_data.append({
                    'strategy': strategy_name,
                    'mean_return': res['mean_return'],
                    'std_return': res['std_return'],
                    'ci_lower': res['ci_lower'],
                    'ci_upper': res['ci_upper'],
                    'mean_sharpe': res['mean_sharpe'],
                    'mean_win_rate': res['mean_win_rate'],
                    'p_value_vs_random': res.get('p_value_vs_random', '')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        print(f'  - 结果保存到: {output_path}')
    
    def run_complete_comparison(self, n_runs=100):
        """运行完整的策略比较"""
        print('='*120)
        print('完整策略比较（Full Strategy Comparison）')
        print('='*120)
        print()
        
        print(f'样本数量: {len(self.df_trading)}')
        print(f'运行次数: {n_runs}')
        print(f'基础随机种子: {self.base_random_seed}')
        print(f'策略数量: {len(self.strategies)}')
        print(f'策略列表: {", ".join(self.strategies.keys())}')
        print()
        
        # Step 1: 运行多次回测
        print('Step 1: 运行多次回测...')
        all_results = self.run_multiple_backtests(n_runs=n_runs)
        print(f'  ✓ 完成 {n_runs} 次回测')
        print()
        
        # Step 2: 有效性检查
        print('Step 2: 有效性检查...')
        self.check_validity(all_results)
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
        self.save_results(aggregated_results)
        print()
        
        print('='*120)
        print('✓ 完整策略比较完成！')
        print('='*120)
        
        return aggregated_results


def run_full_strategy_comparison(df_trading, n_runs=100, base_random_seed=42):
    """运行完整策略比较的便捷函数"""
    comparator = FullStrategyComparison(df_trading, base_random_seed=base_random_seed)
    aggregated_results = comparator.run_complete_comparison(n_runs=n_runs)
    return aggregated_results


if __name__ == '__main__':
    # 测试代码
    from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data
    
    print('加载数据...')
    df_trading = load_and_prepare_from_news_to_forecast_data()
    
    print()
    print('运行完整策略比较...')
    aggregated_results = run_full_strategy_comparison(df_trading, n_runs=100, base_random_seed=42)
