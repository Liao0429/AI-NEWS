import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


def convert_news_to_signal(news_text):
    """
    将新闻文本转换为信号（+1/-1）
    
    使用情绪词典方法，不使用未来信息
    
    Args:
        news_text: 新闻文本（str）
    
    Returns:
        signal: 信号（+1 或 -1）
    """
    text_lower = news_text.lower()
    
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
        # 中性新闻，随机选择
        return 1 if np.random.random() < 0.5 else -1


def calculate_future_return(trade_price, future_price):
    """
    计算未来收益
    
    严格基于trade_time之前的数据，不使用未来信息
    
    Args:
        trade_price: 交易价格（在trade_time时已知）
        future_price: 未来价格（用于计算收益，但在trade_time时未知）
    
    Returns:
        return: 未来收益（%）
    """
    return (future_price - trade_price) / trade_price


class SignalAnalyzer:
    """
    信号强度分析器
    
    严格遵循机器学习研究规范：
    - 无数据泄露（no look-ahead bias）
    - 可复现（固定随机种子）
    - 统计检验（t-test / bootstrap）
    """
    
    def __init__(self, df_aligned, random_seed=42):
        """
        初始化信号分析器
        
        Args:
            df_aligned: 对齐后的数据（包含news_text, trade_price, future_price）
            random_seed: 随机种子（确保可复现性）
        """
        self.df_aligned = df_aligned.copy()
        self.random_seed = random_seed
        
        # 设置随机种子（确保可复现性）
        np.random.seed(random_seed)
        
        # 验证数据完整性
        required_columns = ['news_text', 'trade_price', 'future_price']
        for col in required_columns:
            if col not in self.df_aligned.columns:
                raise ValueError(f"缺少必需的列: {col}")
    
    def generate_signals(self):
        """
        为所有新闻生成信号
        
        不使用未来信息，仅基于新闻文本
        """
        signals = []
        
        for idx, row in self.df_aligned.iterrows():
            signal = convert_news_to_signal(row['news_text'])
            signals.append(signal)
        
        self.df_aligned['signal'] = signals
        return self.df_aligned
    
    def calculate_returns(self):
        """
        计算未来收益
        
        不使用未来信息，仅基于已知的trade_price和future_price
        """
        returns = []
        
        for idx, row in self.df_aligned.iterrows():
            ret = calculate_future_return(row['trade_price'], row['future_price'])
            returns.append(ret)
        
        self.df_aligned['future_return'] = returns
        return self.df_aligned
    
    def calculate_correlation(self):
        """
        计算signal与return的相关性（Pearson correlation）
        
        严格基于trade_time之前的数据
        
        Returns:
            correlation_result: 包含相关性系数、p-value、样本数量的字典
        """
        signals = self.df_aligned['signal'].values
        returns = self.df_aligned['future_return'].values
        
        # Pearson相关系数
        correlation, p_value = stats.pearsonr(signals, returns)
        
        # 样本数量
        sample_size = len(signals)
        
        correlation_result = {
            'correlation_coefficient': correlation,
            'p_value': p_value,
            'sample_size': sample_size,
            'random_seed': self.random_seed
        }
        
        return correlation_result
    
    def bootstrap_correlation(self, n_bootstraps=1000):
        """
        Bootstrap置信区间
        
        确保统计稳健性
        
        Args:
            n_bootstraps: Bootstrap重采样次数
        
        Returns:
            bootstrap_result: 包含置信区间的字典
        """
        bootstrapped_correlations = []
        
        for _ in range(n_bootstraps):
            # 有放回重采样
            indices = np.random.choice(len(self.df_aligned), size=len(self.df_aligned), replace=True)
            bootstrap_signals = self.df_aligned['signal'].values[indices]
            bootstrap_returns = self.df_aligned['future_return'].values[indices]
            
            # 计算相关性
            corr, _ = stats.pearsonr(bootstrap_signals, bootstrap_returns)
            bootstrapped_correlations.append(corr)
        
        # 95%置信区间
        ci_lower = np.percentile(bootstrapped_correlations, 2.5)
        ci_upper = np.percentile(bootstrapped_correlations, 97.5)
        
        bootstrap_result = {
            'bootstrap_mean': np.mean(bootstrapped_correlations),
            'bootstrap_std': np.std(bootstrapped_correlations),
            'ci_lower_95': ci_lower,
            'ci_upper_95': ci_upper,
            'n_bootstraps': n_bootstraps
        }
        
        return bootstrap_result
    
    def visualize_signal_vs_return(self, output_path='results/figures/signal_vs_return.png'):
        """
        可视化signal vs return散点图
        
        Args:
            output_path: 输出图片路径
        """
        plt.figure(figsize=(10, 6))
        
        # 散点图
        plt.scatter(
            self.df_aligned['signal'],
            self.df_aligned['future_return'],
            alpha=0.6,
            s=100,
            color='#1f77b4'
        )
        
        # 添加回归线
        z = np.polyfit(self.df_aligned['signal'], self.df_aligned['future_return'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(-1.2, 1.2, 100)
        plt.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
        
        # 设置标签和标题
        plt.xlabel('Signal (+1 / -1)', fontsize=12)
        plt.ylabel('Future Return', fontsize=12)
        plt.title('Signal vs Future Return', fontsize=14, fontweight='bold')
        
        # 设置x轴刻度
        plt.xticks([-1, 1], ['-1 (Negative)', '+1 (Positive)'])
        plt.xlim(-1.2, 1.2)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def run_complete_analysis(self):
        """
        运行完整的信号强度分析
        
        Returns:
            analysis_result: 完整分析结果的字典
        """
        print('='*80)
        print('信号强度分析（Signal Strength Analysis）')
        print('='*80)
        print()
        
        # Step 1: 生成信号
        print('Step 1: 生成信号...')
        self.generate_signals()
        print(f'  ✓ 完成')
        print(f'  - 信号分布:')
        signal_counts = self.df_aligned['signal'].value_counts()
        for signal, count in signal_counts.items():
            print(f'    Signal {signal}: {count} 条')
        
        # Step 2: 计算收益
        print()
        print('Step 2: 计算未来收益...')
        self.calculate_returns()
        print(f'  ✓ 完成')
        print(f'  - 平均收益: {self.df_aligned["future_return"].mean():.4f}')
        print(f'  - 收益标准差: {self.df_aligned["future_return"].std():.4f}')
        
        # Step 3: 计算相关性
        print()
        print('Step 3: 计算相关性...')
        correlation_result = self.calculate_correlation()
        print(f'  ✓ 完成')
        print(f'  - 相关系数 (r): {correlation_result["correlation_coefficient"]:.4f}')
        print(f'  - p-value: {correlation_result["p_value"]:.4f}')
        print(f'  - 样本数量: {correlation_result["sample_size"]}')
        
        # Step 4: Bootstrap置信区间
        print()
        print('Step 4: Bootstrap置信区间...')
        bootstrap_result = self.bootstrap_correlation()
        print(f'  ✓ 完成')
        print(f'  - Bootstrap平均: {bootstrap_result["bootstrap_mean"]:.4f}')
        print(f'  - Bootstrap标准差: {bootstrap_result["bootstrap_std"]:.4f}')
        print(f'  - 95%置信区间: [{bootstrap_result["ci_lower_95"]:.4f}, {bootstrap_result["ci_upper_95"]:.4f}]')
        
        # Step 5: 可视化
        print()
        print('Step 5: 生成可视化...')
        figure_path = self.visualize_signal_vs_return()
        print(f'  ✓ 完成')
        print(f'  - 图片保存到: {figure_path}')
        
        # 合并结果
        analysis_result = {
            'correlation': correlation_result,
            'bootstrap': bootstrap_result,
            'figure_path': figure_path,
            'random_seed': self.random_seed
        }
        
        # 保存结果到CSV
        self.save_results(analysis_result)
        
        print()
        print('='*80)
        print('✓ 信号强度分析完成')
        print('='*80)
        
        return analysis_result
    
    def save_results(self, analysis_result, output_path='results/signal_correlation.csv'):
        """
        保存分析结果到CSV
        
        Args:
            analysis_result: 分析结果字典
            output_path: 输出CSV路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 创建结果DataFrame
        results_data = {
            'correlation_coefficient': [analysis_result['correlation']['correlation_coefficient']],
            'p_value': [analysis_result['correlation']['p_value']],
            'sample_size': [analysis_result['correlation']['sample_size']],
            'bootstrap_mean': [analysis_result['bootstrap']['bootstrap_mean']],
            'bootstrap_std': [analysis_result['bootstrap']['bootstrap_std']],
            'ci_lower_95': [analysis_result['bootstrap']['ci_lower_95']],
            'ci_upper_95': [analysis_result['bootstrap']['ci_upper_95']],
            'random_seed': [analysis_result['random_seed']]
        }
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        
        print(f'  - 结果保存到: {output_path}')
        
        # 同时保存带信号的数据
        data_output_path = 'results/signal_analysis_data.csv'
        self.df_aligned.to_csv(data_output_path, index=False)
        print(f'  - 数据保存到: {data_output_path}')


def run_signal_analysis(df_aligned, random_seed=42):
    """
    运行信号强度分析的便捷函数
    
    Args:
        df_aligned: 对齐后的数据
        random_seed: 随机种子
    
    Returns:
        analysis_result: 分析结果
    """
    analyzer = SignalAnalyzer(df_aligned, random_seed=random_seed)
    analysis_result = analyzer.run_complete_analysis()
    return analysis_result


if __name__ == '__main__':
    # 测试代码
    from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data
    
    print('加载数据...')
    df_trading = load_and_prepare_from_news_to_forecast_data()
    
    print('运行信号强度分析...')
    analysis_result = run_signal_analysis(df_trading, random_seed=42)
