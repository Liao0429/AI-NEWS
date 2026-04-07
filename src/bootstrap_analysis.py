import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


class BootstrapAnalyzer:
    """
    Bootstrap统计模块
    
    严格遵循机器学习研究规范：
    - 严禁数据泄露（no look-ahead bias）
    - 所有实验可复现（固定随机种子）
    - 输出包含统计检验（bootstrap）
    """
    
    def __init__(self, strategy_returns, base_random_seed=42):
        """
        初始化Bootstrap分析器
        
        Args:
            strategy_returns: 策略收益字典 {strategy_name: [returns]}
            base_random_seed: 基础随机种子
        """
        self.strategy_returns = strategy_returns.copy()
        self.base_random_seed = base_random_seed
        
        # 验证数据完整性
        self._validate_data()
        
        # 设置随机种子
        np.random.seed(base_random_seed)
    
    def _validate_data(self):
        """验证数据完整性"""
        # 检查样本数量
        for strategy_name, returns in self.strategy_returns.items():
            if len(returns) < 30:
                raise ValueError(f"策略 {strategy_name} 样本数量不足: {len(returns)} < 30")
        
        # 检查是否所有策略完全相同
        all_returns = []
        for returns in self.strategy_returns.values():
            all_returns.append(np.array(returns))
        
        all_same = True
        for i in range(1, len(all_returns)):
            if not np.allclose(all_returns[0], all_returns[i], rtol=1e-3):
                all_same = False
                break
        
        if all_same and len(all_returns) > 1:
            raise RuntimeError("所有策略完全相同！可能存在bug")
        
        print("✓ 数据验证通过！")
    
    def bootstrap_single_strategy(self, returns, n_bootstraps=1000):
        """
        对单个策略进行Bootstrap采样
        
        Args:
            returns: 策略收益列表
            n_bootstraps: Bootstrap采样次数
        
        Returns:
            dict: Bootstrap结果
        """
        bootstrapped_means = []
        
        for _ in range(n_bootstraps):
            # 有放回重采样
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrapped_means.append(np.mean(bootstrap_sample))
        
        # 计算统计量
        mean_bootstrap = np.mean(bootstrapped_means)
        ci_lower = np.percentile(bootstrapped_means, 2.5)
        ci_upper = np.percentile(bootstrapped_means, 97.5)
        std_bootstrap = np.std(bootstrapped_means)
        
        return {
            'mean': mean_bootstrap,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': std_bootstrap,
            'bootstrapped_means': bootstrapped_means
        }
    
    def bootstrap_all_strategies(self, n_bootstraps=1000):
        """
        对所有策略进行Bootstrap采样
        
        Args:
            n_bootstraps: Bootstrap采样次数
        
        Returns:
            dict: 所有策略的Bootstrap结果
        """
        bootstrap_results = {}
        
        for strategy_name, returns in self.strategy_returns.items():
            print(f"  Bootstrap: {strategy_name}...")
            result = self.bootstrap_single_strategy(returns, n_bootstraps=n_bootstraps)
            bootstrap_results[strategy_name] = result
        
        return bootstrap_results
    
    def plot_bootstrap_distributions(self, bootstrap_results, output_dir='results/figures'):
        """
        绘制Bootstrap分布图
        
        Args:
            bootstrap_results: Bootstrap结果
            output_dir: 输出目录
        
        Returns:
            str: 输出图片路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        n_strategies = len(bootstrap_results)
        n_cols = min(3, n_strategies)
        n_rows = (n_strategies + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_strategies == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (strategy_name, result) in enumerate(bootstrap_results.items()):
            ax = axes[idx]
            
            # 绘制直方图
            ax.hist(result['bootstrapped_means'], bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
            
            # 绘制均值和置信区间
            ax.axvline(result['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {result["mean"]:.4f}')
            ax.axvline(result['ci_lower'], color='green', linestyle=':', linewidth=1.5, label=f'95% CI')
            ax.axvline(result['ci_upper'], color='green', linestyle=':', linewidth=1.5)
            
            ax.set_title(f'{strategy_name} (n={len(result["bootstrapped_means"])})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Bootstrapped Mean Return', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_strategies, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'bootstrap_distributions.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
    
    def print_results(self, bootstrap_results):
        """打印Bootstrap结果"""
        print('='*100)
        print('Bootstrap统计结果')
        print('='*100)
        print()
        
        print('┌' + '─'*100 + '┐')
        print('│' + ' Strategy '.center(18) + '│' + ' Mean '.center(15) + '│' + 
              ' CI_lower '.center(15) + '│' + ' CI_upper '.center(15) + '│' + ' Std '.center(15) + '│')
        print('├' + '─'*100 + '┤')
        
        for strategy_name, result in bootstrap_results.items():
            mean_str = f"{result['mean']:.6f}"
            ci_lower_str = f"{result['ci_lower']:.6f}"
            ci_upper_str = f"{result['ci_upper']:.6f}"
            std_str = f"{result['std']:.6f}"
            
            print(f'│ {strategy_name:16} │ {mean_str:13} │ {ci_lower_str:13} │ {ci_upper_str:13} │ {std_str:13} │')
        
        print('└' + '─'*100 + '┘')
        print()
        
        print('='*100)
    
    def save_results(self, bootstrap_results, output_path='results/bootstrap_results.csv'):
        """保存Bootstrap结果"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存汇总结果
        summary_data = []
        for strategy_name, result in bootstrap_results.items():
            summary_data.append({
                'strategy': strategy_name,
                'mean': result['mean'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'std': result['std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        print(f'  - 结果保存到: {output_path}')
        
        return output_path
    
    def run_bootstrap_analysis(self, n_bootstraps=1000):
        """
        运行完整的Bootstrap分析
        
        Args:
            n_bootstraps: Bootstrap采样次数
        
        Returns:
            dict: Bootstrap结果
        """
        print('='*100)
        print('Bootstrap统计分析')
        print('='*100)
        print()
        
        print(f'策略数量: {len(self.strategy_returns)}')
        print(f'策略列表: {", ".join(self.strategy_returns.keys())}')
        print(f'Bootstrap采样次数: {n_bootstraps}')
        print(f'基础随机种子: {self.base_random_seed}')
        print()
        
        # Step 1: Bootstrap采样
        print('Step 1: Bootstrap采样...')
        bootstrap_results = self.bootstrap_all_strategies(n_bootstraps=n_bootstraps)
        print(f'  ✓ 完成')
        print()
        
        # Step 2: 绘制分布图
        print('Step 2: 绘制Bootstrap分布图...')
        figure_path = self.plot_bootstrap_distributions(bootstrap_results)
        print(f'  ✓ 完成')
        print(f'  - 图片保存到: {figure_path}')
        print()
        
        # Step 3: 打印结果
        print('Step 3: 打印结果...')
        self.print_results(bootstrap_results)
        print()
        
        # Step 4: 保存结果
        print('Step 4: 保存结果...')
        self.save_results(bootstrap_results)
        print()
        
        print('='*100)
        print('✓ Bootstrap统计分析完成！')
        print('='*100)
        
        return bootstrap_results


def run_bootstrap_analysis_from_comparison(aggregated_comparison, n_bootstraps=1000, base_random_seed=42):
    """
    从完整策略比较结果中提取策略收益，运行Bootstrap分析
    
    Args:
        aggregated_comparison: 完整策略比较结果
        n_bootstraps: Bootstrap采样次数
        base_random_seed: 基础随机种子
    
    Returns:
        dict: Bootstrap结果
    """
    # 注意：这里需要实际的收益数据
    # 在实际使用中，需要从完整策略比较中获取每次运行的收益
    # 这里提供一个示例接口
    print("⚠️  注意：此函数需要完整策略比较的详细收益数据")
    print("   在实际使用中，请提供每个策略每次运行的收益列表")
    print()
    
    # 返回空结果（实际使用时需要修改）
    return {}


if __name__ == '__main__':
    # 测试代码 - 生成示例数据
    print('生成示例数据...')
    np.random.seed(42)
    
    # 生成示例策略收益
    strategy_returns = {
        'LLM': np.random.normal(0.001, 0.003, size=100).tolist(),
        'ML': np.random.normal(0.0008, 0.0035, size=100).tolist(),
        'Rule': np.random.normal(0.0005, 0.004, size=100).tolist(),
        'Random': np.random.normal(0.0002, 0.0045, size=100).tolist(),
        'Momentum': np.random.normal(0.0015, 0.0025, size=100).tolist(),
        'MeanReversion': np.random.normal(-0.0005, 0.005, size=100).tolist()
    }
    
    print()
    print('运行Bootstrap分析...')
    analyzer = BootstrapAnalyzer(strategy_returns, base_random_seed=42)
    bootstrap_results = analyzer.run_bootstrap_analysis(n_bootstraps=1000)
