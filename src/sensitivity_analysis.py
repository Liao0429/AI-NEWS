import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.strategies import calculate_percentage_return
from src.strategy_momentum import MomentumStrategy
from src.strategy_meanreversion import MeanReversionStrategy


class SensitivityAnalyzer:
    """敏感性分析模块
    
    功能：
    - 测试关键参数对策略表现的影响
    - 输出敏感性曲线
    """
    
    def __init__(self, aligned_df: pd.DataFrame):
        """
        初始化敏感性分析器
        
        Args:
            aligned_df: 对齐的数据框
        """
        self.aligned_df = aligned_df.copy()
        # 计算真实收益率
        self.aligned_df['true_return'] = self.aligned_df.apply(
            lambda row: calculate_percentage_return(row['future_price'], row['trade_price']),
            axis=1
        )
    
    def analyze_momentum_window(self, window_sizes: List[int] = None) -> Dict[str, List[float]]:
        """
        分析Momentum策略的时间窗口参数敏感性
        
        Args:
            window_sizes: 时间窗口大小列表
            
        Returns:
            results: 不同窗口大小的策略表现
        """
        if window_sizes is None:
            window_sizes = [1, 3, 5, 10, 15, 20, 30]
        
        results = {
            'window_size': window_sizes,
            'mean_return': [],
            'std_return': [],
            'sharpe': [],
            'win_rate': []
        }
        
        for window in window_sizes:
            strategy = MomentumStrategy(time_window=window)
            
            # 生成信号（使用run方法）
            signals, _ = strategy.run(self.aligned_df)
            
            # 计算收益率
            returns = [signal * true_return for signal, true_return in 
                      zip(signals, self.aligned_df['true_return'].tolist())]
            
            # 计算统计指标
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 1e-8
                sharpe = mean_return / (std_return if std_return > 1e-8 else 1e-8)
                win_rate = np.mean([1 if r > 0 else 0 for r in returns]) * 100
            else:
                mean_return = 0.0
                std_return = 0.0
                sharpe = 0.0
                win_rate = 0.0
            
            results['mean_return'].append(mean_return)
            results['std_return'].append(std_return)
            results['sharpe'].append(sharpe)
            results['win_rate'].append(win_rate)
        
        return results
    
    def analyze_mean_reversion_window(self, window_sizes: List[int] = None) -> Dict[str, List[float]]:
        """
        分析MeanReversion策略的时间窗口参数敏感性
        
        Args:
            window_sizes: 时间窗口大小列表
            
        Returns:
            results: 不同窗口大小的策略表现
        """
        if window_sizes is None:
            window_sizes = [5, 10, 15, 20, 30, 40, 50]
        
        results = {
            'window_size': window_sizes,
            'mean_return': [],
            'std_return': [],
            'sharpe': [],
            'win_rate': []
        }
        
        for window in window_sizes:
            strategy = MeanReversionStrategy(time_window=window)
            
            # 生成信号（使用run方法）
            signals, _ = strategy.run(self.aligned_df)
            
            # 计算收益率
            returns = [signal * true_return for signal, true_return in 
                      zip(signals, self.aligned_df['true_return'].tolist())]
            
            # 计算统计指标
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 1e-8
                sharpe = mean_return / (std_return if std_return > 1e-8 else 1e-8)
                win_rate = np.mean([1 if r > 0 else 0 for r in returns]) * 100
            else:
                mean_return = 0.0
                std_return = 0.0
                sharpe = 0.0
                win_rate = 0.0
            
            results['mean_return'].append(mean_return)
            results['std_return'].append(std_return)
            results['sharpe'].append(sharpe)
            results['win_rate'].append(win_rate)
        
        return results
    
    def generate_sensitivity_curve_data(self, strategy_type: str) -> pd.DataFrame:
        """
        生成敏感性曲线数据
        
        Args:
            strategy_type: 策略类型 ('momentum', 'mean_reversion')
            
        Returns:
            curve_data: 敏感性曲线数据框
        """
        if strategy_type == 'momentum':
            results = self.analyze_momentum_window()
        elif strategy_type == 'mean_reversion':
            results = self.analyze_mean_reversion_window()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return pd.DataFrame(results)
    
    def generate_report(self) -> str:
        """
        生成敏感性分析报告
        
        Returns:
            report: 格式化的报告字符串
        """
        lines = []
        lines.append("=" * 100)
        lines.append("敏感性分析报告")
        lines.append("=" * 100)
        lines.append("")
        
        # Momentum策略敏感性分析
        lines.append("Momentum策略时间窗口敏感性:")
        lines.append("-" * 100)
        momentum_results = self.analyze_momentum_window()
        lines.append(f"{'窗口大小':<10} {'平均收益率':>12} {'标准差':>10} {'夏普比率':>10} {'胜率(%)':>10}")
        lines.append("-" * 100)
        
        for i, window in enumerate(momentum_results['window_size']):
            lines.append(f"{window:<10} {momentum_results['mean_return'][i]:>12.4f} {momentum_results['std_return'][i]:>10.4f} {momentum_results['sharpe'][i]:>10.4f} {momentum_results['win_rate'][i]:>10.2f}")
        
        lines.append("-" * 100)
        lines.append("")
        
        # MeanReversion策略敏感性分析
        lines.append("MeanReversion策略时间窗口敏感性:")
        lines.append("-" * 100)
        mean_reversion_results = self.analyze_mean_reversion_window()
        lines.append(f"{'窗口大小':<10} {'平均收益率':>12} {'标准差':>10} {'夏普比率':>10} {'胜率(%)':>10}")
        lines.append("-" * 100)
        
        for i, window in enumerate(mean_reversion_results['window_size']):
            lines.append(f"{window:<10} {mean_reversion_results['mean_return'][i]:>12.4f} {mean_reversion_results['std_return'][i]:>10.4f} {mean_reversion_results['sharpe'][i]:>10.4f} {mean_reversion_results['win_rate'][i]:>10.2f}")
        
        lines.append("-" * 100)
        lines.append("")
        
        # 敏感性曲线数据
        lines.append("Momentum策略敏感性曲线数据:")
        lines.append("-" * 100)
        momentum_curve = self.generate_sensitivity_curve_data('momentum')
        lines.append(momentum_curve.to_string(index=False))
        lines.append("-" * 100)
        lines.append("")
        
        lines.append("MeanReversion策略敏感性曲线数据:")
        lines.append("-" * 100)
        mean_reversion_curve = self.generate_sensitivity_curve_data('mean_reversion')
        lines.append(mean_reversion_curve.to_string(index=False))
        lines.append("-" * 100)
        lines.append("")
        
        lines.append("=" * 100)
        return "\n".join(lines)
