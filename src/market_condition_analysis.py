import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.strategies import HashStrategy, calculate_percentage_return
from src.llm_model import get_reproducible_llm
from src.strategy_momentum import MomentumStrategy
from src.strategy_meanreversion import MeanReversionStrategy


class MarketConditionAnalyzer:
    """市场条件分析模块
    
    功能：
    - 划分市场状态（牛市/熊市/震荡市）
    - 输出各策略在不同市场状态下的表现热力图
    """
    
    def __init__(self, aligned_df: pd.DataFrame):
        """
        初始化市场条件分析器
        
        Args:
            aligned_df: 对齐的数据框
        """
        self.aligned_df = aligned_df.copy()
        self.llm = get_reproducible_llm(prompt_version="PROMPT_V1", model="kimi-k2")
        self.strategies = {
            'LLM': 'llm',  # 特殊处理
            'Hash': HashStrategy(),
            'Momentum': MomentumStrategy(),
            'MeanReversion': MeanReversionStrategy()
        }
        self._analyze_market_conditions()
    
    def _analyze_market_conditions(self):
        """分析市场条件并划分市场状态"""
        # 计算真实收益率
        self.aligned_df['true_return'] = self.aligned_df.apply(
            lambda row: calculate_percentage_return(row['future_price'], row['trade_price']),
            axis=1
        )
        
        # 计算价格变化率（用于判断市场状态）
        # 使用true_return来判断市场状态
        self.aligned_df['price_change'] = self.aligned_df['true_return']
        
        # 划分市场状态
        # 简单的市场状态划分：
        # - 牛市：价格上涨 > 0.5%
        # - 熊市：价格下跌 > 0.5%
        # - 震荡市：价格变化在 ±0.5% 之间
        conditions = [
            (self.aligned_df['price_change'] > 0.005),  # 牛市
            (self.aligned_df['price_change'] < -0.005),  # 熊市
            ((self.aligned_df['price_change'] >= -0.005) & (self.aligned_df['price_change'] <= 0.005))  # 震荡市
        ]
        
        choices = ['Bull', 'Bear', 'Sideways']
        self.aligned_df['market_condition'] = np.select(conditions, choices, default='Sideways')
        
        # 为LLM策略生成信号（特殊处理）
        if 'LLM' in self.strategies:
            news_texts = self.aligned_df['news_text'].tolist()
            llm_results = self.llm.batch_predict(news_texts)
            llm_signals = [result['signal'] for result in llm_results]
            self.aligned_df['LLM_signal'] = llm_signals
            self.aligned_df['LLM_return'] = self.aligned_df['LLM_signal'] * self.aligned_df['true_return']
        
        # 为其他策略生成信号和收益率
        for strategy_name, strategy in self.strategies.items():
            if strategy_name == 'LLM':
                continue  # 已处理
                
            if strategy_name in ['Momentum', 'MeanReversion']:
                # 这些策略需要价格数据，使用run方法
                signals, _ = strategy.run(self.aligned_df)
                self.aligned_df[f'{strategy_name}_signal'] = signals
            else:
                # 这些策略需要新闻文本
                self.aligned_df[f'{strategy_name}_signal'] = self.aligned_df['news_text'].apply(
                    strategy.generate_signal
                )
            
            # 计算策略收益率
            self.aligned_df[f'{strategy_name}_return'] = (
                self.aligned_df[f'{strategy_name}_signal'] * self.aligned_df['true_return']
            )
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        获取各策略在不同市场状态下的表现
        
        Returns:
            performance: 各策略在不同市场状态下的表现
        """
        performance = {}
        market_conditions = ['Bull', 'Bear', 'Sideways']
        
        for strategy_name in self.strategies:
            performance[strategy_name] = {}
            for condition in market_conditions:
                subset = self.aligned_df[self.aligned_df['market_condition'] == condition]
                if len(subset) > 0:
                    returns = subset[f'{strategy_name}_return'].tolist()
                    mean_return = np.mean(returns)
                    std_return = np.std(returns) if len(returns) > 1 else 0.0
                    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) * 100
                    
                    performance[strategy_name][condition] = {
                        'mean_return': mean_return,
                        'std_return': std_return,
                        'win_rate': win_rate,
                        'count': len(subset)
                    }
                else:
                    performance[strategy_name][condition] = {
                        'mean_return': 0.0,
                        'std_return': 0.0,
                        'win_rate': 0.0,
                        'count': 0
                    }
        
        return performance
    
    def generate_heatmap_data(self, metric: str = 'mean_return') -> pd.DataFrame:
        """
        生成热力图数据
        
        Args:
            metric: 指标名称（'mean_return', 'win_rate'）
            
        Returns:
            heatmap_data: 热力图数据框
        """
        performance = self.get_strategy_performance()
        market_conditions = ['Bull', 'Bear', 'Sideways']
        
        data = []
        for strategy_name in self.strategies:
            row = {'Strategy': strategy_name}
            for condition in market_conditions:
                row[condition] = performance[strategy_name][condition][metric]
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_report(self) -> str:
        """
        生成市场条件分析报告
        
        Returns:
            report: 格式化的报告字符串
        """
        lines = []
        lines.append("=" * 100)
        lines.append("市场条件分析报告")
        lines.append("=" * 100)
        lines.append("")
        
        # 市场状态分布
        condition_counts = self.aligned_df['market_condition'].value_counts()
        total = len(self.aligned_df)
        lines.append("市场状态分布:")
        lines.append("-" * 50)
        for condition, count in condition_counts.items():
            percentage = (count / total) * 100
            lines.append(f"{condition}: {count} ({percentage:.2f}%)")
        lines.append("-" * 50)
        lines.append("")
        
        # 各策略在不同市场状态下的表现
        performance = self.get_strategy_performance()
        market_conditions = ['Bull', 'Bear', 'Sideways']
        
        for metric_name, metric_display in [('mean_return', '平均收益率'), ('win_rate', '胜率(%)')]:
            lines.append(f"各策略{metric_display}:")
            lines.append("-" * 100)
            lines.append(f"{'策略':<15} {'牛市':>10} {'熊市':>10} {'震荡市':>10}")
            lines.append("-" * 100)
            
            for strategy_name in self.strategies:
                row = f"{strategy_name:<15}"
                for condition in market_conditions:
                    value = performance[strategy_name][condition][metric_name]
                    if metric_name == 'mean_return':
                        row += f" {value:>10.4f}"
                    else:
                        row += f" {value:>10.2f}"
                lines.append(row)
            
            lines.append("-" * 100)
            lines.append("")
        
        # 热力图数据
        lines.append("热力图数据 (平均收益率):")
        lines.append("-" * 100)
        heatmap_data = self.generate_heatmap_data('mean_return')
        lines.append(heatmap_data.to_string(index=False))
        lines.append("-" * 100)
        lines.append("")
        
        lines.append("=" * 100)
        return "\n".join(lines)
