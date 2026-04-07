import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.strategies import HashStrategy, calculate_percentage_return
from src.llm_model import get_reproducible_llm


class ErrorAnalyzer:
    """错误分析模块
    
    功能：
    - 分类样本（Easy/Hard/LLM-special/Rule-special）
    - 输出错误分析报告（统计+案例）
    """
    
    def __init__(self, aligned_df: pd.DataFrame):
        """
        初始化错误分析器
        
        Args:
            aligned_df: 对齐的数据框
        """
        self.aligned_df = aligned_df.copy()
        self.llm = get_reproducible_llm(prompt_version="PROMPT_V1", model="kimi-k2")
        self.hash_strategy = HashStrategy()
        self._analyze_errors()
    
    def _analyze_errors(self):
        """分析错误并分类样本"""
        # 计算真实收益率
        self.aligned_df['true_return'] = self.aligned_df.apply(
            lambda row: calculate_percentage_return(row['future_price'], row['trade_price']),
            axis=1
        )
        
        # 生成策略信号
        # 使用LLM生成信号
        news_texts = self.aligned_df['news_text'].tolist()
        llm_results = self.llm.batch_predict(news_texts)
        llm_signals = [result['signal'] for result in llm_results]
        self.aligned_df['keyword_signal'] = llm_signals
        
        # 使用Hash策略生成信号
        self.aligned_df['hash_signal'] = self.aligned_df['news_text'].apply(
            self.hash_strategy.generate_signal
        )
        
        # 计算策略收益率
        self.aligned_df['keyword_return'] = self.aligned_df['keyword_signal'] * self.aligned_df['true_return']
        self.aligned_df['hash_return'] = self.aligned_df['hash_signal'] * self.aligned_df['true_return']
        
        # 计算策略是否正确
        self.aligned_df['keyword_correct'] = (self.aligned_df['keyword_return'] > 0).astype(int)
        self.aligned_df['hash_correct'] = (self.aligned_df['hash_return'] > 0).astype(int)
        
        # 分类样本
        self._classify_samples()
    
    def _classify_samples(self):
        """分类样本"""
        conditions = [
            # Easy: 两个策略都正确
            ((self.aligned_df['keyword_correct'] == 1) & (self.aligned_df['hash_correct'] == 1)),
            # Hard: 两个策略都错误
            ((self.aligned_df['keyword_correct'] == 0) & (self.aligned_df['hash_correct'] == 0)),
            # LLM-special: LLM正确，Hash错误
            ((self.aligned_df['keyword_correct'] == 1) & (self.aligned_df['hash_correct'] == 0)),
            # Hash-special: Hash正确，LLM错误
            ((self.aligned_df['keyword_correct'] == 0) & (self.aligned_df['hash_correct'] == 1))
        ]
        
        choices = ['Easy', 'Hard', 'LLM-special', 'Hash-special']
        self.aligned_df['sample_type'] = np.select(conditions, choices, default='Unknown')
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        获取错误分析统计
        
        Returns:
            statistics: 各样本类型的统计信息
        """
        stats = {}
        sample_types = ['Easy', 'Hard', 'LLM-special', 'Hash-special']
        
        for sample_type in sample_types:
            subset = self.aligned_df[self.aligned_df['sample_type'] == sample_type]
            if len(subset) > 0:
                stats[sample_type] = {
                    'count': len(subset),
                    'percentage': len(subset) / len(self.aligned_df) * 100,
                    'keyword_accuracy': subset['keyword_correct'].mean() * 100,
                    'hash_accuracy': subset['hash_correct'].mean() * 100,
                    'average_return': subset['true_return'].mean()
                }
            else:
                stats[sample_type] = {
                    'count': 0,
                    'percentage': 0.0,
                    'keyword_accuracy': 0.0,
                    'hash_accuracy': 0.0,
                    'average_return': 0.0
                }
        
        return stats
    
    def get_case_studies(self, sample_type: str, n_cases: int = 3) -> List[Dict]:
        """
        获取案例研究
        
        Args:
            sample_type: 样本类型
            n_cases: 案例数量
            
        Returns:
            cases: 案例列表
        """
        subset = self.aligned_df[self.aligned_df['sample_type'] == sample_type]
        cases = []
        
        for _, row in subset.head(n_cases).iterrows():
            case = {
                'news_text': row['news_text'][:200] + '...' if len(row['news_text']) > 200 else row['news_text'],
                'true_return': row['true_return'],
                'keyword_signal': row['keyword_signal'],
                'hash_signal': row['hash_signal'],
                'keyword_correct': row['keyword_correct'],
                'hash_correct': row['hash_correct']
            }
            cases.append(case)
        
        return cases
    
    def generate_report(self) -> str:
        """
        生成错误分析报告
        
        Returns:
            report: 格式化的报告字符串
        """
        lines = []
        lines.append("=" * 100)
        lines.append("错误分析报告")
        lines.append("=" * 100)
        lines.append("")
        
        # 统计信息
        stats = self.get_statistics()
        lines.append("样本分类统计:")
        lines.append("-" * 100)
        lines.append(f"{'样本类型':<15} {'数量':>8} {'占比(%)':>10} {'Keyword准确率(%)':>18} {'Hash准确率(%)':>16} {'平均收益率':>12}")
        lines.append("-" * 100)
        
        for sample_type, stat in stats.items():
            lines.append(f"{sample_type:<15} {stat['count']:>8} {stat['percentage']:>10.2f} {stat['keyword_accuracy']:>18.2f} {stat['hash_accuracy']:>16.2f} {stat['average_return']:>12.4f}")
        
        lines.append("-" * 100)
        lines.append("")
        
        # 案例研究
        sample_types = ['Easy', 'Hard', 'LLM-special', 'Hash-special']
        for sample_type in sample_types:
            cases = self.get_case_studies(sample_type, 2)
            if cases:
                lines.append(f"{sample_type}案例:")
                lines.append("-" * 50)
                for i, case in enumerate(cases, 1):
                    lines.append(f"案例 {i}:")
                    lines.append(f"  新闻: {case['news_text']}")
                    lines.append(f"  真实收益率: {case['true_return']:.4f}")
                    lines.append(f"  LLM信号: {case['keyword_signal']} ({'正确' if case['keyword_correct'] else '错误'})")
                    lines.append(f"  Hash信号: {case['hash_signal']} ({'正确' if case['hash_correct'] else '错误'})")
                    lines.append("")
                lines.append("-" * 50)
                lines.append("")
        
        lines.append("=" * 100)
        return "\n".join(lines)
