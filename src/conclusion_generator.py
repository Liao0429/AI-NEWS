import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import os


class ConclusionGenerator:
    """
    自动论文结论生成器
    
    输入：
    - final_experiment结果
    - ablation结果
    - robustness结果
    
    输出：
    results/conclusion.txt
    """
    
    def __init__(self):
        """
        初始化结论生成器
        """
        pass
    
    def load_final_experiment_results(self, filepath='results/final_table.csv'):
        """
        加载最终实验结果
        
        Args:
            filepath: 最终实验结果文件路径
            
        Returns:
            实验结果数据
        """
        df = pd.read_csv(filepath)
        
        def parse_numeric(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                x = x.strip()
                if x == '' or x == '-':
                    return np.nan
                try:
                    return float(x)
                except:
                    return np.nan
            return float(x)
        
        df['Return'] = df['Return'].apply(parse_numeric)
        df['p-value'] = df['p-value'].apply(parse_numeric)
        
        return df
    
    def load_ablation_results(self, filepath='results/ablation_results.csv'):
        """
        加载消融实验结果
        
        Args:
            filepath: 消融实验结果文件路径
            
        Returns:
            消融实验结果数据
        """
        df = pd.read_csv(filepath)
        return df
    
    def load_robustness_results(self, filepath='results/confidence_analysis.csv'):
        """
        加载鲁棒性实验结果
        
        Args:
            filepath: 鲁棒性实验结果文件路径
            
        Returns:
            鲁棒性实验结果数据
        """
        df = pd.read_csv(filepath)
        return df
    
    def analyze_main_finding(self, final_experiment_df):
        """
        分析主要发现
        
        Args:
            final_experiment_df: 最终实验结果DataFrame
            
        Returns:
            主要发现结论
        """
        llm_row = final_experiment_df[final_experiment_df['Strategy'] == 'LLM']
        ml_row = final_experiment_df[final_experiment_df['Strategy'] == 'ML']
        rule_row = final_experiment_df[final_experiment_df['Strategy'] == 'Rule']
        
        if llm_row.empty or ml_row.empty:
            return "LLM does not show significant advantage"
        
        llm_return = float(llm_row['Return'].iloc[0]) if pd.notna(llm_row['Return'].iloc[0]) else 0
        llm_p_value = float(llm_row['p-value'].iloc[0]) if pd.notna(llm_row['p-value'].iloc[0]) else 1.0
        
        ml_return = float(ml_row['Return'].iloc[0]) if pd.notna(ml_row['Return'].iloc[0]) else 0
        
        significance_threshold = 0.05
        advantage_threshold = 0.0001
        
        if llm_return > ml_return + advantage_threshold and llm_p_value < significance_threshold:
            return "LLM shows significant advantage"
        else:
            return "LLM does not show significant advantage"
    
    def analyze_ablation(self, ablation_df):
        """
        分析消融实验
        
        Args:
            ablation_df: 消融实验结果DataFrame
            
        Returns:
            消融实验结论
        """
        real_news_row = ablation_df[ablation_df['setup'] == 'Real News']
        shuffled_row = ablation_df[ablation_df['setup'] == 'Shuffled']
        
        if real_news_row.empty or shuffled_row.empty:
            return "LLM does not extract meaningful signal"
        
        real_return = real_news_row['mean_return'].iloc[0]
        shuffled_return = shuffled_row['mean_return'].iloc[0]
        p_value = real_news_row['p_value_vs_shuffled'].iloc[0] if pd.notna(real_news_row['p_value_vs_shuffled'].iloc[0]) else 1.0
        
        significance_threshold = 0.05
        advantage_threshold = 0.0001
        
        if real_return > shuffled_return + advantage_threshold and p_value < significance_threshold:
            return "LLM extracts meaningful signal"
        else:
            return "LLM does not extract meaningful signal"
    
    def analyze_robustness(self, robustness_df):
        """
        分析鲁棒性
        
        Args:
            robustness_df: 鲁棒性实验结果DataFrame
            
        Returns:
            鲁棒性结论
        """
        if len(robustness_df) < 2:
            return "Results are not robust"
        
        returns = robustness_df['Return'].values
        
        positive_count = sum(1 for r in returns if r > 0)
        all_positive = positive_count == len(returns)
        
        max_return = max(returns)
        min_return = min(returns)
        stability_ratio = abs(max_return - min_return) / (abs(max_return) + 1e-10) if max_return != 0 else 0
        
        stability_threshold = 0.5
        
        if all_positive and stability_ratio < stability_threshold:
            return "Results are robust"
        else:
            return "Results are not robust"
    
    def analyze_economic_significance(self, final_experiment_df):
        """
        分析经济显著性
        
        Args:
            final_experiment_df: 最终实验结果DataFrame
            
        Returns:
            经济显著性结论（如果适用）
        """
        llm_row = final_experiment_df[final_experiment_df['Strategy'] == 'LLM']
        
        if llm_row.empty:
            return None
        
        llm_return = float(llm_row['Return'].iloc[0]) if pd.notna(llm_row['Return'].iloc[0]) else 0
        
        cost_threshold = 0.002
        
        if llm_return < cost_threshold:
            return "Advantage disappears after cost"
        else:
            return None
    
    def generate_conclusion(self):
        """
        生成完整结论
        
        Returns:
            结论文本
        """
        print('='*120)
        print('自动论文结论生成器')
        print('='*120)
        print()
        
        print('Step 1: 加载实验结果...')
        final_experiment_df = self.load_final_experiment_results()
        ablation_df = self.load_ablation_results()
        robustness_df = self.load_robustness_results()
        print('  ✓ 完成')
        print()
        
        print('Step 2: 分析主要发现...')
        main_finding = self.analyze_main_finding(final_experiment_df)
        print('  ✓ 完成')
        print()
        
        print('Step 3: 分析消融实验...')
        ablation = self.analyze_ablation(ablation_df)
        print('  ✓ 完成')
        print()
        
        print('Step 4: 分析鲁棒性...')
        robustness = self.analyze_robustness(robustness_df)
        print('  ✓ 完成')
        print()
        
        print('Step 5: 分析经济显著性...')
        economic_significance = self.analyze_economic_significance(final_experiment_df)
        print('  ✓ 完成')
        print()
        
        print('Step 6: 生成结论文件...')
        conclusion_text = self._format_conclusion(main_finding, ablation, robustness, economic_significance)
        output_path = self._save_conclusion(conclusion_text)
        print('  ✓ 完成')
        print()
        
        print('='*120)
        print('✓ 结论生成完成！')
        print('='*120)
        print()
        print('结论内容：')
        print(conclusion_text)
        
        return conclusion_text, output_path
    
    def _format_conclusion(self, main_finding, ablation, robustness, economic_significance):
        """
        格式化结论
        
        Args:
            main_finding: 主要发现
            ablation: 消融实验结论
            robustness: 鲁棒性结论
            economic_significance: 经济显著性结论
            
        Returns:
            格式化后的结论文本
        """
        lines = []
        
        lines.append('1. Main Finding:')
        lines.append(f'    "{main_finding}"')
        lines.append('')
        
        lines.append('2. Ablation:')
        lines.append(f'    "{ablation}"')
        lines.append('')
        
        lines.append('3. Robustness:')
        lines.append(f'    "{robustness}"')
        lines.append('')
        
        if economic_significance is not None:
            lines.append('4. Economic Significance:')
            lines.append(f'    "{economic_significance}"')
        
        return '\n'.join(lines)
    
    def _save_conclusion(self, conclusion_text, output_path='results/conclusion.txt'):
        """
        保存结论文件
        
        Args:
            conclusion_text: 结论文本
            output_path: 输出文件路径
            
        Returns:
            输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(conclusion_text)
        
        print(f'  - 结论保存到: {output_path}')
        
        return output_path


def generate_conclusion():
    """
    生成论文结论的便捷函数
    
    Returns:
        结论文本和输出文件路径
    """
    generator = ConclusionGenerator()
    conclusion_text, output_path = generator.generate_conclusion()
    return conclusion_text, output_path


if __name__ == '__main__':
    print('生成论文结论...')
    conclusion_text, output_path = generate_conclusion()
