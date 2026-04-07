import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import AcademicVisualizationManager
import numpy as np


def generate_sample_visualizations():
    """生成示例可视化图表"""
    print("生成可视化图表...")
    
    # 初始化可视化管理器
    viz = AcademicVisualizationManager(output_dir='results/figures')
    
    # 生成模型比较图表
    models = ['LLM', 'Keyword', 'Hash', 'Momentum', 'MeanReversion']
    accuracies = [53.3, 52.7, 46.7, 48.0, 46.7]
    sample_sizes = [150, 150, 150, 150, 150]
    p_values = [0.977, 0.592, 0.999, 0.999, 0.999]
    errors = [5.0, 4.5, 6.0, 5.5, 6.5]
    
    path1 = viz.plot_model_comparison(
        models=models,
        accuracies=accuracies,
        sample_sizes=sample_sizes,
        p_values=p_values,
        errors=errors,
        title='策略性能比较',
        output_name='strategy_comparison'
    )
    print(f"生成策略比较图表: {path1}")
    
    # 生成消融研究图表
    conditions = ['Real News', 'Shuffled']
    values = [53.3, 49.1]
    p_vals_ablation = [0.977, 0.999]
    
    path2 = viz.plot_ablation_study(
        conditions=conditions,
        values=values,
        p_values=p_vals_ablation,
        title='消融研究 - 真实新闻 vs 打乱新闻'
    )
    print(f"生成消融研究图表: {path2}")
    
    # 生成风险分析图表
    path3 = viz.plot_risk_analysis(
        win_count=80,
        loss_count=70,
        win_return=10.5,
        loss_return=-8.3,
        win_rate=53.3,
        profit_loss_ratio=1.26,
        drawdown_events=[
            {'name': 'Market Crash', 'drawdown': -25.0, 'color': '#E64B35'},
            {'name': 'Regulatory News', 'drawdown': -15.0, 'color': '#F4A582'},
            {'name': 'Other', 'drawdown': -10.0, 'color': '#92C5DE'},
        ],
        max_drawdown_excluded=25.0,
        title='风险分析'
    )
    print(f"生成风险分析图表: {path3}")
    
    # 生成累积收益图表
    strategy_returns = {
        'LLM': np.random.normal(0.01, 0.05, 30).tolist(),
        'Keyword': np.random.normal(0.008, 0.04, 30).tolist(),
        'Hash': np.random.normal(0.005, 0.06, 30).tolist(),
        'Random': np.random.choice([-1, 1], 30).tolist(),
    }
    
    path4 = viz.plot_cumulative_returns_regime(strategy_returns)
    print(f"生成累积收益图表: {path4}")
    
    # 生成市场条件分析图表
    market_perf = {
        'LLM': {
            'Bull': {'win_rate': 65.0, 'mean_return': 1.2, 'count': 50},
            'Bear': {'win_rate': 40.0, 'mean_return': -0.8, 'count': 30},
            'Sideways': {'win_rate': 50.0, 'mean_return': 0.2, 'count': 70}
        },
        'Keyword': {
            'Bull': {'win_rate': 74.2, 'mean_return': 1.5, 'count': 50},
            'Bear': {'win_rate': 26.8, 'mean_return': -1.2, 'count': 30},
            'Sideways': {'win_rate': 40.6, 'mean_return': 0.1, 'count': 70}
        }
    }
    
    path5 = viz.plot_market_condition_heatmap_enhanced(market_perf)
    print(f"生成市场条件分析图表: {path5}")
    
    print("\n✓ 所有图表生成完成！")


if __name__ == '__main__':
    generate_sample_visualizations()