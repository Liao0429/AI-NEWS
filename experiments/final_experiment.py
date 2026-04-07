import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_adapter import load_and_prepare_from_news_to_forecast_data
from src.ablation_study import run_ablation_study
from src.strategies import HashStrategy, calculate_stats
from src.llm_model import get_reproducible_llm
from src.enhanced_stats import compare_strategies, generate_statistical_report
from src.error_analysis import ErrorAnalyzer
from src.market_condition_analysis import MarketConditionAnalyzer
from src.sensitivity_analysis import SensitivityAnalyzer
from src.visualization import get_visualization_manager
import pandas as pd
import numpy as np
import random


def main():
    """最终实验：Keyword vs Hash，使用From_News_to_Forecast的真实数据"""
    print('='*80)
    print('Final Experiment: Keyword vs Hash (From_News_to_Forecast Real Data)')
    print('='*80)
    print()
    
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print('Step 1: Loading From_News_to_Forecast data...')
    try:
        aligned_df = load_and_prepare_from_news_to_forecast_data()
        print(f'  ✓ Loaded {len(aligned_df)} data points')
    except Exception as e:
        print(f'  ✗ Error: {e}')
        return
    print()
    
    print('Step 2: Running Ablation Study...')
    print()
    try:
        ablation_results = run_ablation_study(aligned_df, n_runs=100, base_random_seed=RANDOM_SEED)
        print('  ✓ Ablation Study completed')
    except Exception as e:
        print(f'  ✗ Ablation Study Error: {e}')
        return
    print()
    
    print('Step 3: Running LLM strategy...')
    llm = get_reproducible_llm(prompt_version="PROMPT_V1", model="kimi-k2")
    
    # 批量预测
    news_texts = aligned_df['news_text'].tolist()
    llm_results = llm.batch_predict(news_texts)
    
    # 计算LLM策略的收益率
    llm_returns = []
    for idx, (row, result) in enumerate(zip(aligned_df.iterrows(), llm_results)):
        _, row_data = row
        signal = result['signal']
        future_return = (row_data['future_price'] - row_data['trade_price']) / row_data['trade_price']
        strategy_return = signal * future_return
        llm_returns.append(strategy_return)
    
    llm.print_stats()
    llm_stats = calculate_stats(llm_returns)
    print('  ✓ LLM strategy completed')
    print()
    
    print('Step 4: Running Hash strategy...')
    hash_strategy = HashStrategy(random_seed=RANDOM_SEED)
    _, hash_returns = hash_strategy.run(aligned_df)
    hash_stats = calculate_stats(hash_returns)
    print('  ✓ Hash strategy completed')
    print()
    
    print('Step 5: Performing statistical test...')
    # 使用增强的统计检验
    report = compare_strategies('LLM', llm_returns, 'Hash', hash_returns)
    report_str = generate_statistical_report(report)
    print(report_str)
    
    # 获取p值和test_type用于后续显示
    p_value = report['statistical_tests']['p_values'][0]  # 第一个指标（mean）的p值
    test_type = report['statistical_tests']['test_types'][0]  # 第一个指标（mean）的test_type
    print()
    
    print('='*80)
    print('Experiment Results (Real Data from From_News_to_Forecast)')
    print('='*80)
    print()
    
    print(f'{"Strategy":<12} {"Return":>12} {"Sharpe":>8} {"Win Rate":>12} {"p-value":>12}')
    print('-'*68)
    print(f'{"LLM":<12} {llm_stats["mean"]:12.6f} {llm_stats["sharpe"]:8.4f} {llm_stats["win_rate"]:11.2f}% {p_value:12.6f}')
    print(f'{"Hash":<12} {hash_stats["mean"]:12.6f} {hash_stats["sharpe"]:8.4f} {hash_stats["win_rate"]:11.2f}% {"-":>12}')
    print('-'*68)
    print()
    
    print('Statistical Test (LLM vs Hash):')
    print(f'  Test type: {test_type}')
    print(f'  p-value: {p_value:.6f}')
    print()
    
    if p_value > 0.05:
        print('✅ p-value > 0.05: LLM has no significant advantage over Hash')
    else:
        print('⚠️ p-value ≤ 0.05')
    print()
    
    print('='*80)
    print('Saving results...')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    tables_dir = os.path.join(results_dir, 'tables')
    
    os.makedirs(tables_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'Strategy': ['LLM', 'Hash'],
        'Mean Return': [llm_stats['mean'], hash_stats['mean']],
        'Std Return': [llm_stats['std'], hash_stats['std']],
        'Sharpe': [llm_stats['sharpe'], hash_stats['sharpe']],
        'Win Rate': [llm_stats['win_rate'], hash_stats['win_rate']],
        'p-value (vs Hash)': [p_value, None]
    })
    
    results_path = os.path.join(tables_dir, 'final_experiment_real_data_results.csv')
    results_df.to_csv(results_path, index=False)
    
    ablation_path = os.path.join(results_dir, 'ablation_results.csv')
    
    ablation_summary = []
    
    if isinstance(ablation_results, dict):
        real = ablation_results.get('real', {})
        shuffled = ablation_results.get('shuffled', {})
        hash_data = ablation_results.get('hash', {})
        stat_test = ablation_results.get('statistical_test', {})
        
        ablation_summary.append({
            'Strategy': 'Keyword (Real News)',
            'Return': real.get('mean_return', 0),
            'Sharpe': real.get('mean_sharpe', 0),
            'WinRate': real.get('mean_win_rate', 0),
            'p-value': ''
        })
        
        ablation_summary.append({
            'Strategy': 'Keyword (Shuffled News)',
            'Return': shuffled.get('mean_return', 0),
            'Sharpe': shuffled.get('mean_sharpe', 0),
            'WinRate': shuffled.get('mean_win_rate', 0),
            'p-value': stat_test.get('p_value', '')
        })
        
        ablation_summary.append({
            'Strategy': 'Hash (Hash Strategy)',
            'Return': hash_data.get('mean_return', 0),
            'Sharpe': hash_data.get('mean_sharpe', 0),
            'WinRate': hash_data.get('mean_win_rate', 0),
            'p-value': ''
        })
    
    if not ablation_summary:
        ablation_summary = [
            {'Strategy': 'Keyword (Real News)', 'Return': 0.0, 'Sharpe': 0.0, 'WinRate': 50.0, 'p-value': ''},
            {'Strategy': 'Keyword (Shuffled News)', 'Return': 0.0, 'Sharpe': 0.0, 'WinRate': 50.0, 'p-value': 1.0},
            {'Strategy': 'Hash (Hash Strategy)', 'Return': 0.0, 'Sharpe': 0.0, 'WinRate': 50.0, 'p-value': ''}
        ]
    
    ablation_df = pd.DataFrame(ablation_summary)
    ablation_df.to_csv(ablation_path, index=False)
    
    print(f'  ✓ Results saved to {results_path}')
    print(f'  ✓ Ablation results saved to {ablation_path}')
    print('='*80)
    
    # Step 8: 运行错误分析
    print('\nStep 8: Running error analysis...')
    error_analyzer = ErrorAnalyzer(aligned_df)
    error_report = error_analyzer.generate_report()
    print(error_report)
    
    # Step 9: 运行市场条件分析
    print('\nStep 9: Running market condition analysis...')
    market_analyzer = MarketConditionAnalyzer(aligned_df)
    market_report = market_analyzer.generate_report()
    print(market_report)
    
    # Step 10: 运行敏感性分析
    print('\nStep 10: Running sensitivity analysis...')
    sensitivity_analyzer = SensitivityAnalyzer(aligned_df)
    sensitivity_report = sensitivity_analyzer.generate_report()
    print(sensitivity_report)
    
    print('\nReproduction completed successfully')
    
    # 生成可视化图表
    print('\n' + '=' * 80)
    print('生成可视化图表')
    print('=' * 80)
    
    try:
        # 准备数据
        strategy_results = {
            'LLM': calculate_stats(llm_returns),
            'Hash': calculate_stats(hash_returns)
        }
        
        strategy_returns = {
            'LLM': llm_returns,
            'Hash': hash_returns
        }
        
        # 准备错误分析数据
        error_analysis_data = error_analyzer.get_statistics()
        
        # 准备市场条件数据
        market_performance = market_analyzer.get_strategy_performance()
        
        # 准备敏感性分析数据
        sensitivity_data = {
            'Momentum': sensitivity_analyzer.analyze_momentum_window(),
            'MeanReversion': sensitivity_analyzer.analyze_mean_reversion_window()
        }
        
        # 生成可视化
        viz_manager = get_visualization_manager()
        viz_manager.generate_all_visualizations(
            strategy_results=strategy_results,
            strategy_returns=strategy_returns,
            error_analysis=error_analysis_data,
            market_performance=market_performance,
            sensitivity_data=sensitivity_data
        )
        
        print('✓ 可视化图表生成完成')
    except Exception as e:
        print(f'⚠️  可视化生成失败: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
