"""
单资产详细结果生成脚本

生成4个资产（AAPL、MSFT、GOOGL、BTC-USD）的详细结果表格
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.dataset_adapter import DataProcessor
from src.strategies import KeywordStrategy, HashStrategy, RandomStrategy
from src.strategy_momentum import MomentumStrategy
from src.strategy_meanreversion import MeanReversionStrategy
from src.strategies import calculate_stats


def run_single_asset_experiment(ticker, sample_size=150):
    """
    对单个资产运行所有策略实验

    Args:
        ticker: 资产代码
        sample_size: 采样数量

    Returns:
        DataFrame: 实验结果
    """
    print(f"\n{'='*80}")
    print(f"处理资产: {ticker}")
    print('='*80)

    # 加载数据
    processor = DataProcessor()
    df = processor.load_from_news_to_forecast_data(ticker, sample_size)

    if len(df) == 0:
        print(f"⚠️  {ticker} 没有数据")
        return None

    # 计算true_return
    df['true_return'] = df['price_return']

    # 初始化策略
    strategies = {
        'Keyword': KeywordStrategy(),
        'Hash': HashStrategy(),
        'Random': RandomStrategy(),
        'Momentum': MomentumStrategy(),
        'MeanReversion': MeanReversionStrategy()
    }

    # 运行每个策略
    results = []
    for name, strategy in strategies.items():
        if name in ['Momentum', 'MeanReversion']:
            signals, returns = strategy.run(df)
        else:
            signals = df['news_text'].apply(strategy.generate_signal).tolist()
            pct_returns = df.apply(
                lambda row: (row['future_price'] - row['trade_price']) / row['trade_price'] if row['trade_price'] != 0 else 0,
                axis=1
            ).tolist()
            returns = [s * r for s, r in zip(signals, pct_returns)]

        stats = calculate_stats(returns)
        stats['Strategy'] = name
        stats['Ticker'] = ticker
        results.append(stats)

    return pd.DataFrame(results)


def generate_single_asset_results():
    """
    生成所有单资产的详细结果
    """
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD']
    all_results = []

    for ticker in tickers:
        result = run_single_asset_experiment(ticker, sample_size=150)
        if result is not None:
            all_results.append(result)

    # 合并结果
    df_all = pd.concat(all_results, ignore_index=True)

    # 保存完整结果
    os.makedirs('results/tables', exist_ok=True)
    output_path = 'results/tables/single_asset_results.csv'
    df_all.to_csv(output_path, index=False)
    print(f"\n✓ 结果已保存到: {output_path}")

    return df_all


def print_single_asset_tables(df):
    """
    打印单资产详细结果表格
    """
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        df_ticker = df[df['Ticker'] == ticker]

        print(f"\n{'='*100}")
        print(f"{ticker} 实验结果")
        print('='*100)

        # 创建表格
        table = df_ticker[['Strategy', 'mean', 'std', 'sharpe', 'win_rate', 'max_drawdown', 'calmar_ratio', 'sortino_ratio']].copy()
        table.columns = ['策略', '收益率', '标准差', '夏普比率', '胜率(%)', '最大回撤(%)', '卡玛比率', '索提诺比率']

        # 格式化数值
        table['收益率'] = table['收益率'].apply(lambda x: f"{x:.6f}")
        table['标准差'] = table['标准差'].apply(lambda x: f"{x:.6f}")
        table['夏普比率'] = table['夏普比率'].apply(lambda x: f"{x:.6f}")
        table['胜率(%)'] = table['胜率(%)'].apply(lambda x: f"{x:.2f}")
        table['最大回撤(%)'] = table['最大回撤(%)'].apply(lambda x: f"{x:.2f}")
        table['卡玛比率'] = table['卡玛比率'].apply(lambda x: f"{x:.6f}")
        table['索提诺比率'] = table['索提诺比率'].apply(lambda x: f"{x:.6f}")

        print(table.to_string(index=False))

        # 保存单个资产结果
        ticker_path = f'results/tables/{ticker}_results.csv'
        df_ticker.to_csv(ticker_path, index=False)
        print(f"✓ {ticker} 结果已保存到: {ticker_path}")


def print_summary_table(df):
    """
    打印汇总表格
    """
    print(f"\n{'='*100}")
    print("所有资产汇总")
    print('='*100)

    # 按策略汇总
    summary = df.groupby('Strategy').agg({
        'mean': ['mean', 'std'],
        'sharpe': ['mean', 'std'],
        'win_rate': ['mean', 'std']
    }).round(4)

    summary.columns = ['平均收益率', '收益率标准差', '平均夏普比率', '夏普比率标准差', '平均胜率', '胜率标准差']

    print(summary.to_string())

    # 保存汇总结果
    summary_path = 'results/tables/summary_results.csv'
    summary.to_csv(summary_path)
    print(f"\n✓ 汇总结果已保存到: {summary_path}")


if __name__ == '__main__':
    print("开始生成单资产详细结果...")
    print("="*100)

    # 生成所有结果
    df_all = generate_single_asset_results()

    # 打印单资产表格
    print_single_asset_tables(df_all)

    # 打印汇总表格
    print_summary_table(df_all)

    print("\n" + "="*100)
    print("✓ 单资产详细结果生成完成！")
    print("="*100)
